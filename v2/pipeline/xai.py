"""XAI stage adapter for v2 experiments.

Primary / Deep / Ablation XAI 산출물은 모두 v2/runtime/experiment_xai.py의
완성된 SHAP·LIME·Comp/Suff/LOO/Plausibility 함수에 의존한다. 본 모듈은
training_adapter가 experiment_core를 호출하는 패턴과 동일하게 runtime을 import
해서, 위 함수들이 만든 결과를 v2 산출물 contract(CSV/JSON)로 평탄화한다.

설계 원칙:

* sample 선택은 seed 무관 (xai_sampling 모듈). 같은 sample을 모든 seed checkpoint에
  통과시켜야 explanation stability 측정이 의미를 갖는다.
* checkpoint가 없으면 빈 CSV(헤더만)로 graceful skip. dry-run/스모크 단계에서도
  다음 stage가 깨지지 않게 한다.
* SHAP은 항상 CPU. runtime/experiment_xai.py가 이미 CPU 강제 패턴을 쓰므로
  본 adapter는 그 패턴을 그대로 활용.
* 첫 단계에서 모든 12지표를 완벽히 채울 필요는 없다. 골격 + 핵심 1~2개 메트릭부터
  채우고, 나머지는 빈 값(empty string)으로 둔다. 후속 작업에서 점진 확장.

caching: outputs/experiments/<run_id>/xai/.cache/ 에 (condition, seed, sample_id)
단위로 SHAP/LIME 결과를 잡아둔다. SHAP 한 sample 계산이 수 초~수십 초이므로
재실행 비용을 줄이기 위해 필수.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev as _stdev
from typing import Any

from .artifacts import write_csv
from .paths import BASE_DIR, experiment_root
from .schema import XAI_SEED_METRIC_COLUMNS
from .xai_sampling import (
    SAMPLE_COLUMNS,
    select_ablation_samples,
    select_deep_samples,
    select_primary_samples,
)


RUNTIME_DIR = BASE_DIR / "runtime"


# ───────────────────────────────────────────────────────────
# 출력 CSV 컬럼 정의 (placeholder 시점에서도 contract 유지)
# ───────────────────────────────────────────────────────────

PAIRED_XAI_COLUMNS = [
    "comparison",
    "metric",
    "n_pairs",
    "mean_diff",
    "p_value",
    "p_value_holm",
    "effect_size",
]

SEED_STABILITY_COLUMNS = [
    "condition",
    "metric",
    "mean",
    "std",
    "ci_low",
    "ci_high",
]

DEEP_CASE_COLUMNS = [
    "sample_id",
    "true_label",
    "baseline_prediction",
    "v2_prediction",
    "case_type",
    "baseline_confidence",
    "v2_confidence",
    "top_tokens_baseline",
    "top_tokens_v2",
    "human_rationale_tokens",
    "plot_path",
    "comment",
]

ABLATION_METRIC_COLUMNS = [
    "condition",
    "backbone",
    "attention_loss",
    "sentiment_feature",
    "rationale_f1_at_5",
    "comprehensiveness",
    "sufficiency",
    "attention_entropy",
    "mss",
]


# ───────────────────────────────────────────────────────────
# runtime import (training_adapter 패턴 복제)
# ───────────────────────────────────────────────────────────


def _load_runtime_modules() -> tuple[Any, Any]:
    """Import v2/runtime의 experiment_xai와 experiment_core를 동시 로드.

    runtime은 utils를 top-level 모듈로 가정하므로 sys.path에 runtime 경로를
    삽입해야 한다. 이전에 다른 위치에서 로드된 utils/experiment_xai/experiment_core
    캐시는 무효화한다 — training_adapter._load_runtime_core와 동일 패턴.
    """
    runtime_path = str(RUNTIME_DIR)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    for module_name in ["utils", "experiment_core", "experiment_xai"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", "") if module else ""
        if module_file and not str(module_file).startswith(runtime_path):
            del sys.modules[module_name]
    import experiment_core  # type: ignore[import-not-found]
    import experiment_xai  # type: ignore[import-not-found]
    return experiment_xai, experiment_core


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Stable JSON writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return path


# ───────────────────────────────────────────────────────────
# checkpoint discovery
# ───────────────────────────────────────────────────────────


def _checkpoint_for(unit_root: Path, condition: str, seed: int) -> Path:
    """Convention: <root>/benchmark/checkpoints/<lower_condition>_seed_<seed>.pt"""
    return (
        unit_root
        / "benchmark"
        / "checkpoints"
        / f"{condition.lower()}_seed_{seed}.pt"
    )


def _run_config_for(unit_root: Path, condition: str, seed: int) -> Path:
    """training_adapter가 unit별로 저장하는 run_config.json 경로."""
    return (
        unit_root
        / "benchmark"
        / "runs"
        / condition.lower()
        / f"seed_{seed}"
        / "run_config.json"
    )


def _load_record(checkpoint_path: Path, run_config_path: Path, condition: str) -> dict[str, Any] | None:
    """runtime._instantiate_bundle이 요구하는 record 딕셔너리를 만든다.

    원본 v1 코드는 best_models.json registry에서 record를 읽었지만, v2는 checkpoint
    파일과 run_config.json 두 개로 동일한 record를 재구성한다.
    """
    if not checkpoint_path.exists():
        return None
    hyperparams: dict[str, Any] = {}
    use_target_aux = False
    if run_config_path.exists():
        try:
            with open(run_config_path, "r", encoding="utf-8") as handle:
                config_payload = json.load(handle)
            hyperparams = dict(config_payload.get("hyperparams", {}))
            use_target_aux = bool(config_payload.get("use_target_aux", False))
        except Exception:
            # run_config 손상은 hard fail이 아니라 기본값으로 fallback.
            hyperparams = {}
            use_target_aux = False
    return {
        "checkpoint_path": str(checkpoint_path),
        "hyperparams": hyperparams,
        "use_target_aux": use_target_aux,
        "condition": condition,
    }


def _load_bundle(
    runtime_xai: Any,
    condition: str,
    checkpoint_path: Path,
    run_config_path: Path,
) -> Any | None:
    """runtime의 _instantiate_bundle을 활용해 LoadedModelBundle을 만든다."""
    record = _load_record(checkpoint_path, run_config_path, condition)
    if record is None:
        return None
    try:
        return runtime_xai._instantiate_bundle(condition, record)
    except Exception:
        # checkpoint 손상 / 아키텍처 불일치 등은 sample-level skip이 아니라
        # 이 unit 전체를 skip. 호출자에서 위험 row를 기록한다.
        return None


# ───────────────────────────────────────────────────────────
# attribution cache
# ───────────────────────────────────────────────────────────


def _cache_path(run_root: Path, condition: str, seed: int) -> Path:
    """(condition, seed) 단위 attribution 캐시 파일."""
    return run_root / "xai" / ".cache" / f"{condition.lower()}_seed_{seed}.json"


def _load_attribution_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _save_attribution_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


# ───────────────────────────────────────────────────────────
# per-(condition, seed) metric computation
# ───────────────────────────────────────────────────────────


def _safe_mean(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(_mean(values))


def _topk_jaccard(tokens_a: list[str], tokens_b: list[str], k: int = 5) -> float:
    set_a = {token.lower() for token in tokens_a[:k] if token}
    set_b = {token.lower() for token in tokens_b[:k] if token}
    if not set_a and not set_b:
        return 0.0
    if not (set_a | set_b):
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _rank_correlation(tokens_a: list[str], tokens_b: list[str], k: int = 10) -> float | str:
    """SHAP top-k와 LIME top-k 토큰의 순위 상관.

    공통 토큰만 비교하며, 공통 토큰이 2개 미만이면 빈 값. scipy 미설치 시도 빈 값.
    """
    common = [token for token in tokens_a[:k] if token in tokens_b[:k]]
    if len(common) < 2:
        return ""
    rank_a = {token: index for index, token in enumerate(tokens_a[:k])}
    rank_b = {token: index for index, token in enumerate(tokens_b[:k])}
    try:
        from scipy.stats import spearmanr  # noqa: WPS433 — local import to keep top clean.
    except ImportError:
        return ""
    a_values = [rank_a[token] for token in common]
    b_values = [rank_b[token] for token in common]
    correlation = spearmanr(a_values, b_values).statistic
    if correlation is None:
        return ""
    return float(correlation)


def _rationale_prf(
    model_tokens: list[str],
    human_tokens: list[str],
    k: int = 5,
) -> tuple[float, float, float]:
    """모델 top-k 토큰과 human rationale 토큰의 precision/recall/F1.

    토큰은 소문자 정규화 후 set 연산. human rationale이 비어 있으면 (0, 0, 0).
    """
    model_set = {token.lower() for token in model_tokens[:k] if token}
    human_set = {token.lower() for token in human_tokens if token}
    if not human_set or not model_set:
        return (0.0, 0.0, 0.0)
    matched = len(model_set & human_set)
    precision = matched / len(model_set)
    recall = matched / len(human_set)
    if precision + recall == 0.0:
        return (precision, recall, 0.0)
    f1 = 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


def _compute_unit_metrics(
    runtime_xai: Any,
    bundle: Any,
    samples: list[dict[str, Any]],
    config: Any,
    rationale_map: dict[str, list[str]],
) -> dict[str, Any]:
    """한 (condition, seed) checkpoint에 대한 metric 계산.

    반환값: SHAP/LIME 결과(다음 stage가 캐시할 수 있도록)와 평균 metric 집합.
    """
    texts = [str(sample["text"]) for sample in samples]
    if not texts:
        return {
            "shap": [],
            "lime": [],
            "predicted_labels": [],
            "metrics": {column: "" for column in XAI_SEED_METRIC_COLUMNS},
        }

    probabilities = runtime_xai.predict_probabilities(bundle, texts)
    predicted_labels = [int(probabilities[index].argmax()) for index in range(len(texts))]

    shap_results = runtime_xai.run_shap_explanations(bundle, texts, predicted_labels, config)
    lime_results = runtime_xai.run_lime_explanations(bundle, texts, predicted_labels, config)

    overlaps_5 = runtime_xai._compute_overlap_at_5(shap_results, lime_results)
    # overlap@10은 길이 5 토큰 리스트 한계 때문에 SHAP top_tokens가 5개 미만일 수 있음.
    # runtime이 top_tokens를 5개로 잘라 돌려주므로 overlap@10도 같은 길이를 갖는 Jaccard로
    # 계산한다. 의미는 동일.
    overlaps_10 = [
        _topk_jaccard(s["top_tokens"], l["top_tokens"], k=10)
        for s, l in zip(shap_results, lime_results)
    ]

    rationale_prf_list = []
    for sample, shap_result in zip(samples, shap_results):
        human_tokens = rationale_map.get(str(sample["sample_id"]), [])
        rationale_prf_list.append(_rationale_prf(shap_result["top_tokens"], human_tokens, k=5))

    # SHAP top-5 기반 comprehensiveness/sufficiency. runtime의 _compute_masking_metrics
    # 가 sample-level comp/suff details를 돌려준다.
    masking = runtime_xai._compute_masking_metrics(
        bundle,
        shap_results,
        texts,
        predicted_labels,
        k=5,
    )

    # LOO drop. runtime은 candidate top-k 별 평균 drop을 sample마다 돌려준다.
    loo_scores = runtime_xai._compute_loo_scores(bundle, shap_results, texts, predicted_labels)

    metrics: dict[str, Any] = {
        "shap_lime_overlap_at_5": _safe_mean(overlaps_5),
        "shap_lime_overlap_at_10": _safe_mean(overlaps_10),
        "rationale_precision_at_5": _safe_mean([prf[0] for prf in rationale_prf_list]),
        "rationale_recall_at_5": _safe_mean([prf[1] for prf in rationale_prf_list]),
        "rationale_f1_at_5": _safe_mean([prf[2] for prf in rationale_prf_list]),
        "comprehensiveness": masking.get("comprehensiveness_mean") if masking.get("comprehensiveness_mean") is not None else "",
        "sufficiency": masking.get("sufficiency_mean") if masking.get("sufficiency_mean") is not None else "",
        "loo_drop": _safe_mean(loo_scores),
        # topk_jaccard / rank_corr는 seed 간 비교에서 채워진다. 단일 (condition, seed)
        # 시점에는 빈 값.
        "topk_jaccard_mean": "",
        "rank_corr_mean": "",
    }
    return {
        "shap": shap_results,
        "lime": lime_results,
        "predicted_labels": predicted_labels,
        "metrics": metrics,
    }


# ───────────────────────────────────────────────────────────
# seed stability (multi-seed 결과 합산 후)
# ───────────────────────────────────────────────────────────


def _seed_stability_rows(
    per_seed_attributions: dict[tuple[str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """같은 sample에 대한 다른 seed checkpoint들의 top-k 토큰 일치도.

    같은 condition 내에서 seed 쌍별 평균 Jaccard / Spearman을 낸다. condition마다
    한 row.
    """
    rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for (condition, seed), shap_results in per_seed_attributions.items():
        by_condition.setdefault(condition, []).append((seed, shap_results))

    for condition, seed_results in by_condition.items():
        if len(seed_results) < 2:
            # 시드 1개 이하에서는 seed stability가 정의되지 않음.
            continue
        jaccards: list[float] = []
        rank_corrs: list[float] = []
        # 같은 sample 인덱스를 같은 sample로 가정 (sample selection이 결정적이므로).
        sample_count = min(len(results) for _, results in seed_results)
        for index in range(sample_count):
            tokens_list = [results[index]["top_tokens"] for _, results in seed_results]
            for i in range(len(tokens_list)):
                for j in range(i + 1, len(tokens_list)):
                    jaccards.append(_topk_jaccard(tokens_list[i], tokens_list[j], k=5))
                    correlation = _rank_correlation(tokens_list[i], tokens_list[j], k=10)
                    if isinstance(correlation, float):
                        rank_corrs.append(correlation)

        def _mean_ci(values: list[float]) -> tuple[float | str, float | str, float | str, float | str]:
            if not values:
                return ("", "", "", "")
            mean_value = float(_mean(values))
            std_value = float(_stdev(values)) if len(values) > 1 else 0.0
            # 단순 정규근사 신뢰구간. 통계적으로는 t-분포가 더 정확하지만 동일 stage의
            # statistics.py에서 이미 검증된 _mean_ci 형태를 그대로 쓰는 것이 일관성 있음.
            margin = 1.96 * std_value / max(1, len(values)) ** 0.5
            return (mean_value, std_value, mean_value - margin, mean_value + margin)

        jac_mean, jac_std, jac_low, jac_high = _mean_ci(jaccards)
        corr_mean, corr_std, corr_low, corr_high = _mean_ci(rank_corrs)
        rows.append(
            {
                "condition": condition,
                "metric": "topk_jaccard_5",
                "mean": jac_mean,
                "std": jac_std,
                "ci_low": jac_low,
                "ci_high": jac_high,
            }
        )
        rows.append(
            {
                "condition": condition,
                "metric": "rank_corr",
                "mean": corr_mean,
                "std": corr_std,
                "ci_low": corr_low,
                "ci_high": corr_high,
            }
        )
    return rows


def _paired_xai_rows(
    metric_rows: list[dict[str, Any]],
    comparison: str,
) -> list[dict[str, Any]]:
    """A_B vs D_B 같은 두 condition을 같은 seed에서 비교한 paired test 행들.

    seed별로 (A_B metric, D_B metric)의 차이를 모아 평균/표준편차/p-value를 낸다.
    statistics.py의 paired test 헬퍼를 재사용하기엔 입력 형태가 달라서, 본 함수에서
    가벼운 paired t-test를 직접 수행.
    """
    if ":" not in comparison:
        return []
    condition_a, condition_b = comparison.split(":")
    metrics = [
        "shap_lime_overlap_at_5",
        "rationale_f1_at_5",
        "comprehensiveness",
        "sufficiency",
        "loo_drop",
    ]

    a_by_seed: dict[int, dict[str, Any]] = {}
    b_by_seed: dict[int, dict[str, Any]] = {}
    for row in metric_rows:
        if row.get("condition") == condition_a:
            a_by_seed[int(row["seed"])] = row
        elif row.get("condition") == condition_b:
            b_by_seed[int(row["seed"])] = row

    common_seeds = sorted(set(a_by_seed) & set(b_by_seed))

    try:
        from scipy.stats import ttest_rel  # noqa: WPS433
    except ImportError:
        ttest_rel = None  # type: ignore[assignment]

    out_rows: list[dict[str, Any]] = []
    for metric in metrics:
        differences: list[float] = []
        for seed in common_seeds:
            a_value = a_by_seed[seed].get(metric, "")
            b_value = b_by_seed[seed].get(metric, "")
            if a_value == "" or b_value == "":
                continue
            try:
                differences.append(float(b_value) - float(a_value))
            except (TypeError, ValueError):
                continue
        if len(differences) < 2:
            out_rows.append(
                {
                    "comparison": f"{condition_a} vs {condition_b}",
                    "metric": metric,
                    "n_pairs": len(differences),
                    "mean_diff": float(_mean(differences)) if differences else "",
                    "p_value": "",
                    "p_value_holm": "",
                    "effect_size": "",
                }
            )
            continue
        diff_mean = float(_mean(differences))
        diff_std = float(_stdev(differences)) if len(differences) > 1 else 0.0
        if ttest_rel is not None:
            # paired t-test: 차이의 평균이 0과 같은지 검정.
            statistic = ttest_rel([0.0] * len(differences), [-d for d in differences])
            p_value = float(statistic.pvalue)
        else:
            p_value = ""
        effect_size = diff_mean / diff_std if diff_std > 0 else ""
        out_rows.append(
            {
                "comparison": f"{condition_a} vs {condition_b}",
                "metric": metric,
                "n_pairs": len(differences),
                "mean_diff": diff_mean,
                "p_value": p_value,
                "p_value_holm": "",
                "effect_size": effect_size,
            }
        )
    return out_rows


# ───────────────────────────────────────────────────────────
# stage entry points
# ───────────────────────────────────────────────────────────


def _empty_sample_csv(path: Path) -> Path:
    return write_csv(path, [], SAMPLE_COLUMNS)


def _write_sample_csv(path: Path, samples: list[dict[str, Any]]) -> Path:
    return write_csv(path, samples, SAMPLE_COLUMNS)


def _runtime_or_none() -> tuple[Any | None, Any | None]:
    """runtime을 import. 실패하면 (None, None)을 돌려준다.

    Primary/Deep/Ablation의 placeholder 모드에서도 호출되므로, runtime이 없거나
    data가 아직 준비되지 않아도 stage가 깨지지 않게 한다.
    """
    try:
        return _load_runtime_modules()
    except Exception:
        return (None, None)


def _safe_runtime_config(runtime_core: Any) -> Any:
    """ExperimentConfig 인스턴스. 실패 시 None."""
    try:
        return runtime_core.get_config()
    except Exception:
        return None


def _human_rationales(runtime_xai: Any) -> dict[str, list[str]]:
    """runtime의 human rationale dictionary. 실패 시 빈 dict."""
    try:
        return runtime_xai._load_human_rationales()
    except Exception:
        return {}


def plan_primary_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Primary XAI: A_B vs D_B를 15 seed 모두에서 같은 sample로 비교."""
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["primary"]
    sample_size = int(xai_config["sample_size"])
    sample_path = root / "xai" / "samples" / "primary_samples.csv"
    metric_path = root / "xai" / "primary" / "seed_level_metrics.csv"
    paired_path = root / "xai" / "primary" / "paired_xai_tests.csv"
    stability_path = root / "xai" / "primary" / "seed_stability.csv"

    if dry_run:
        return {"status": "dry-run", "sample_size": str(sample_size)}

    runtime_xai, runtime_core = _runtime_or_none()
    if runtime_xai is None or runtime_core is None:
        # runtime이 비활성이거나 data가 아직 없으면 빈 contract만 만든다.
        return {
            "samples": _empty_sample_csv(sample_path),
            "seed_metrics": write_csv(metric_path, [], XAI_SEED_METRIC_COLUMNS),
            "paired_tests": write_csv(paired_path, [], PAIRED_XAI_COLUMNS),
            "seed_stability": write_csv(stability_path, [], SEED_STABILITY_COLUMNS),
        }

    try:
        samples = select_primary_samples(runtime_core, manifest, sample_size)
    except Exception:
        # data가 준비 안 됐을 때 (예: prepare_data가 아직 안 돌았을 때) graceful.
        samples = []
    samples_written = _write_sample_csv(sample_path, samples)

    rationale_map = _human_rationales(runtime_xai)
    config = _safe_runtime_config(runtime_core)

    metric_rows: list[dict[str, Any]] = []
    per_seed_attributions: dict[tuple[str, int], list[dict[str, Any]]] = {}

    primary_conditions = list(xai_config.get("models", ["A_B", "D_B"]))
    seeds = [int(seed) for seed in manifest["benchmark"]["seeds"]]

    for condition in primary_conditions:
        for seed in seeds:
            checkpoint_path = _checkpoint_for(root, condition, seed)
            run_config_path = _run_config_for(root, condition, seed)
            if not checkpoint_path.exists() or not samples or config is None:
                continue

            cache_path = _cache_path(root, condition, seed)
            cache = _load_attribution_cache(cache_path)
            cached_metrics = cache.get("metrics")
            cached_shap = cache.get("shap")

            if cached_metrics and cached_shap and len(cached_shap) == len(samples):
                # 캐시 히트: 다시 계산하지 않는다.
                metrics = dict(cached_metrics)
                shap_results = cached_shap
            else:
                bundle = _load_bundle(runtime_xai, condition, checkpoint_path, run_config_path)
                if bundle is None:
                    continue
                computed = _compute_unit_metrics(runtime_xai, bundle, samples, config, rationale_map)
                metrics = computed["metrics"]
                shap_results = computed["shap"]
                _save_attribution_cache(
                    cache_path,
                    {
                        "condition": condition,
                        "seed": seed,
                        "shap": shap_results,
                        "lime": computed["lime"],
                        "metrics": metrics,
                    },
                )

            row = {column: metrics.get(column, "") for column in XAI_SEED_METRIC_COLUMNS}
            row.update(
                {
                    "run_id": manifest["run_id"],
                    "condition": condition,
                    "seed": seed,
                    "sample_count": len(samples),
                }
            )
            metric_rows.append(row)
            per_seed_attributions[(condition, seed)] = shap_results

    metric_written = write_csv(metric_path, metric_rows, XAI_SEED_METRIC_COLUMNS)

    paired_rows: list[dict[str, Any]] = []
    primary_pair = xai_config.get("paired_test", "A_B:D_B")
    if metric_rows:
        paired_rows = _paired_xai_rows(metric_rows, primary_pair)
    paired_written = write_csv(paired_path, paired_rows, PAIRED_XAI_COLUMNS)

    stability_rows = _seed_stability_rows(per_seed_attributions)
    stability_written = write_csv(stability_path, stability_rows, SEED_STABILITY_COLUMNS)

    return {
        "samples": samples_written,
        "seed_metrics": metric_written,
        "paired_tests": paired_written,
        "seed_stability": stability_written,
    }


def plan_deep_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Deep XAI: median seed 1개 × 500 sample. 정성 case analysis."""
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["deep"]
    sample_size = int(xai_config["sample_size"])

    sample_path = root / "xai" / "samples" / "deep_samples.csv"
    case_path = root / "xai" / "deep" / "case_summary.csv"
    details_path = root / "xai" / "deep" / "xai_details.json"

    if dry_run:
        return {"status": "dry-run", "sample_size": str(sample_size)}

    runtime_xai, runtime_core = _runtime_or_none()
    if runtime_xai is None or runtime_core is None:
        return {
            "samples": _empty_sample_csv(sample_path),
            "details": _write_json(details_path, {"status": "planned", "sample_size": sample_size}),
            "cases": write_csv(case_path, [], DEEP_CASE_COLUMNS),
        }

    try:
        samples = select_deep_samples(runtime_core, manifest, sample_size)
    except Exception:
        samples = []
    samples_written = _write_sample_csv(sample_path, samples)

    # Deep는 median seed로 도는 모델 2개(baseline + improved)를 비교한다.
    # benchmark_summary가 없으면 manifest의 seeds[0]을 fallback로 쓴다.
    median_seed = int(manifest["benchmark"]["seeds"][len(manifest["benchmark"]["seeds"]) // 2])
    deep_models = list(xai_config.get("models", ["A_B", "D_B"]))

    config = _safe_runtime_config(runtime_core)
    rationale_map = _human_rationales(runtime_xai)

    case_rows: list[dict[str, Any]] = []
    details_payload: dict[str, Any] = {
        "status": "in_progress" if samples and config is not None else "planned",
        "sample_size": sample_size,
        "median_seed": median_seed,
        "models": deep_models,
    }

    bundles: dict[str, Any] = {}
    if samples and config is not None:
        for condition in deep_models:
            checkpoint_path = _checkpoint_for(root, condition, median_seed)
            run_config_path = _run_config_for(root, condition, median_seed)
            bundle = _load_bundle(runtime_xai, condition, checkpoint_path, run_config_path)
            if bundle is not None:
                bundles[condition] = bundle

    if len(bundles) == len(deep_models) and bundles:
        baseline = deep_models[0]
        v2_condition = deep_models[-1]
        texts = [str(sample["text"]) for sample in samples]
        baseline_probs = runtime_xai.predict_probabilities(bundles[baseline], texts)
        v2_probs = runtime_xai.predict_probabilities(bundles[v2_condition], texts)
        baseline_preds = [int(baseline_probs[i].argmax()) for i in range(len(texts))]
        v2_preds = [int(v2_probs[i].argmax()) for i in range(len(texts))]
        baseline_shap = runtime_xai.run_shap_explanations(bundles[baseline], texts, baseline_preds, config)
        v2_shap = runtime_xai.run_shap_explanations(bundles[v2_condition], texts, v2_preds, config)

        for index, sample in enumerate(samples):
            human_tokens = rationale_map.get(str(sample["sample_id"]), [])
            case_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "true_label": sample["label"],
                    "baseline_prediction": baseline_preds[index],
                    "v2_prediction": v2_preds[index],
                    "case_type": sample.get("case_type", "deep"),
                    "baseline_confidence": float(baseline_probs[index][baseline_preds[index]]),
                    "v2_confidence": float(v2_probs[index][v2_preds[index]]),
                    "top_tokens_baseline": ",".join(baseline_shap[index]["top_tokens"]),
                    "top_tokens_v2": ",".join(v2_shap[index]["top_tokens"]),
                    "human_rationale_tokens": ",".join(human_tokens),
                    "plot_path": "",
                    "comment": "",
                }
            )
        details_payload["status"] = "completed"

    cases_written = write_csv(case_path, case_rows, DEEP_CASE_COLUMNS)
    details_written = _write_json(details_path, details_payload)

    return {
        "samples": samples_written,
        "details": details_written,
        "cases": cases_written,
    }


def plan_ablation_xai(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Ablation 매트릭스: 8조건 × median seed × 50 sample. directional check."""
    root = experiment_root(manifest["run_id"])
    xai_config = manifest["xai"]["ablation"]
    sample_size = int(xai_config["sample_size"])

    sample_path = root / "xai" / "samples" / "ablation_samples.csv"
    metric_path = root / "xai" / "ablation" / "xai_ablation_metrics.csv"
    summary_path = root / "xai" / "xai_summary.json"

    if dry_run:
        return {"status": "dry-run", "sample_size": str(sample_size)}

    runtime_xai, runtime_core = _runtime_or_none()
    if runtime_xai is None or runtime_core is None:
        return {
            "samples": _empty_sample_csv(sample_path),
            "metrics": write_csv(metric_path, [], ABLATION_METRIC_COLUMNS),
            "summary": _write_json(
                summary_path,
                {
                    "status": "planned",
                    "primary": manifest["xai"]["primary"],
                    "deep": manifest["xai"]["deep"],
                    "ablation": xai_config,
                },
            ),
        }

    try:
        samples = select_ablation_samples(runtime_core, manifest, sample_size)
    except Exception:
        samples = []
    samples_written = _write_sample_csv(sample_path, samples)

    config = _safe_runtime_config(runtime_core)
    rationale_map = _human_rationales(runtime_xai)

    median_seed = int(manifest["benchmark"]["seeds"][len(manifest["benchmark"]["seeds"]) // 2])

    metric_rows: list[dict[str, Any]] = []
    for condition in ["A_B", "B_B", "C_B", "D_B", "A_R", "B_R", "C_R", "D_R"]:
        if not samples or config is None:
            continue
        checkpoint_path = _checkpoint_for(root, condition, median_seed)
        run_config_path = _run_config_for(root, condition, median_seed)
        bundle = _load_bundle(runtime_xai, condition, checkpoint_path, run_config_path)
        if bundle is None:
            continue

        computed = _compute_unit_metrics(runtime_xai, bundle, samples, config, rationale_map)
        metrics = computed["metrics"]
        backbone = "BERT" if condition.endswith("_B") else "RoBERTa"
        attention_flag = condition.startswith(("B_", "D_"))
        vader_flag = condition.startswith(("C_", "D_"))
        metric_rows.append(
            {
                "condition": condition,
                "backbone": backbone,
                "attention_loss": str(attention_flag),
                "sentiment_feature": str(vader_flag),
                "rationale_f1_at_5": metrics.get("rationale_f1_at_5", ""),
                "comprehensiveness": metrics.get("comprehensiveness", ""),
                "sufficiency": metrics.get("sufficiency", ""),
                "attention_entropy": "",
                "mss": "",
            }
        )

    metric_written = write_csv(metric_path, metric_rows, ABLATION_METRIC_COLUMNS)
    summary_written = _write_json(
        summary_path,
        {
            "status": "completed" if metric_rows else "planned",
            "primary": manifest["xai"]["primary"],
            "deep": manifest["xai"]["deep"],
            "ablation": xai_config,
            "ablation_rows": len(metric_rows),
        },
    )

    return {
        "samples": samples_written,
        "metrics": metric_written,
        "summary": summary_written,
    }
