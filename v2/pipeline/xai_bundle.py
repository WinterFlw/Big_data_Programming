"""Evidence bundle builder for v2 XAI outputs.

작업 #4에서 채운 primary/deep/ablation 산출물을 읽어 report/dashboard가 우선
소비할 bundle 14종을 만든다. 입력이 비어 있으면(=GPU 학습 전) 기존 placeholder
구조를 그대로 유지하고, 입력이 들어오면 row를 자동으로 채운다.

설계 원칙(브리프 §5):

* **저장은 full, 노출은 요약**. CSV는 sample 단위 raw row, JSON은 요약 카드.
* 모든 claim에 source_artifacts 필드를 박는다. 출처 없는 claim 금지.
* 통계 미확증(p-value 없음/유의하지 않음)은 strong claim으로 쓰지 않는다.
* "_no XAI evidence yet_" 같은 placeholder 메시지를 입력 없을 때 보존.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import write_csv
from .paths import CONFIG_DIR, display_path, experiment_root


# ───────────────────────────────────────────────────────────
# 작은 IO 헬퍼
# ───────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows. Missing file → empty list (placeholder safe)."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object. Missing or invalid → empty dict."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _coerce_float(value: Any) -> float | None:
    """Try to parse a CSV cell into float, return None for blanks."""
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_map(sample_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """sample_id → row 매핑. case_summary/predictions row 채울 때 활용."""
    return {row["sample_id"]: row for row in sample_rows if row.get("sample_id")}


def _flatten_attributions_to_jsonl(
    root: Path, primary_samples: list[dict[str, str]]
) -> list[str]:
    """outputs/.../xai/.cache/ 의 SHAP/LIME 결과를 jsonl 한 줄씩 평탄화.

    cache는 작업 #4에서 (condition, seed)당 하나의 JSON으로 저장됨. 각 JSON은
    {"shap": [{text, top_tokens, top_scores, ...}, ...], "lime": [...], ...}
    구조. primary_samples 순서가 cache의 sample 순서와 동일하므로 zip으로
    sample_id를 매핑한다.

    sample 수가 cache 길이와 다르면(예: sample_size 변경 후 stale cache) 그 unit은 skip.
    """
    cache_dir = root / "xai" / ".cache"
    if not cache_dir.exists():
        return []
    lines: list[str] = []
    sample_ids = [str(row.get("sample_id", "")) for row in primary_samples]
    for cache_file in sorted(cache_dir.glob("*_seed_*.json")):
        try:
            with open(cache_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        condition = payload.get("condition", "")
        seed = payload.get("seed", "")
        shap_results = payload.get("shap") or []
        lime_results = payload.get("lime") or []
        # 길이 일치 검사 — sample_size 변경 시 stale cache 방어.
        if not shap_results or len(shap_results) != len(sample_ids):
            continue
        for index, shap_record in enumerate(shap_results):
            lime_record = lime_results[index] if index < len(lime_results) else {}
            record = {
                "sample_id": sample_ids[index] if index < len(sample_ids) else "",
                "condition": condition,
                "seed": seed,
                "tokens": shap_record.get("top_tokens", []),
                "shap_scores": shap_record.get("top_scores", []),
                "lime_tokens": lime_record.get("top_tokens", []),
                "lime_scores": lime_record.get("top_scores", []),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
    return lines


# ───────────────────────────────────────────────────────────
# bundle 각 CSV를 채우는 함수들
# ───────────────────────────────────────────────────────────


def _build_sample_manifest(
    primary_samples: list[dict[str, str]],
    deep_samples: list[dict[str, str]],
    ablation_samples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """xai_sample_manifest.csv 행. 같은 sample이 여러 stage에 선택됐는지 표시."""
    primary_ids = {row["sample_id"] for row in primary_samples}
    deep_ids = {row["sample_id"] for row in deep_samples}
    ablation_ids = {row["sample_id"] for row in ablation_samples}
    union_rows: dict[str, dict[str, Any]] = {}
    for source_rows in (primary_samples, deep_samples, ablation_samples):
        for row in source_rows:
            sid = row["sample_id"]
            if sid in union_rows:
                continue
            rationale_count = _coerce_float(row.get("rationale_count")) or 0.0
            union_rows[sid] = {
                "sample_id": sid,
                "label": row.get("label", ""),
                "split": "test",  # XAI는 test split 기준이 표준.
                "case_type": row.get("case_type", ""),
                "selected_for_primary": str(sid in primary_ids).lower(),
                "selected_for_deep": str(sid in deep_ids).lower(),
                "selected_for_ablation": str(sid in ablation_ids).lower(),
                "rationale_available": str(rationale_count > 0).lower(),
            }
    return list(union_rows.values())


def _build_predictions(
    deep_cases: list[dict[str, str]],
    sample_lookup: dict[str, dict[str, str]],
    median_seed: int | str,
) -> list[dict[str, Any]]:
    """xai_predictions.csv. deep case_summary에서 baseline/v2 prediction을 추출."""
    rows: list[dict[str, Any]] = []
    for case in deep_cases:
        sid = case.get("sample_id", "")
        if not sid:
            continue
        for cond_key, pred_key, conf_key in (
            ("baseline", "baseline_prediction", "baseline_confidence"),
            ("v2", "v2_prediction", "v2_confidence"),
        ):
            rows.append(
                {
                    "sample_id": sid,
                    "condition": cond_key,
                    "seed": median_seed,
                    "true_label": case.get("true_label", ""),
                    "predicted_label": case.get(pred_key, ""),
                    "probability": case.get(conf_key, ""),
                    "checkpoint_path": "",
                }
            )
    return rows


def _build_method_agreement(seed_metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    """method_agreement.csv. seed_level_metrics에서 overlap/rank_corr 추출."""
    rows: list[dict[str, Any]] = []
    for metric_row in seed_metrics:
        rows.append(
            {
                "sample_id": "<aggregated>",  # primary는 sample 200개의 평균이므로 aggregate 마크.
                "condition": metric_row.get("condition", ""),
                "seed": metric_row.get("seed", ""),
                "overlap_at_5": metric_row.get("shap_lime_overlap_at_5", ""),
                "overlap_at_10": metric_row.get("shap_lime_overlap_at_10", ""),
                "rank_corr": metric_row.get("rank_corr_mean", ""),
                "notes": "sample_count=" + str(metric_row.get("sample_count", "")),
            }
        )
    return rows


def _build_faithfulness(seed_metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    """faithfulness_metrics.csv."""
    rows: list[dict[str, Any]] = []
    for metric_row in seed_metrics:
        rows.append(
            {
                "sample_id": "<aggregated>",
                "condition": metric_row.get("condition", ""),
                "seed": metric_row.get("seed", ""),
                "comprehensiveness": metric_row.get("comprehensiveness", ""),
                "sufficiency": metric_row.get("sufficiency", ""),
                "loo_drop": metric_row.get("loo_drop", ""),
            }
        )
    return rows


def _build_plausibility(seed_metrics: list[dict[str, str]]) -> list[dict[str, Any]]:
    """plausibility_metrics.csv."""
    rows: list[dict[str, Any]] = []
    for metric_row in seed_metrics:
        rows.append(
            {
                "sample_id": "<aggregated>",
                "condition": metric_row.get("condition", ""),
                "seed": metric_row.get("seed", ""),
                "rationale_precision_at_5": metric_row.get("rationale_precision_at_5", ""),
                "rationale_recall_at_5": metric_row.get("rationale_recall_at_5", ""),
                "rationale_f1_at_5": metric_row.get("rationale_f1_at_5", ""),
            }
        )
    return rows


def _build_context_metrics(
    deep_cases: list[dict[str, str]],
    sample_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """context_metrics.csv. baseline vs v2 prediction 차이를 context sensitivity 근사로 사용.

    context_window: deep case의 텍스트 토큰 수 (BERT max_len 128 기준 상대 비율).
    context_sensitivity: |baseline_confidence - v2_confidence|. 두 모델의 예측 확률 차이가
    클수록 v2가 추가 맥락 단서를 활용했다는 신호(절대값이라 방향 부호는 별도 분석).

    실제 sample-level attribution 기반 context window 측정은 token_attributions.jsonl
    경유 후속 단계에서 정확히 들어간다. 본 라운드에서는 readily-available한 근사치로 채움.
    """
    rows: list[dict[str, Any]] = []
    for case in deep_cases:
        sid = case.get("sample_id", "")
        sample_row = sample_lookup.get(sid, {})
        text = sample_row.get("text", "")
        token_count = len(text.split()) if text else 0
        # 정규화: 128 토큰 기준 상대 비율 (0~1+ 범위, 1 초과 가능).
        context_window = round(token_count / 128.0, 4) if token_count else ""
        baseline_conf = _coerce_float(case.get("baseline_confidence"))
        v2_conf = _coerce_float(case.get("v2_confidence"))
        if baseline_conf is not None and v2_conf is not None:
            context_sensitivity = round(abs(baseline_conf - v2_conf), 4)
        else:
            context_sensitivity = ""
        rows.append(
            {
                "sample_id": sid,
                "condition": "deep_primary_pair",
                "target": sample_row.get("target", ""),
                "source": sample_row.get("source", ""),
                "context_window": context_window,
                "context_sensitivity": context_sensitivity,
            }
        )
    return rows


def _build_subgroup_metrics(
    seed_metrics: list[dict[str, str]],
    sample_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """subgroup_xai_metrics.csv. source(gab/twitter)와 target(인종/종교/...) 두 차원 분해.

    primary sample의 source/target 분포를 보고 각 subgroup의 표본 수를 헤더에 박는다.
    seed_metrics는 sample-level이 아닌 (condition, seed) 평균이라, 실제 subgroup 평균은
    후속 sample-level 메트릭이 들어와야 정확하다. 본 라운드에서는 표본 수 + condition 평균을
    표 형태로 노출만 한다 (subgroup-aware 분해는 token_attributions.jsonl 기반 후속 단계).
    """
    if not seed_metrics or not sample_rows:
        return []

    # HateXplain post_id 접미사가 비정상인 sample은 source 추출이 망가져서
    # 숫자 토큰("4")이 들어올 수 있다. 알려진 집합 외는 "other"로 묶는다.
    KNOWN_SOURCES = {"gab", "twitter"}
    # source 분포.
    source_counts: dict[str, int] = {}
    # target 분포 (multi-label, comma-separated).
    target_counts: dict[str, int] = {}
    for sample_row in sample_rows:
        raw_source = sample_row.get("source", "").strip().lower()
        source = raw_source if raw_source in KNOWN_SOURCES else "other"
        source_counts[source] = source_counts.get(source, 0) + 1
        target_field = sample_row.get("target", "")
        if target_field:
            for target in target_field.split(","):
                target = target.strip()
                if target and target.lower() not in ("none", "other"):
                    target_counts[target] = target_counts.get(target, 0) + 1

    rows: list[dict[str, Any]] = []
    metric_names = ("rationale_f1_at_5", "comprehensiveness", "sufficiency")

    # source-별 분해.
    for source, count in sorted(source_counts.items()):
        for metric_row in seed_metrics:
            for metric_name in metric_names:
                value = metric_row.get(metric_name, "")
                rows.append(
                    {
                        "subgroup": f"source={source}(n={count})",
                        "condition": metric_row.get("condition", ""),
                        "seed": metric_row.get("seed", ""),
                        "metric": metric_name,
                        "value": value,
                    }
                )

    # target-별 분해 (상위 10개만; 너무 많으면 dashboard noisy).
    top_targets = sorted(target_counts.items(), key=lambda item: -item[1])[:10]
    for target, count in top_targets:
        for metric_row in seed_metrics:
            for metric_name in metric_names:
                value = metric_row.get(metric_name, "")
                rows.append(
                    {
                        "subgroup": f"target={target}(n={count})",
                        "condition": metric_row.get("condition", ""),
                        "seed": metric_row.get("seed", ""),
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return rows


def _build_risk_flags(
    seed_metrics: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """xai_risk_flags.csv. 데이터 부재·표본 부족·이상치 등 경고를 한 줄씩 적는다.

    현재 규칙:
      * seed_level_metrics가 비어 있으면 'no_seed_metrics' 경고 1행.
      * paired 검정에서 n_pairs < 5인 metric이 있으면 'low_pair_count' 경고.
    """
    rows: list[dict[str, Any]] = []
    if not seed_metrics:
        rows.append(
            {
                "sample_id": "<run-level>",
                "condition": "all",
                "seed": "",
                "flag_type": "no_seed_metrics",
                "severity": "info",
                "evidence": "xai/primary/seed_level_metrics.csv has no rows.",
                "recommended_report_note": "Mark XAI section as not yet executed.",
            }
        )
    for paired_row in paired_rows:
        try:
            n_pairs = int(paired_row.get("n_pairs") or 0)
        except ValueError:
            n_pairs = 0
        if 0 < n_pairs < 5:
            rows.append(
                {
                    "sample_id": "<run-level>",
                    "condition": paired_row.get("comparison", ""),
                    "seed": "",
                    "flag_type": "low_pair_count",
                    "severity": "warning",
                    "evidence": f"n_pairs={n_pairs} for metric {paired_row.get('metric', '')}",
                    "recommended_report_note": "Treat XAI paired test as exploratory.",
                }
            )
    return rows


# ───────────────────────────────────────────────────────────
# claims / interpretation cards / dashboard bundle
# ───────────────────────────────────────────────────────────


def _strength_from_p(p_value: float | None) -> str:
    """p-value → claim 강도."""
    if p_value is None:
        return "exploratory"
    if p_value < 0.01:
        return "strong"
    if p_value < 0.05:
        return "moderate"
    return "weak"


def _build_claims(
    benchmark_paired: list[dict[str, str]],
    xai_paired: list[dict[str, str]],
    source_artifacts: dict[str, str],
) -> list[dict[str, Any]]:
    """통계적으로 확증된 주장만 모아 claim list 생성.

    * benchmark/paired_tests_holm.csv의 p_value_holm < 0.05 인 macro_f1 비교
    * xai/primary/paired_xai_tests.csv의 p_value < 0.05 인 metric 비교
    """
    claims: list[dict[str, Any]] = []
    claim_index = 1
    for row in benchmark_paired:
        if row.get("metric") != "macro_f1":
            continue
        p_value_holm = _coerce_float(row.get("p_value_holm"))
        mean_diff = _coerce_float(row.get("mean_diff"))
        if p_value_holm is None or mean_diff is None:
            continue
        if p_value_holm >= 0.05:
            continue
        comparison = row.get("comparison", "")
        direction = "higher" if mean_diff > 0 else "lower"
        claims.append(
            {
                "id": f"claim_{claim_index:03d}",
                "category": "benchmark",
                "text": (
                    f"{comparison} shows {direction} macro F1 (mean diff "
                    f"{mean_diff:+.4f}, Holm-adjusted p={p_value_holm:.4f})."
                ),
                "strength": _strength_from_p(p_value_holm),
                "source_artifacts": [source_artifacts["benchmark_paired_holm"]],
            }
        )
        claim_index += 1

    for row in xai_paired:
        p_value = _coerce_float(row.get("p_value"))
        mean_diff = _coerce_float(row.get("mean_diff"))
        if p_value is None or mean_diff is None:
            continue
        if p_value >= 0.05:
            continue
        comparison = row.get("comparison", "")
        metric_name = row.get("metric", "")
        direction = "higher" if mean_diff > 0 else "lower"
        claims.append(
            {
                "id": f"claim_{claim_index:03d}",
                "category": "xai",
                "text": (
                    f"{comparison} differs on {metric_name} "
                    f"(mean diff {mean_diff:+.4f}, p={p_value:.4f}, {direction})."
                ),
                "strength": _strength_from_p(p_value),
                "source_artifacts": [source_artifacts["primary_paired_tests"]],
            }
        )
        claim_index += 1
    return claims


def _build_interpretation_cards(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """xai_interpretation_cards.json. claim 한 줄을 풀어쓴 사람 친화 해석.

    "이게 무엇이고 어떻게 해석하나"를 짧게 적어 발표·보고서에서 직접 인용 가능.
    """
    cards: list[dict[str, Any]] = []
    for claim in claims:
        cards.append(
            {
                "claim_id": claim["id"],
                "headline": claim["text"],
                "what_it_means": (
                    "Strong/moderate strength claims are recommended for the main report. "
                    "Weak/exploratory results should remain in the appendix."
                ),
                "how_to_read": (
                    "Interpret the sign of mean_diff relative to the comparison label. "
                    "A positive diff means the second condition exceeds the first."
                ),
                "source_artifacts": claim["source_artifacts"],
                "strength": claim["strength"],
            }
        )
    return cards


def _summary_cards(
    benchmark_summary: list[dict[str, str]],
    seed_metrics: list[dict[str, str]],
    seed_stability: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """dashboard bundle의 summary_cards 영역."""
    cards: list[dict[str, Any]] = []
    completed = [
        row for row in benchmark_summary
        if row.get("n_seeds") and row["n_seeds"] not in ("", "0")
    ]
    if completed:
        try:
            top = max(
                completed,
                key=lambda row: _coerce_float(row.get("macro_f1_mean")) or float("-inf"),
            )
            cards.append(
                {
                    "title": "Top benchmark condition",
                    "value": f"{top.get('condition', '?')} "
                             f"({top.get('macro_f1_mean', '?')} ± {top.get('macro_f1_std', '?')})",
                    "source": "benchmark/benchmark_summary.csv",
                }
            )
        except ValueError:
            pass

    if seed_metrics:
        valid = [
            row for row in seed_metrics
            if _coerce_float(row.get("rationale_f1_at_5")) is not None
        ]
        if valid:
            average_f1 = sum(_coerce_float(row["rationale_f1_at_5"]) or 0.0 for row in valid) / len(valid)
            cards.append(
                {
                    "title": "Rationale F1@5 (mean across seeds)",
                    "value": f"{average_f1:.4f}",
                    "source": "xai/primary/seed_level_metrics.csv",
                }
            )

    if seed_stability:
        jaccard_rows = [row for row in seed_stability if row.get("metric") == "topk_jaccard_5"]
        if jaccard_rows:
            try:
                best = max(
                    jaccard_rows,
                    key=lambda row: _coerce_float(row.get("mean")) or float("-inf"),
                )
                cards.append(
                    {
                        "title": "Most stable explanation (top-k Jaccard)",
                        "value": f"{best.get('condition', '?')} mean={best.get('mean', '?')}",
                        "source": "xai/primary/seed_stability.csv",
                    }
                )
            except ValueError:
                pass

    if not cards:
        cards.append(
            {
                "title": "No XAI evidence yet",
                "value": "Run xai-primary / xai-deep / xai-ablation, then re-run xai-bundle.",
                "source": "<bundle placeholder>",
            }
        )
    return cards


# ───────────────────────────────────────────────────────────
# 메인 entry: build_xai_evidence_bundle
# ───────────────────────────────────────────────────────────


def build_xai_evidence_bundle(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Path | str]:
    """Create the canonical XAI evidence bundle surface.

    작업 #4의 산출물을 읽어 14개 bundle 파일을 채운다. 입력이 없으면 14개 모두
    placeholder(헤더만 / 빈 객체)로 유지.
    """
    root = experiment_root(manifest["run_id"])
    bundle_dir = root / "xai" / "evidence_bundle"
    if dry_run:
        return {"status": "dry-run", "bundle_dir": bundle_dir}

    source_artifacts = {
        "primary_samples": "xai/samples/primary_samples.csv",
        "deep_samples": "xai/samples/deep_samples.csv",
        "ablation_samples": "xai/samples/ablation_samples.csv",
        "primary_seed_metrics": "xai/primary/seed_level_metrics.csv",
        "primary_paired_tests": "xai/primary/paired_xai_tests.csv",
        "primary_seed_stability": "xai/primary/seed_stability.csv",
        "deep_case_summary": "xai/deep/case_summary.csv",
        "deep_details": "xai/deep/xai_details.json",
        "ablation_metrics": "xai/ablation/xai_ablation_metrics.csv",
        "xai_summary": "xai/xai_summary.json",
        "benchmark_summary": "benchmark/benchmark_summary.csv",
        "benchmark_paired_holm": "benchmark/paired_tests_holm.csv",
    }

    # 입력 읽기.
    primary_samples = _read_csv_rows(root / source_artifacts["primary_samples"])
    deep_samples = _read_csv_rows(root / source_artifacts["deep_samples"])
    ablation_samples = _read_csv_rows(root / source_artifacts["ablation_samples"])
    seed_metrics = _read_csv_rows(root / source_artifacts["primary_seed_metrics"])
    primary_paired = _read_csv_rows(root / source_artifacts["primary_paired_tests"])
    seed_stability = _read_csv_rows(root / source_artifacts["primary_seed_stability"])
    deep_cases = _read_csv_rows(root / source_artifacts["deep_case_summary"])
    benchmark_summary = _read_csv_rows(root / source_artifacts["benchmark_summary"])
    benchmark_paired = _read_csv_rows(root / source_artifacts["benchmark_paired_holm"])
    deep_details = _read_json(root / source_artifacts["deep_details"])

    sample_lookup = _sample_map(primary_samples)
    median_seed = deep_details.get("median_seed", "")

    # 인벤토리: 어떤 입력이 있고 어떤 게 비었는지 한 눈에.
    inventory_rows = [
        {
            "artifact": name,
            "path": relative_path,
            "exists": str((root / relative_path).exists()).lower(),
        }
        for name, relative_path in source_artifacts.items()
    ]
    inventory_path = write_csv(
        bundle_dir / "evidence_inventory.csv",
        inventory_rows,
        ["artifact", "path", "exists"],
    )

    # 메타데이터.
    metadata_path = _write_json(
        bundle_dir / "xai_run_metadata.json",
        {
            "status": "completed" if seed_metrics else "planned",
            "run_id": manifest["run_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commit_hash": "pending",
            "config_path": display_path(CONFIG_DIR / f"{manifest['run_id']}.json"),
            "manifest_path": display_path(root / "manifest.json"),
            "data_split_hash": "pending",
            "conditions": manifest["benchmark"]["conditions"],
            "seeds": manifest["benchmark"]["seeds"],
            "source_artifacts": source_artifacts,
        },
    )

    # 각 CSV 채움.
    sample_manifest_rows = _build_sample_manifest(primary_samples, deep_samples, ablation_samples)
    sample_manifest_path = write_csv(
        bundle_dir / "xai_sample_manifest.csv",
        sample_manifest_rows,
        [
            "sample_id",
            "label",
            "split",
            "case_type",
            "selected_for_primary",
            "selected_for_deep",
            "selected_for_ablation",
            "rationale_available",
        ],
    )

    predictions_rows = _build_predictions(deep_cases, sample_lookup, median_seed)
    predictions_path = write_csv(
        bundle_dir / "xai_predictions.csv",
        predictions_rows,
        ["sample_id", "condition", "seed", "true_label", "predicted_label", "probability", "checkpoint_path"],
    )

    method_agreement_rows = _build_method_agreement(seed_metrics)
    method_agreement_path = write_csv(
        bundle_dir / "method_agreement.csv",
        method_agreement_rows,
        ["sample_id", "condition", "seed", "overlap_at_5", "overlap_at_10", "rank_corr", "notes"],
    )

    faithfulness_rows = _build_faithfulness(seed_metrics)
    faithfulness_path = write_csv(
        bundle_dir / "faithfulness_metrics.csv",
        faithfulness_rows,
        ["sample_id", "condition", "seed", "comprehensiveness", "sufficiency", "loo_drop"],
    )

    plausibility_rows = _build_plausibility(seed_metrics)
    plausibility_path = write_csv(
        bundle_dir / "plausibility_metrics.csv",
        plausibility_rows,
        ["sample_id", "condition", "seed", "rationale_precision_at_5", "rationale_recall_at_5", "rationale_f1_at_5"],
    )

    context_rows = _build_context_metrics(deep_cases, sample_lookup)
    context_path = write_csv(
        bundle_dir / "context_metrics.csv",
        context_rows,
        ["sample_id", "condition", "target", "source", "context_window", "context_sensitivity"],
    )

    subgroup_rows = _build_subgroup_metrics(seed_metrics, primary_samples)
    subgroup_path = write_csv(
        bundle_dir / "subgroup_xai_metrics.csv",
        subgroup_rows,
        ["subgroup", "condition", "seed", "metric", "value"],
    )

    risk_rows = _build_risk_flags(seed_metrics, primary_paired)
    risk_flags_path = write_csv(
        bundle_dir / "xai_risk_flags.csv",
        risk_rows,
        ["sample_id", "condition", "seed", "flag_type", "severity", "evidence", "recommended_report_note"],
    )

    # claims / cards / dashboard bundle.
    claims = _build_claims(benchmark_paired, primary_paired, source_artifacts)
    claims_path = _write_json(
        bundle_dir / "xai_claims.json",
        {
            "status": "completed" if claims else "planned",
            "run_id": manifest["run_id"],
            "purpose": "Report-ready XAI claims derived from primary/deep/ablation artifacts.",
            "claims": claims,
            "source_artifacts": source_artifacts,
            "required_before_claiming": [
                "Fill primary seed-level XAI metrics.",
                "Fill paired XAI tests and seed stability.",
                "Fill deep qualitative case summaries.",
                "Fill ablation-level XAI metrics.",
            ] if not claims else [],
        },
    )

    cards = _build_interpretation_cards(claims)
    interpretation_cards_path = _write_json(
        bundle_dir / "xai_interpretation_cards.json",
        {
            "status": "completed" if cards else "planned",
            "run_id": manifest["run_id"],
            "cards": cards,
        },
    )

    dashboard_bundle_path = _write_json(
        bundle_dir / "xai_dashboard_bundle.json",
        {
            "status": "completed" if seed_metrics else "planned",
            "run_id": manifest["run_id"],
            "summary_cards": _summary_cards(benchmark_summary, seed_metrics, seed_stability),
            "primary": {
                "seed_metric_count": len(seed_metrics),
                "paired_test_count": len(primary_paired),
            },
            "seed_stability": {
                "rows": len(seed_stability),
            },
            "deep_cases": [
                {"sample_id": row.get("sample_id"), "true_label": row.get("true_label")}
                for row in deep_cases[:20]  # noisy 방지: dashboard에 너무 많이 박지 않는다.
            ],
            "ablation": {
                "row_count": deep_details.get("ablation_rows", 0),
            },
            "artifact_links": source_artifacts,
        },
    )

    # token_attributions.jsonl: 작업 #4가 outputs/.../xai/.cache/<cond>_seed_<seed>.json
    # 에 저장한 SHAP/LIME 결과를 한 줄당 (sample_id, condition, seed, tokens, shap_scores,
    # lime_scores) 형태로 평탄화한다. cache가 비어 있으면 빈 파일 유지.
    token_attributions_path = bundle_dir / "token_attributions.jsonl"
    token_lines = _flatten_attributions_to_jsonl(root, primary_samples)
    # cache가 비면 jsonl도 비운다 — stale row 방지.
    token_attributions_path.write_text(
        ("\n".join(token_lines) + "\n") if token_lines else "",
        encoding="utf-8",
    )

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        f"""# XAI Evidence Bundle

run_id: `{manifest["run_id"]}`

This directory is the canonical XAI bundle for report/dashboard stages.
Files are populated automatically from primary/deep/ablation artifacts.
If a file appears empty, the corresponding upstream stage has not produced
its metric rows yet — re-run the relevant `./run.sh e2e xai-*` command.

Report and dashboard code should prefer:

- `xai_claims.json`
- `xai_dashboard_bundle.json`
- `xai_interpretation_cards.json`

Raw per-sample evidence remains available through the CSV/JSONL files in this
directory.
""",
        encoding="utf-8",
    )

    return {
        "inventory": inventory_path,
        "metadata": metadata_path,
        "sample_manifest": sample_manifest_path,
        "predictions": predictions_path,
        "method_agreement": method_agreement_path,
        "faithfulness": faithfulness_path,
        "context": context_path,
        "plausibility": plausibility_path,
        "subgroup": subgroup_path,
        "risk_flags": risk_flags_path,
        "claims": claims_path,
        "interpretation_cards": interpretation_cards_path,
        "dashboard_bundle": dashboard_bundle_path,
        "token_attributions": token_attributions_path,
        "readme": readme_path,
    }
