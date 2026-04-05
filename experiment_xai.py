"""
XAI (설명 가능한 AI) 분석 파이프라인.

연구의 핵심 흐름:
  Phase 1: BERT baseline에 SHAP + LIME 적용 → 오분류 원인 진단
  Phase 2: 개선 모델(+VADER)에 동일 분석 → Before/After 비교

분석 산출물:
  - Overlap@5: SHAP Top-5 ∩ LIME Top-5 일치도 (≥60%이면 높은 신뢰)
  - 케이스 비교: 오분류→정분류 전환 샘플의 SHAP attribution 변화
  - xai_summary.md: 보고서에 바로 사용 가능한 요약

주의: SHAP는 CPU에서 실행 (MPS에서 DeepExplainer 비호환).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = MODULE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
CACHE_DIR = MODULE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from experiment_core import (
    BENCHMARK_SUMMARY_PATH,
    BEST_MODELS_PATH,
    ExperimentConfig,
    HybridSentimentClassifier,
    SPLITS_PICKLE_PATH,
    TransformerCLSClassifier,
    get_config,
)
from utils import (
    LABEL_NAMES,
    NUM_LABELS,
    VADER_COLUMNS,
    XAI_DIR,
    clear_device_cache,
    compute_metrics,
    dataframe_to_markdown,
    ensure_dir,
    load_json,
    load_pickle,
    plot_confusion_matrix,
    save_dataframe,
    save_json,
    save_text,
    set_seed,
    slugify,
)


@dataclass
class LoadedModelBundle:
    """Loaded model bundle used by the XAI helpers."""

    display_name: str
    model_type: str
    model_name: str
    model: torch.nn.Module
    tokenizer: Any
    device: torch.device


IMPROVED_MODEL_NAMES = ["BERT+VADER", "RoBERTa+VADER"]


def _load_best_registry() -> dict[str, dict[str, Any]]:
    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError("Benchmark results are missing. Run the benchmark first.")
    return load_json(BEST_MODELS_PATH)


def _select_best_improved_model_name(registry: dict[str, dict[str, Any]]) -> str:
    """Select the strongest improved model from saved benchmark artifacts."""
    if BENCHMARK_SUMMARY_PATH.exists():
        summary_frame = pd.read_csv(BENCHMARK_SUMMARY_PATH)
        candidate_frame = summary_frame[summary_frame["model"].isin(IMPROVED_MODEL_NAMES)].copy()
        if not candidate_frame.empty:
            candidate_frame = candidate_frame.sort_values(
                by=["macro_f1_mean", "macro_precision_mean", "macro_recall_mean", "accuracy_mean"],
                ascending=[False, False, False, False],
            )
            return str(candidate_frame.iloc[0]["model"])

    available_candidates = [
        name
        for name in IMPROVED_MODEL_NAMES
        if name in registry
    ]
    if not available_candidates:
        raise KeyError("No improved-model registry entries were found for XAI.")

    return max(
        available_candidates,
        key=lambda name: registry[name].get("best_val_macro_f1", float("-inf")),
    )


def _instantiate_bundle(display_name: str, record: dict[str, Any]) -> LoadedModelBundle:
    checkpoint_path = Path(record["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyperparams = record.get("hyperparams", {})
    dropout = float(hyperparams.get("dropout") or 0.1)
    hidden_dim = int(hyperparams.get("mlp_hidden") or 256)

    if display_name == "BERT-base":
        model = TransformerCLSClassifier(
            model_name="bert-base-uncased",
            dropout=dropout,
        )
        model_type = "transformer"
        model_name = "bert-base-uncased"
    elif display_name == "BERT+VADER":
        model = HybridSentimentClassifier(
            model_name="bert-base-uncased",
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model_type = "hybrid"
        model_name = "bert-base-uncased"
    elif display_name == "RoBERTa+VADER":
        model = HybridSentimentClassifier(
            model_name="roberta-base",
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model_type = "hybrid"
        model_name = "roberta-base"
    else:
        raise ValueError(f"Unsupported XAI model: {display_name}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return LoadedModelBundle(
        display_name=display_name,
        model_type=model_type,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )


def load_bundles_for_xai() -> tuple[LoadedModelBundle, LoadedModelBundle]:
    """Load the baseline and improved model used in the XAI report."""
    registry = _load_best_registry()
    baseline = _instantiate_bundle("BERT-base", registry["BERT-base"])
    improved_model_name = _select_best_improved_model_name(registry)
    improved = _instantiate_bundle(improved_model_name, registry[improved_model_name])
    return baseline, improved


def _compute_vader_array(texts: list[str]) -> np.ndarray:
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        rows.append([scores[column] for column in VADER_COLUMNS])
    return np.asarray(rows, dtype=np.float32)


def predict_probabilities(bundle: LoadedModelBundle, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """텍스트 리스트에 대해 3-class 확률 예측. LIME의 predict_fn으로도 사용됨."""
    if not texts:
        return np.zeros((0, NUM_LABELS), dtype=np.float32)

    device = bundle.device
    bundle.model.to(device)
    all_probabilities = []

    vader_array = _compute_vader_array(texts) if bundle.model_type == "hybrid" else None
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = bundle.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=get_config().max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            if bundle.model_type == "hybrid":
                vader = torch.tensor(vader_array[start : start + batch_size], dtype=torch.float32).to(device)
                logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask, vader=vader)
            else:
                logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probabilities.append(probabilities)

    bundle.model.to("cpu")
    clear_device_cache()
    return np.vstack(all_probabilities)


def _normalize_token(token: str) -> str:
    return token.lower().replace("##", "").replace("Ġ", "").strip()


def _extract_shap_scores(values: np.ndarray, tokens: list[str], predicted_label: int) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        return np.zeros(len(tokens), dtype=float)
    if values.shape[0] == len(tokens):
        return values[:, predicted_label]
    if values.shape[1] == len(tokens):
        return values[predicted_label, :]
    return np.zeros(len(tokens), dtype=float)


def run_shap_explanations(
    bundle: LoadedModelBundle,
    texts: list[str],
    predicted_labels: list[int],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """
    SHAP 설명 생성 (Lundberg & Lee, 2017).
    각 토큰의 예측 기여도(Shapley value)를 계산하여 Top-5 중요 토큰 추출.
    CPU에서 실행됨 — MPS 미지원.
    """
    explainer = shap.Explainer(
        lambda batch: predict_probabilities(bundle, list(batch)),
        bundle.tokenizer,
        output_names=LABEL_NAMES,
    )
    shap_values = explainer(
        texts,
        max_evals=config.shap_max_evals,
        batch_size=config.shap_batch_size,
    )

    results = []
    special_tokens = set(bundle.tokenizer.all_special_tokens)
    for index, text in enumerate(texts):
        raw_tokens = shap_values.data[index]
        tokens = [str(token) for token in raw_tokens] if not isinstance(raw_tokens, str) else bundle.tokenizer.tokenize(raw_tokens)
        token_scores = _extract_shap_scores(np.asarray(shap_values.values[index]), tokens, predicted_labels[index])

        token_pairs = []
        for token, score in zip(tokens, token_scores):
            normalized = _normalize_token(token)
            if not normalized or token in special_tokens:
                continue
            token_pairs.append({"token": token, "score": float(score), "abs_score": float(abs(score))})
        token_pairs = sorted(token_pairs, key=lambda item: item["abs_score"], reverse=True)

        results.append(
            {
                "text": text,
                "top_tokens": [item["token"] for item in token_pairs[:5]],
                "top_scores": [item["score"] for item in token_pairs[:5]],
                "token_details": token_pairs,
            }
        )
    return results


def run_lime_explanations(
    bundle: LoadedModelBundle,
    texts: list[str],
    predicted_labels: list[int],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """
    LIME 설명 생성 (Ribeiro et al., 2016).
    입력 텍스트를 perturbation하여 로컬 선형 모델로 근사, Top-5 피처 추출.
    Model-agnostic이므로 SHAP과 독립적인 교차 검증 수단으로 활용.
    """
    explainer = LimeTextExplainer(class_names=LABEL_NAMES, split_expression=r"\s+")
    results = []

    def predict_fn(batch_texts: list[str]) -> np.ndarray:
        return predict_probabilities(bundle, list(batch_texts))

    for text, predicted_label in zip(texts, predicted_labels):
        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
        )
        feature_weights = explanation.as_list(label=int(predicted_label))
        results.append(
            {
                "text": text,
                "top_tokens": [token for token, _ in feature_weights[:5]],
                "top_scores": [float(weight) for _, weight in feature_weights[:5]],
                "feature_weights": [{"token": token, "score": float(weight)} for token, weight in feature_weights],
            }
        )
    return results


def _compute_overlap_at_5(
    shap_results: list[dict[str, Any]],
    lime_results: list[dict[str, Any]],
) -> list[float]:
    overlaps = []
    for shap_result, lime_result in zip(shap_results, lime_results):
        shap_top = {_normalize_token(token) for token in shap_result["top_tokens"] if _normalize_token(token)}
        lime_top = {_normalize_token(token) for token in lime_result["top_tokens"] if _normalize_token(token)}
        matched_shap_tokens = set()
        for shap_token in shap_top:
            if shap_token in lime_top:
                matched_shap_tokens.add(shap_token)
                continue
            for lime_token in lime_top:
                if shap_token in lime_token or lime_token in shap_token:
                    matched_shap_tokens.add(shap_token)
                    break
        overlaps.append(len(matched_shap_tokens & lime_top) / 5.0)
    return overlaps


def _plot_overlap_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plot_frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_frame.boxplot(column="overlap_at_5", by="model", ax=ax)
    ax.axhline(0.6, color="red", linestyle="--", label="Trust threshold (0.6)")
    ax.set_title("Overlap@5 by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Overlap@5")
    plt.suptitle("")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_case_comparison(
    case_index: int,
    text: str,
    baseline_result: dict[str, Any],
    improved_result: dict[str, Any],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, title, result in [
        (axes[0], "BERT-base SHAP", baseline_result),
        (axes[1], "RoBERTa+VADER SHAP", improved_result),
    ]:
        tokens = result["top_tokens"][:5]
        scores = result["top_scores"][:5]
        axis.barh(range(len(tokens)), scores, color="#4c72b0")
        axis.set_yticks(range(len(tokens)))
        axis.set_yticklabels(tokens)
        axis.invert_yaxis()
        axis.set_title(title)
    fig.suptitle(f"Case {case_index + 1}: {text[:90]}...")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _select_analysis_samples(
    texts: list[str],
    labels: np.ndarray,
    baseline_preds: np.ndarray,
    improved_preds: np.ndarray,
    sample_size: int,
) -> pd.DataFrame:
    fixed_rows = []
    stable_rows = []
    disagreement_rows = []
    fallback_rows = []
    columns = [
        "index",
        "text",
        "true_label",
        "true_label_name",
        "baseline_pred",
        "baseline_pred_name",
        "improved_pred",
        "improved_pred_name",
        "category",
    ]

    for index, (text, label, baseline_pred, improved_pred) in enumerate(
        zip(texts, labels, baseline_preds, improved_preds)
    ):
        row = {
            "index": index,
            "text": text,
            "true_label": int(label),
            "true_label_name": LABEL_NAMES[int(label)],
            "baseline_pred": int(baseline_pred),
            "baseline_pred_name": LABEL_NAMES[int(baseline_pred)],
            "improved_pred": int(improved_pred),
            "improved_pred_name": LABEL_NAMES[int(improved_pred)],
        }
        if baseline_pred != label and improved_pred == label:
            row["category"] = "fixed_error"
            fixed_rows.append(row)
        elif baseline_pred == label and improved_pred == label:
            row["category"] = "consistently_correct"
            stable_rows.append(row)
        elif baseline_pred != improved_pred:
            row["category"] = "model_disagreement"
            disagreement_rows.append(row)
        else:
            row["category"] = "fallback_sample"
            fallback_rows.append(row)

    selected = fixed_rows[: max(sample_size // 2, 1)]
    remaining = max(sample_size - len(selected), 0)
    selected.extend(stable_rows[:remaining])
    remaining = max(sample_size - len(selected), 0)
    if remaining > 0:
        selected.extend(disagreement_rows[:remaining])
    remaining = max(sample_size - len(selected), 0)
    if remaining > 0:
        selected.extend(fallback_rows[:remaining])

    return pd.DataFrame(selected, columns=columns)


def run_xai(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """
    XAI 전체 파이프라인 실행: Baseline vs 개선 모델의 SHAP/LIME 비교.

    흐름:
      1. 테스트셋 예측 (baseline + improved)
      2. 분석 대상 샘플 선정 (오분류→정분류 전환 우선)
      3. SHAP + LIME 설명 생성 (각 모델별)
      4. Overlap@5 계산 및 시각화
      5. 케이스별 SHAP attribution 비교 차트 생성
      6. xai_summary.json + xai_summary.md 저장
    """
    config = config or get_config()
    set_seed(config.tuning_seed)
    ensure_dir(XAI_DIR)

    if not SPLITS_PICKLE_PATH.exists():
        raise FileNotFoundError("Data split artifact is missing. Run data preparation first.")

    splits = load_pickle(SPLITS_PICKLE_PATH)
    test_df = splits["test"]
    texts = test_df["text"].tolist()
    labels = test_df["label"].to_numpy()

    baseline_bundle, improved_bundle = load_bundles_for_xai()
    baseline_prob = predict_probabilities(baseline_bundle, texts)
    improved_prob = predict_probabilities(improved_bundle, texts)
    baseline_preds = baseline_prob.argmax(axis=1)
    improved_preds = improved_prob.argmax(axis=1)

    baseline_metrics = compute_metrics(labels, baseline_preds, baseline_prob)
    improved_metrics = compute_metrics(labels, improved_preds, improved_prob)

    baseline_dir = ensure_dir(XAI_DIR / slugify(baseline_bundle.display_name))
    improved_dir = ensure_dir(XAI_DIR / slugify(improved_bundle.display_name))
    plot_confusion_matrix(
        np.asarray(baseline_metrics["confusion_matrix"]),
        baseline_bundle.display_name,
        baseline_dir / "confusion_matrix.png",
    )
    plot_confusion_matrix(
        np.asarray(improved_metrics["confusion_matrix"]),
        improved_bundle.display_name,
        improved_dir / "confusion_matrix.png",
    )

    sample_frame = _select_analysis_samples(
        texts=texts,
        labels=labels,
        baseline_preds=baseline_preds,
        improved_preds=improved_preds,
        sample_size=config.xai_sample_size,
    )
    save_dataframe(sample_frame, XAI_DIR / "analysis_samples.csv")

    if sample_frame.empty:
        empty_overlap = pd.DataFrame(columns=["model", "sample_id", "overlap_at_5"])
        empty_cases = pd.DataFrame(
            columns=[
                "sample_id",
                "category",
                "baseline_top_tokens",
                "improved_top_tokens",
                "baseline_overlap_at_5",
                "improved_overlap_at_5",
            ]
        )
        save_dataframe(empty_overlap, XAI_DIR / "overlap_at_5.csv")
        save_dataframe(empty_cases, XAI_DIR / "case_summary.csv")

        summary = {
            "baseline_model": baseline_bundle.display_name,
            "improved_model": improved_bundle.display_name,
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "improved_macro_f1": improved_metrics["macro_f1"],
            "baseline_overlap_mean": None,
            "improved_overlap_mean": None,
            "baseline_overlap_ge_60": 0,
            "improved_overlap_ge_60": 0,
            "sample_count": 0,
            "fixed_error_count": 0,
            "message": "No eligible XAI samples were selected.",
        }
        save_json(summary, XAI_DIR / "xai_summary.json")
        save_text(
            "# XAI Summary\n\n"
            + dataframe_to_markdown(pd.DataFrame([summary])),
            XAI_DIR / "xai_summary.md",
        )
        save_json(
            {
                "baseline_shap": [],
                "baseline_lime": [],
                "improved_shap": [],
                "improved_lime": [],
            },
            XAI_DIR / "xai_details.json",
        )
        return summary

    sample_texts = sample_frame["text"].tolist()
    baseline_sample_preds = sample_frame["baseline_pred"].tolist()
    improved_sample_preds = sample_frame["improved_pred"].tolist()

    baseline_shap = run_shap_explanations(baseline_bundle, sample_texts, baseline_sample_preds, config)
    baseline_lime = run_lime_explanations(baseline_bundle, sample_texts, baseline_sample_preds, config)
    improved_shap = run_shap_explanations(improved_bundle, sample_texts, improved_sample_preds, config)
    improved_lime = run_lime_explanations(improved_bundle, sample_texts, improved_sample_preds, config)

    baseline_overlap = _compute_overlap_at_5(baseline_shap, baseline_lime)
    improved_overlap = _compute_overlap_at_5(improved_shap, improved_lime)

    overlap_rows = []
    for index, overlap in enumerate(baseline_overlap):
        overlap_rows.append({"model": baseline_bundle.display_name, "sample_id": int(sample_frame.iloc[index]["index"]), "overlap_at_5": overlap})
    for index, overlap in enumerate(improved_overlap):
        overlap_rows.append({"model": improved_bundle.display_name, "sample_id": int(sample_frame.iloc[index]["index"]), "overlap_at_5": overlap})
    overlap_frame = pd.DataFrame(overlap_rows)
    save_dataframe(overlap_frame, XAI_DIR / "overlap_at_5.csv")
    _plot_overlap_summary(overlap_rows, XAI_DIR / "overlap_at_5.png")

    case_rows = []
    for local_index, row in sample_frame.head(8).reset_index(drop=True).iterrows():
        baseline_case = baseline_shap[local_index]
        improved_case = improved_shap[local_index]
        _plot_case_comparison(
            case_index=local_index,
            text=row["text"],
            baseline_result=baseline_case,
            improved_result=improved_case,
            output_path=XAI_DIR / "cases" / f"case_{local_index + 1:02d}.png",
        )
        case_rows.append(
            {
                "sample_id": int(row["index"]),
                "category": row["category"],
                "baseline_top_tokens": ", ".join(baseline_case["top_tokens"][:5]),
                "improved_top_tokens": ", ".join(improved_case["top_tokens"][:5]),
                "baseline_overlap_at_5": baseline_overlap[local_index],
                "improved_overlap_at_5": improved_overlap[local_index],
            }
        )

    case_frame = pd.DataFrame(case_rows)
    save_dataframe(case_frame, XAI_DIR / "case_summary.csv")

    summary = {
        "baseline_model": baseline_bundle.display_name,
        "improved_model": improved_bundle.display_name,
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "improved_macro_f1": improved_metrics["macro_f1"],
        "baseline_overlap_mean": float(np.mean(baseline_overlap)) if baseline_overlap else None,
        "improved_overlap_mean": float(np.mean(improved_overlap)) if improved_overlap else None,
        "baseline_overlap_ge_60": int(sum(value >= 0.6 for value in baseline_overlap)),
        "improved_overlap_ge_60": int(sum(value >= 0.6 for value in improved_overlap)),
        "sample_count": int(len(sample_frame)),
        "fixed_error_count": int((sample_frame["category"] == "fixed_error").sum()),
    }
    save_json(summary, XAI_DIR / "xai_summary.json")
    save_text(
        "# XAI Summary\n\n"
        + dataframe_to_markdown(pd.DataFrame([summary]))
        + "\n\n## Case Summary\n\n"
        + dataframe_to_markdown(case_frame),
        XAI_DIR / "xai_summary.md",
    )
    save_json(
        {
            "baseline_shap": baseline_shap,
            "baseline_lime": baseline_lime,
            "improved_shap": improved_shap,
            "improved_lime": improved_lime,
        },
        XAI_DIR / "xai_details.json",
    )
    return summary
