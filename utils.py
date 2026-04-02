"""Shared utilities for reproducible hate-speech experiments."""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"
RUNS_DIR = OUTPUT_DIR / "runs"
TUNING_DIR = OUTPUT_DIR / "tuning"
XAI_DIR = OUTPUT_DIR / "xai"

LABEL_NAMES = ["hatespeech", "offensive", "normal"]
LABEL2ID = {name: index for index, name in enumerate(LABEL_NAMES)}
ID2LABEL = {index: name for name, index in LABEL2ID.items()}
NUM_LABELS = len(LABEL_NAMES)
VADER_COLUMNS = ["pos", "neg", "neu", "compound"]

for directory in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, REPORT_DIR, RUNS_DIR, TUNING_DIR, XAI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Prefer Apple MPS when available, otherwise CUDA, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.mps, "manual_seed") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def clear_device_cache() -> None:
    """Release cached accelerator memory when possible."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def slugify(value: str) -> str:
    """Convert text into a filesystem-friendly slug."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "artifact"


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_serializable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_json(data: Any, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """Save JSON data under the outputs directory."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(data), handle, ensure_ascii=False, indent=2)
    return target


def load_json(filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Any:
    """Load JSON data from disk."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data: Any, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """Save a Python object as pickle."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "wb") as handle:
        pickle.dump(data, handle)
    return target


def load_pickle(filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Any:
    """Load a pickle artifact."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    with open(target, "rb") as handle:
        return pickle.load(handle)


def save_dataframe(frame: pd.DataFrame, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """Persist a DataFrame as CSV."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)
    return target


def save_text(text: str, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """Save plain text."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return target


def remove_tree(path: Path | str) -> None:
    """Delete a directory tree if it exists."""
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


def compute_class_weight_tensor(labels: Iterable[int], imbalance_threshold: float = 0.10) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Return balanced class weights only when the minority ratio is below threshold."""
    label_array = np.asarray(list(labels))
    counts = np.bincount(label_array, minlength=NUM_LABELS)
    ratios = counts / max(counts.sum(), 1)
    minority_ratio = float(ratios.min()) if len(ratios) else 0.0
    use_weights = minority_ratio < imbalance_threshold

    metadata = {
        "counts": {LABEL_NAMES[index]: int(count) for index, count in enumerate(counts)},
        "ratios": {LABEL_NAMES[index]: float(ratio) for index, ratio in enumerate(ratios)},
        "minority_ratio": minority_ratio,
        "use_class_weight": use_weights,
        "threshold": imbalance_threshold,
    }

    if not use_weights:
        return None, metadata

    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_LABELS), y=label_array)
    metadata["weights"] = {LABEL_NAMES[index]: float(weight) for index, weight in enumerate(weights)}
    return torch.tensor(weights, dtype=torch.float32), metadata


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    """Compute report-friendly classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_precision = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_LABELS)),
        zero_division=0,
    )
    per_class_recall = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_LABELS)),
        zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(NUM_LABELS)),
        zero_division=0,
    )

    auroc = None
    if y_prob is not None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = None

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "auroc": auroc,
        "per_class_precision": {LABEL_NAMES[i]: float(per_class_precision[i]) for i in range(NUM_LABELS)},
        "per_class_recall": {LABEL_NAMES[i]: float(per_class_recall[i]) for i in range(NUM_LABELS)},
        "per_class_f1": {LABEL_NAMES[i]: float(per_class_f1[i]) for i in range(NUM_LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS))).tolist(),
    }


def format_mean_std(mean_value: float | None, std_value: float | None) -> str:
    """Format a mean ± std value for tables."""
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def aggregate_run_metrics(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-run metrics into mean/std summary rows."""
    metric_columns = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "auroc"]
    grouped_rows = []

    for model_name, group in pd.DataFrame(run_records).groupby("model"):
        row: dict[str, Any] = {"model": model_name, "runs": int(len(group))}
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_display"] = "N/A"
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
                row[f"{metric}_display"] = format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])

        for metric_name in ["per_class_f1", "per_class_precision", "per_class_recall"]:
            for label in LABEL_NAMES:
                values = group[f"{metric_name}.{label}"].to_numpy(dtype=float)
                row[f"{metric_name}.{label}_mean"] = float(values.mean())
                row[f"{metric_name}.{label}_std"] = float(values.std(ddof=0))

        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


class EarlyStopping:
    """Simple early stopping helper."""

    def __init__(self, patience: int = 2, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def update(self, value: float) -> bool:
        """Update the stopper with a new validation value."""
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path, labels: list[str] | None = None) -> None:
    """Save a confusion matrix heatmap."""
    labels = labels or LABEL_NAMES
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_learning_curves(history_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    """Save train/validation loss and F1 curves."""
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="train_loss")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], marker="o", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history_frame["epoch"], history_frame["val_macro_f1"], marker="o", color="#dd8452")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_split_distribution(splits: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save split-wise label distribution chart."""
    ensure_dir(output_path.parent)
    rows = []
    for split_name, frame in splits.items():
        counts = frame["label"].value_counts().reindex(range(NUM_LABELS), fill_value=0)
        for label_index, count in counts.items():
            rows.append({"split": split_name, "label": LABEL_NAMES[label_index], "count": int(count)})
    plot_frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_frame, x="split", y="count", hue="label", ax=ax)
    ax.set_title("Split-wise Label Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_comparison(summary_frame: pd.DataFrame, output_path: Path) -> None:
    """Save a report-ready metric comparison figure with error bars."""
    ensure_dir(output_path.parent)
    metrics = [
        ("macro_f1_mean", "macro_f1_std", "Macro F1"),
        ("macro_precision_mean", "macro_precision_std", "Macro Precision"),
        ("macro_recall_mean", "macro_recall_std", "Macro Recall"),
        ("accuracy_mean", "accuracy_std", "Accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for axis, (mean_col, std_col, title) in zip(axes, metrics):
        axis.bar(
            summary_frame["model"],
            summary_frame[mean_col],
            yerr=summary_frame[std_col],
            color="#4c72b0",
            edgecolor="black",
            capsize=5,
        )
        axis.set_title(title)
        axis.set_ylim(0.0, 1.05)
        axis.tick_params(axis="x", rotation=30)
        for index, value in enumerate(summary_frame[mean_col]):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison Across Repeated Runs")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_per_class_heatmap(summary_frame: pd.DataFrame, output_path: Path) -> None:
    """Save a per-class F1 heatmap using aggregated mean scores."""
    ensure_dir(output_path.parent)
    heatmap_data = []
    for _, row in summary_frame.iterrows():
        heatmap_data.append([row[f"per_class_f1.{label}_mean"] for label in LABEL_NAMES])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        np.asarray(heatmap_data),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=LABEL_NAMES,
        yticklabels=summary_frame["model"].tolist(),
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Per-Class F1 Mean Across Runs")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple markdown table."""
    if frame.empty:
        return "| Empty |\n| --- |"

    headers = [str(column) for column in frame.columns]
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def flatten_run_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metric dictionaries for CSV export."""
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat
