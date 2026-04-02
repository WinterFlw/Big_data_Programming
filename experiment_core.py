"""Core experiment pipeline aligned with the report specification."""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import (
    BASE_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    ID2LABEL,
    LABEL2ID,
    LABEL_NAMES,
    NUM_LABELS,
    OUTPUT_DIR,
    REPORT_DIR,
    RUNS_DIR,
    TUNING_DIR,
    VADER_COLUMNS,
    XAI_DIR,
    EarlyStopping,
    aggregate_run_metrics,
    clear_device_cache,
    compute_class_weight_tensor,
    compute_metrics,
    dataframe_to_markdown,
    ensure_dir,
    flatten_run_record,
    get_device,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_metric_comparison,
    plot_per_class_heatmap,
    plot_split_distribution,
    save_dataframe,
    save_json,
    save_pickle,
    save_text,
    set_seed,
    slugify,
)


@dataclass
class ExperimentConfig:
    """Configuration used across the end-to-end experiment pipeline."""

    split_train: float = 0.70
    split_val: float = 0.10
    split_test: float = 0.20
    max_len: int = 128
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    dropout: float = 0.10
    mlp_hidden: int = 256
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 1e-4
    imbalance_threshold: float = 0.10
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])
    tune_learning_rates: list[float] = field(default_factory=lambda: [1e-5, 2e-5, 3e-5])
    tune_batch_sizes: list[int] = field(default_factory=lambda: [64])
    tune_dropouts: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    tune_epochs: list[int] = field(default_factory=lambda: [5])
    tuning_seed: int = 42
    tuning_max_epochs: int = 5
    xai_sample_size: int = 24
    lime_num_features: int = 5
    lime_num_samples: int = 500
    shap_max_evals: int = 300
    shap_batch_size: int = 32


DEFAULT_CONFIG = ExperimentConfig()

RAW_DATASET_PATH = DATA_DIR / "dataset.json"
RAW_SPLIT_PATH = DATA_DIR / "post_id_divisions.json"
SPLITS_PICKLE_PATH = OUTPUT_DIR / "data_splits.pkl"
VADER_PICKLE_PATH = OUTPUT_DIR / "vader_features.pkl"
CONFIG_PATH = OUTPUT_DIR / "experiment_config.json"
BEST_MODELS_PATH = REPORT_DIR / "best_models.json"
BENCHMARK_RUNS_PATH = REPORT_DIR / "benchmark_runs.csv"
BENCHMARK_SUMMARY_PATH = REPORT_DIR / "benchmark_summary.csv"
BENCHMARK_MARKDOWN_PATH = REPORT_DIR / "benchmark_summary.md"
FREEZE_STUDY_PATH = REPORT_DIR / "freeze_study.csv"
FREEZE_STUDY_MARKDOWN_PATH = REPORT_DIR / "freeze_study.md"
TUNING_LOG_PATH = TUNING_DIR / "transformer_tuning_log.csv"
TUNING_SUMMARY_PATH = TUNING_DIR / "transformer_tuning_best.json"
DATA_PROFILE_PATH = REPORT_DIR / "data_profile.json"


def get_config() -> ExperimentConfig:
    """Load the active config from disk if available."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return ExperimentConfig(**data)
    save_json(asdict(DEFAULT_CONFIG), CONFIG_PATH)
    return DEFAULT_CONFIG


def save_config(config: ExperimentConfig) -> None:
    """Persist the active experiment config."""
    save_json(asdict(config), CONFIG_PATH)


def ensure_raw_hatexplain(force_download: bool = False) -> None:
    """Download HateXplain raw files when missing."""
    base_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url_name, destination in [
        ("dataset.json", RAW_DATASET_PATH),
        ("post_id_divisions.json", RAW_SPLIT_PATH),
    ]:
        if force_download or not destination.exists():
            print(f"Downloading {url_name}...")
            urlretrieve(base_url + url_name, destination)


def _majority_label(sample: dict[str, Any]) -> int | None:
    annotator_labels = [annotator["label"] for annotator in sample["annotators"]]
    label_counts = Counter(annotator_labels)
    majority_label, majority_count = label_counts.most_common(1)[0]
    if majority_count < 2:
        return None
    return LABEL2ID.get(majority_label)


def _collect_targets(sample: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    for annotator in sample["annotators"]:
        if "target" in annotator:
            targets.extend(annotator["target"])
    return sorted(set(targets)) if targets else ["None"]


def load_processed_dataframe(force_download: bool = False) -> pd.DataFrame:
    """Download and preprocess HateXplain into a single DataFrame."""
    ensure_raw_hatexplain(force_download=force_download)

    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as handle:
        raw_dataset = json.load(handle)

    records = []
    for post_id, sample in raw_dataset.items():
        label = _majority_label(sample)
        if label is None:
            continue
        text = " ".join(sample["post_tokens"])
        text = re.sub(r"@\S+", "<user>", text)
        records.append(
            {
                "post_id": post_id,
                "text": text,
                "label": label,
                "label_name": ID2LABEL[label],
                "targets": _collect_targets(sample),
            }
        )

    frame = pd.DataFrame(records)
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return frame


def prepare_data(config: ExperimentConfig | None = None, force_refresh: bool = False, force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Create the 70/10/20 stratified splits and save dataset diagnostics."""
    config = config or get_config()
    save_config(config)

    if SPLITS_PICKLE_PATH.exists() and not force_refresh:
        return pd.read_pickle(SPLITS_PICKLE_PATH)

    frame = load_processed_dataframe(force_download=force_download)

    train_val_df, test_df = train_test_split(
        frame,
        test_size=config.split_test,
        random_state=config.tuning_seed,
        stratify=frame["label"],
    )
    val_relative_size = config.split_val / (config.split_train + config.split_val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=config.tuning_seed,
        stratify=train_val_df["label"],
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    profile = {
        "total_samples": int(len(frame)),
        "splits": {name: int(len(df)) for name, df in splits.items()},
        "label_distribution": {
            name: {
                LABEL_NAMES[index]: int(count)
                for index, count in df["label"].value_counts().reindex(range(NUM_LABELS), fill_value=0).items()
            }
            for name, df in splits.items()
        },
        "split_ratio": {
            "train": config.split_train,
            "val": config.split_val,
            "test": config.split_test,
        },
    }

    pd.to_pickle(splits, SPLITS_PICKLE_PATH)
    save_json(profile, DATA_PROFILE_PATH)
    plot_split_distribution(splits, REPORT_DIR / "split_distribution.png")

    profile_rows = []
    for split_name, distribution in profile["label_distribution"].items():
        row = {"split": split_name, "samples": profile["splits"][split_name]}
        for label_name, count in distribution.items():
            row[label_name] = count
        profile_rows.append(row)
    save_dataframe(pd.DataFrame(profile_rows), REPORT_DIR / "split_distribution.csv")
    save_text(
        "# Data Profile\n\n"
        + dataframe_to_markdown(pd.DataFrame(profile_rows)),
        REPORT_DIR / "data_profile.md",
    )
    return splits


def load_splits() -> dict[str, pd.DataFrame]:
    """Load preprocessed train/val/test splits."""
    if not SPLITS_PICKLE_PATH.exists():
        return prepare_data()
    return pd.read_pickle(SPLITS_PICKLE_PATH)


def extract_vader_features(
    splits: dict[str, pd.DataFrame] | None = None,
    force_refresh: bool = False,
) -> dict[str, np.ndarray]:
    """Compute VADER features for every split."""
    splits = splits or load_splits()
    if VADER_PICKLE_PATH.exists() and not force_refresh:
        return pd.read_pickle(VADER_PICKLE_PATH)

    analyzer = SentimentIntensityAnalyzer()
    features: dict[str, np.ndarray] = {}

    for split_name, frame in splits.items():
        rows = []
        for text in frame["text"].tolist():
            scores = analyzer.polarity_scores(text)
            rows.append([scores[column] for column in VADER_COLUMNS])
        features[split_name] = np.asarray(rows, dtype=np.float32)

        save_dataframe(
            pd.DataFrame(features[split_name], columns=VADER_COLUMNS),
            REPORT_DIR / f"vader_{split_name}.csv",
        )

    pd.to_pickle(features, VADER_PICKLE_PATH)
    return features


class TransformerTextDataset(Dataset):
    """Tokenized text dataset for transformer-only models."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


class HybridTextDataset(Dataset):
    """Dataset for transformer + VADER hybrid models."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        vader_features: np.ndarray,
        tokenizer,
        max_len: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vader = torch.tensor(vader_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "vader": self.vader[index],
        }


class TransformerCLSClassifier(nn.Module):
    """Transformer baseline using the CLS token and a linear classifier."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class HybridSentimentClassifier(nn.Module):
    """Transformer CLS embedding concatenated with 4 VADER features."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(hidden_size + len(VADER_COLUMNS), hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, vader: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([pooled_output, vader], dim=1)
        combined = self.dropout(combined)
        combined = self.hidden(combined)
        combined = self.relu(combined)
        return self.out(combined)


def _build_collate_fn(tokenizer) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = data_collator(model_features)
        batch["labels"] = torch.stack([feature["labels"] for feature in features])
        if "vader" in features[0]:
            batch["vader"] = torch.stack([feature["vader"] for feature in features])
        return batch

    return collate_fn


def _make_loader(dataset: Dataset, tokenizer, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_build_collate_fn(tokenizer),
    )


def build_transformer_datasets(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> tuple[Any, dict[str, Dataset]]:
    """Build tokenizer and datasets for a transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = {
        name: TransformerTextDataset(df["text"].tolist(), df["label"].to_numpy(), tokenizer, config.max_len)
        for name, df in splits.items()
    }
    return tokenizer, datasets


def build_hybrid_datasets(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    vader_features: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[Any, dict[str, Dataset]]:
    """Build tokenizer and datasets for a transformer+VADER model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = {
        name: HybridTextDataset(
            df["text"].tolist(),
            df["label"].to_numpy(),
            vader_features[name],
            tokenizer,
            config.max_len,
        )
        for name, df in splits.items()
    }
    return tokenizer, datasets


def _forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if "vader" in batch:
        vader = batch["vader"].to(device)
        return model(input_ids=input_ids, attention_mask=attention_mask, vader=vader)
    return model(input_ids=input_ids, attention_mask=attention_mask)


def evaluate_neural_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a transformer-based model and return metrics and loss."""
    model.eval()
    losses = []
    predictions = []
    labels = []
    probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            batch_labels = batch["labels"].to(device)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, batch_labels)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            losses.append(loss.item())
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


def train_neural_model(
    model_name: str,
    display_name: str,
    dataset_builder: Callable[[ExperimentConfig], tuple[Any, dict[str, Dataset]]],
    model_factory: Callable[[ExperimentConfig], nn.Module],
    config: ExperimentConfig,
    seed: int,
    hyperparams: dict[str, Any] | None = None,
    output_root: Path | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Train a neural model for one seed and save run artifacts."""
    hyperparams = hyperparams or {}
    run_config = ExperimentConfig(**{**asdict(config), **hyperparams})
    set_seed(seed)
    device = get_device()
    print(
        f"[train] {display_name} | seed={seed} | device={device} | batch={run_config.batch_size} | "
        f"lr={run_config.learning_rate} | dropout={run_config.dropout} | epochs={run_config.epochs}",
        flush=True,
    )
    tokenizer, datasets = dataset_builder(run_config)

    loaders = {
        "train": _make_loader(datasets["train"], tokenizer, run_config.batch_size, True, seed),
        "val": _make_loader(datasets["val"], tokenizer, run_config.batch_size, False, seed),
        "test": _make_loader(datasets["test"], tokenizer, run_config.batch_size, False, seed),
    }

    model = model_factory(run_config).to(device)
    class_weight_tensor, class_weight_meta = compute_class_weight_tensor(
        datasets["train"].labels.numpy(),
        imbalance_threshold=run_config.imbalance_threshold,
    )
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    total_steps = len(loaders["train"]) * run_config.epochs
    warmup_steps = int(total_steps * run_config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_dir = ensure_dir(output_root or (RUNS_DIR / slugify(display_name) / f"seed_{seed}"))
    checkpoint_path = CHECKPOINT_DIR / f"{slugify(display_name)}_seed_{seed}.pt"
    early_stopping = EarlyStopping(
        patience=run_config.early_stopping_patience,
        mode="min",
        min_delta=run_config.early_stopping_min_delta,
    )

    history_rows = []
    best_val_macro_f1 = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, run_config.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            batch_labels = batch["labels"].to(device)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_metrics = evaluate_neural_model(model, loaders["val"], criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
            }
        )
        print(
            f"[epoch] {display_name} | seed={seed} | epoch={epoch}/{run_config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            print(
                f"[best] {display_name} | seed={seed} | epoch={epoch} | "
                f"best_val_macro_f1={best_val_macro_f1:.4f}",
                flush=True,
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": model_name,
                    "display_name": display_name,
                    "seed": seed,
                    "hyperparams": asdict(run_config),
                },
                checkpoint_path,
            )

        if early_stopping.update(val_metrics["loss"]):
            print(
                f"[early-stop] {display_name} | seed={seed} | epoch={epoch} | "
                f"best_epoch={best_epoch} | best_val_macro_f1={best_val_macro_f1:.4f}",
                flush=True,
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if evaluate_test:
        final_metrics = evaluate_neural_model(model, loaders["test"], criterion, device)
    else:
        final_metrics = {
            "accuracy": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "auroc": None,
            "per_class_precision": {label: None for label in LABEL_NAMES},
            "per_class_recall": {label: None for label in LABEL_NAMES},
            "per_class_f1": {label: None for label in LABEL_NAMES},
            "confusion_matrix": None,
        }
    elapsed_seconds = time.time() - start_time

    history_frame = pd.DataFrame(history_rows)
    save_dataframe(history_frame, run_dir / "history.csv")
    plot_learning_curves(history_frame, f"{display_name} (seed={seed})", run_dir / "learning_curve.png")
    predictions_path = None
    if evaluate_test and final_metrics["confusion_matrix"] is not None:
        plot_confusion_matrix(
            np.asarray(final_metrics["confusion_matrix"]),
            f"{display_name} (seed={seed})",
            run_dir / "confusion_matrix.png",
        )
        predictions_path = save_pickle(
            {
                "y_true": final_metrics["y_true"],
                "y_pred": final_metrics["y_pred"],
                "y_prob": final_metrics["y_prob"],
            },
            run_dir / "predictions.pkl",
            directory=BASE_DIR,
        )

    metrics_to_save = {
        "display_name": display_name,
        "model_name": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "elapsed_seconds": elapsed_seconds,
        "class_weight": class_weight_meta,
        "hyperparams": asdict(run_config),
        "metrics": {
            key: value
            for key, value in final_metrics.items()
            if key not in {"y_true", "y_pred", "y_prob"}
        },
        "prediction_artifact": str(predictions_path) if predictions_path else None,
    }
    save_json(metrics_to_save, run_dir / "metrics.json", directory=BASE_DIR)

    result_record = {
        "model": display_name,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "elapsed_seconds": elapsed_seconds,
        "prediction_artifact": str(predictions_path) if predictions_path else "",
        "history_path": str(run_dir / "history.csv"),
        "run_dir": str(run_dir),
        "hyperparams": asdict(run_config),
        **{
            key: value
            for key, value in final_metrics.items()
            if key not in {"y_true", "y_pred", "y_prob"}
        },
    }

    model.to("cpu")
    clear_device_cache()
    if evaluate_test and final_metrics["macro_f1"] is not None:
        print(
            f"[done] {display_name} | seed={seed} | test_macro_f1={final_metrics['macro_f1']:.4f} | "
            f"accuracy={final_metrics['accuracy']:.4f}",
            flush=True,
        )
    else:
        print(
            f"[done] {display_name} | seed={seed} | tuning-only run complete | "
            f"best_val_macro_f1={best_val_macro_f1:.4f}",
            flush=True,
        )
    return result_record


def _predict_ml_bundle(bundle: dict[str, Any], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = bundle["tfidf"].transform(texts)
    predictions = bundle["model"].predict(features)
    if hasattr(bundle["model"], "predict_proba"):
        probabilities = bundle["model"].predict_proba(features)
    else:
        probabilities = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    return predictions, probabilities


def run_tfidf_baselines(
    splits: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Train TF-IDF + LR / SVM baselines with report-friendly outputs."""
    seeds = seeds or config.seeds
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].to_numpy()
    val_texts = splits["val"]["text"].tolist()
    val_labels = splits["val"]["label"].to_numpy()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].to_numpy()

    model_specs = {
        "TF-IDF + LR": {
            "factory": lambda c, seed, class_weight: LogisticRegression(
                C=c,
                max_iter=2000,
                class_weight=class_weight,
                random_state=seed,
                multi_class="auto",
            ),
            "candidates": [0.5, 1.0, 2.0],
        },
        "TF-IDF + SVM": {
            "factory": lambda c, seed, class_weight: LinearSVC(
                C=c,
                class_weight=class_weight,
                random_state=seed,
                max_iter=10000,
            ),
            "candidates": [0.5, 1.0, 2.0],
        },
    }

    _, class_weight_meta = compute_class_weight_tensor(train_labels, config.imbalance_threshold)
    use_class_weight = class_weight_meta["use_class_weight"]
    class_weight = "balanced" if use_class_weight else None

    results = []
    for seed in seeds:
        set_seed(seed)
        for display_name, spec in model_specs.items():
            print(f"[ml] {display_name} | seed={seed} | tuning C over {spec['candidates']}", flush=True)
            vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True)
            x_train = vectorizer.fit_transform(train_texts)
            x_val = vectorizer.transform(val_texts)
            x_test = vectorizer.transform(test_texts)

            best_bundle = None
            best_val_f1 = -1.0
            best_c = None

            for candidate_c in spec["candidates"]:
                print(f"[ml-candidate] {display_name} | seed={seed} | C={candidate_c}", flush=True)
                base_model = spec["factory"](candidate_c, seed, class_weight)
                if display_name.endswith("SVM"):
                    candidate_model = CalibratedClassifierCV(base_model, cv=3)
                else:
                    candidate_model = base_model

                candidate_model.fit(x_train, train_labels)
                val_pred = candidate_model.predict(x_val)
                val_prob = candidate_model.predict_proba(x_val)
                val_metrics = compute_metrics(val_labels, val_pred, val_prob)
                print(
                    f"[ml-val] {display_name} | seed={seed} | C={candidate_c} | "
                    f"val_macro_f1={val_metrics['macro_f1']:.4f}",
                    flush=True,
                )
                if val_metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["macro_f1"]
                    best_bundle = {"tfidf": vectorizer, "model": candidate_model}
                    best_c = candidate_c

            assert best_bundle is not None
            test_pred, test_prob = _predict_ml_bundle(best_bundle, test_texts)
            test_metrics = compute_metrics(test_labels, test_pred, test_prob)
            print(
                f"[ml-done] {display_name} | seed={seed} | best_C={best_c} | "
                f"test_macro_f1={test_metrics['macro_f1']:.4f}",
                flush=True,
            )

            run_dir = ensure_dir(RUNS_DIR / slugify(display_name) / f"seed_{seed}")
            save_pickle(best_bundle, run_dir / "model_bundle.pkl", directory=BASE_DIR)
            save_json(
                {
                    "display_name": display_name,
                    "seed": seed,
                    "best_c": best_c,
                    "best_val_macro_f1": best_val_f1,
                    "class_weight": class_weight_meta,
                    "metrics": test_metrics,
                },
                run_dir / "metrics.json",
                directory=BASE_DIR,
            )
            plot_confusion_matrix(
                np.asarray(test_metrics["confusion_matrix"]),
                f"{display_name} (seed={seed})",
                run_dir / "confusion_matrix.png",
            )

            results.append(
                {
                    "model": display_name,
                    "seed": seed,
                    "best_c": best_c,
                    "best_val_macro_f1": best_val_f1,
                    "checkpoint_path": str(run_dir / "model_bundle.pkl"),
                    "prediction_artifact": str(run_dir / "metrics.json"),
                    "elapsed_seconds": 0.0,
                    "history_path": "",
                    "run_dir": str(run_dir),
                    "hyperparams": {
                        "batch_size": None,
                        "learning_rate": None,
                        "dropout": None,
                        "epochs": None,
                    },
                    **test_metrics,
                }
            )

    return results


def _ml_summary_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in run_records:
        rows.append(flatten_run_record(record))
    return rows


def _transformer_dataset_builder(
    config: ExperimentConfig,
    model_name: str,
) -> tuple[Any, dict[str, Dataset]]:
    splits = load_splits()
    return build_transformer_datasets(model_name, splits, config)


def _hybrid_dataset_builder(
    config: ExperimentConfig,
    model_name: str,
) -> tuple[Any, dict[str, Dataset]]:
    splits = load_splits()
    vader_features = extract_vader_features(splits)
    return build_hybrid_datasets(model_name, splits, vader_features, config)


def run_transformer_benchmark(
    config: ExperimentConfig | None = None,
    tuned_params: dict[str, dict[str, Any]] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run repeated experiments for the report-aligned transformer models."""
    config = config or get_config()
    seeds = seeds or config.seeds
    tuned_params = tuned_params or load_tuned_hyperparams()

    model_specs = [
        {
            "display_name": "BERT-base",
            "model_name": "bert-base-uncased",
            "builder": lambda local_config: _transformer_dataset_builder(local_config, "bert-base-uncased"),
            "factory": lambda local_config: TransformerCLSClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
            ),
            "tuning_key": "BERT-base",
        },
        {
            "display_name": "BERT+VADER",
            "model_name": "bert-base-uncased",
            "builder": lambda local_config: _hybrid_dataset_builder(local_config, "bert-base-uncased"),
            "factory": lambda local_config: HybridSentimentClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
                freeze_encoder=False,
            ),
            "tuning_key": "BERT+VADER",
        },
        {
            "display_name": "RoBERTa+VADER",
            "model_name": "roberta-base",
            "builder": lambda local_config: _hybrid_dataset_builder(local_config, "roberta-base"),
            "factory": lambda local_config: HybridSentimentClassifier(
                model_name="roberta-base",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
                freeze_encoder=False,
            ),
            "tuning_key": "RoBERTa+VADER",
        },
    ]

    records = []
    for seed in seeds:
        for spec in model_specs:
            model_hparams = tuned_params.get(spec["tuning_key"], {})
            record = train_neural_model(
                model_name=spec["model_name"],
                display_name=spec["display_name"],
                dataset_builder=spec["builder"],
                model_factory=spec["factory"],
                config=config,
                seed=seed,
                hyperparams=model_hparams,
            )
            records.append(record)
    return records


def run_freeze_study(
    config: ExperimentConfig | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Compare frozen vs fine-tuned encoder for BERT+VADER."""
    config = config or get_config()
    seeds = seeds or config.seeds
    tuned_hyperparams = load_tuned_hyperparams().get("BERT+VADER", {})

    records = []
    for freeze_encoder in [True, False]:
        variant_name = "BERT+VADER (Frozen Encoder)" if freeze_encoder else "BERT+VADER (Fine-tuned Encoder)"
        for seed in seeds:
            record = train_neural_model(
                model_name="bert-base-uncased",
                display_name=variant_name,
                dataset_builder=lambda local_config: _hybrid_dataset_builder(local_config, "bert-base-uncased"),
                model_factory=lambda local_config, freeze_encoder=freeze_encoder: HybridSentimentClassifier(
                    model_name="bert-base-uncased",
                    dropout=local_config.dropout,
                    hidden_dim=local_config.mlp_hidden,
                    freeze_encoder=freeze_encoder,
                ),
                config=config,
                seed=seed,
                hyperparams=tuned_hyperparams,
            )
            records.append(record)

    flat_rows = [flatten_run_record(record) for record in records]
    freeze_frame = pd.DataFrame(flat_rows)
    save_dataframe(freeze_frame, FREEZE_STUDY_PATH)

    summary = aggregate_run_metrics(flat_rows)
    markdown_frame = summary[["model", "macro_f1_display", "macro_precision_display", "macro_recall_display", "accuracy_display"]]
    save_text(
        "# Encoder Freeze Study\n\n"
        + "Tuned hyperparameters reused from `BERT+VADER` benchmark settings.\n\n"
        + dataframe_to_markdown(markdown_frame),
        FREEZE_STUDY_MARKDOWN_PATH,
    )
    return summary


def _select_best_runs(run_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_runs: dict[str, dict[str, Any]] = {}
    for record in run_records:
        current_best = best_runs.get(record["model"])
        if current_best is None or record["best_val_macro_f1"] > current_best["best_val_macro_f1"]:
            best_runs[record["model"]] = record
    return best_runs


def save_benchmark_artifacts(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Persist repeated benchmark outputs and report-friendly summaries."""
    flat_runs = [flatten_run_record(record) for record in run_records]
    run_frame = pd.DataFrame(flat_runs)
    save_dataframe(run_frame, BENCHMARK_RUNS_PATH)

    summary_frame = aggregate_run_metrics(flat_runs)
    save_dataframe(summary_frame, BENCHMARK_SUMMARY_PATH)
    save_text(
        "# Benchmark Summary\n\n"
        + dataframe_to_markdown(
            summary_frame[
                [
                    "model",
                    "macro_f1_display",
                    "macro_precision_display",
                    "macro_recall_display",
                    "accuracy_display",
                    "auroc_display",
                ]
            ]
        ),
        BENCHMARK_MARKDOWN_PATH,
    )

    plot_metric_comparison(summary_frame, REPORT_DIR / "model_comparison.png")
    plot_per_class_heatmap(summary_frame, REPORT_DIR / "per_class_f1_heatmap.png")

    best_runs = _select_best_runs(run_records)
    save_json(best_runs, BEST_MODELS_PATH)
    return summary_frame


def load_tuned_hyperparams() -> dict[str, dict[str, Any]]:
    """Load tuned hyperparameters if available, otherwise return defaults."""
    if not TUNING_SUMMARY_PATH.exists():
        return {
            "BERT-base": {},
            "BERT+VADER": {},
            "RoBERTa+VADER": {},
        }
    with open(TUNING_SUMMARY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _tune_single_model(
    tuning_key: str,
    base_config: ExperimentConfig,
    dataset_builder: Callable[[ExperimentConfig], tuple[Any, dict[str, Dataset]]],
    model_factory: Callable[[ExperimentConfig], nn.Module],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sequential tuning over learning rate, batch size, dropout, and epoch budget."""
    search_history: list[dict[str, Any]] = []
    tuned_values = {
        "learning_rate": base_config.learning_rate,
        "batch_size": base_config.batch_size,
        "dropout": base_config.dropout,
        "epochs": base_config.epochs,
    }

    search_plan = [
        ("learning_rate", base_config.tune_learning_rates),
        ("batch_size", base_config.tune_batch_sizes),
        ("dropout", base_config.tune_dropouts),
        ("epochs", base_config.tune_epochs),
    ]

    print(f"[tuning-model] {tuning_key} | start", flush=True)
    for parameter_name, candidates in search_plan:
        if len(candidates) <= 1:
            print(
                f"[tuning-skip] {tuning_key} | parameter={parameter_name} | fixed={candidates[0]}",
                flush=True,
            )
            tuned_values[parameter_name] = candidates[0]
            continue
        best_candidate = tuned_values[parameter_name]
        best_score = -1.0
        print(
            f"[tuning-param] {tuning_key} | parameter={parameter_name} | "
            f"candidates={list(candidates)} | current={tuned_values[parameter_name]}",
            flush=True,
        )
        for candidate in candidates:
            hyperparams = dict(tuned_values)
            hyperparams[parameter_name] = candidate
            print(
                f"[tuning-candidate] {tuning_key} | parameter={parameter_name} | candidate={candidate}",
                flush=True,
            )
            tuning_result = train_neural_model(
                model_name=tuning_key.lower(),
                display_name=f"{tuning_key} Tuning [{parameter_name}={candidate}]",
                dataset_builder=dataset_builder,
                model_factory=model_factory,
                config=base_config,
                seed=base_config.tuning_seed,
                hyperparams=hyperparams,
                output_root=TUNING_DIR / slugify(tuning_key) / f"{parameter_name}_{candidate}",
                evaluate_test=False,
            )
            search_history.append(
                {
                    "model": tuning_key,
                    "parameter": parameter_name,
                    "candidate": candidate,
                    "val_macro_f1": tuning_result["best_val_macro_f1"],
                    "seed": base_config.tuning_seed,
                }
            )
            print(
                f"[tuning-score] {tuning_key} | parameter={parameter_name} | "
                f"candidate={candidate} | val_macro_f1={tuning_result['best_val_macro_f1']:.4f}",
                flush=True,
            )
            if tuning_result["best_val_macro_f1"] > best_score:
                best_score = tuning_result["best_val_macro_f1"]
                best_candidate = candidate
        tuned_values[parameter_name] = best_candidate
        print(
            f"[tuning-best] {tuning_key} | parameter={parameter_name} | "
            f"best_candidate={best_candidate} | best_val_macro_f1={best_score:.4f}",
            flush=True,
        )

    print(f"[tuning-model] {tuning_key} | done | selected={tuned_values}", flush=True)
    return tuned_values, search_history


def run_hyperparameter_tuning(config: ExperimentConfig | None = None) -> dict[str, dict[str, Any]]:
    """Run sequential hyperparameter tuning and save a log for the report."""
    config = config or get_config()
    prepare_data(config)
    extract_vader_features(force_refresh=False)

    tuning_specs = [
        (
            "BERT-base",
            lambda local_config: _transformer_dataset_builder(local_config, "bert-base-uncased"),
            lambda local_config: TransformerCLSClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
            ),
        ),
        (
            "BERT+VADER",
            lambda local_config: _hybrid_dataset_builder(local_config, "bert-base-uncased"),
            lambda local_config: HybridSentimentClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
            ),
        ),
        (
            "RoBERTa+VADER",
            lambda local_config: _hybrid_dataset_builder(local_config, "roberta-base"),
            lambda local_config: HybridSentimentClassifier(
                model_name="roberta-base",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
            ),
        ),
    ]

    all_history = []
    tuned_summary: dict[str, dict[str, Any]] = {}

    for tuning_key, dataset_builder, model_factory in tuning_specs:
        print(f"[tuning] Running sequential search for {tuning_key}", flush=True)
        tuned_values, history = _tune_single_model(tuning_key, config, dataset_builder, model_factory)
        all_history.extend(history)
        tuned_summary[tuning_key] = tuned_values

    history_frame = pd.DataFrame(all_history)
    save_dataframe(history_frame, TUNING_LOG_PATH)
    save_json(tuned_summary, TUNING_SUMMARY_PATH)
    save_text(
        "# Hyperparameter Tuning Summary\n\n"
        + dataframe_to_markdown(pd.DataFrame([{"model": key, **value} for key, value in tuned_summary.items()])),
        TUNING_DIR / "transformer_tuning_best.md",
    )
    return tuned_summary


def run_benchmark(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """Run the full repeated benchmark suite and save final artifacts."""
    config = config or get_config()
    prepare_data(config)
    extract_vader_features()

    tfidf_runs = run_tfidf_baselines(load_splits(), config)
    transformer_runs = run_transformer_benchmark(config=config)
    all_runs = tfidf_runs + transformer_runs
    return save_benchmark_artifacts(all_runs)


def describe_status() -> dict[str, Any]:
    """Return the current pipeline status."""
    return {
        "config_exists": CONFIG_PATH.exists(),
        "data_ready": SPLITS_PICKLE_PATH.exists(),
        "vader_ready": VADER_PICKLE_PATH.exists(),
        "tuning_ready": TUNING_LOG_PATH.exists(),
        "benchmark_ready": BENCHMARK_SUMMARY_PATH.exists(),
        "freeze_study_ready": FREEZE_STUDY_PATH.exists(),
        "xai_ready": (XAI_DIR / "xai_summary.json").exists(),
        "dashboard_ready": (OUTPUT_DIR / "dashboard" / "index.html").exists(),
    }
