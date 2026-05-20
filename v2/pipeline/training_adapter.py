"""Training adapter that connects v2-local model code to the v2 run contract.

The expensive model implementation lives under v2/runtime. This adapter keeps
the pipeline package focused on orchestration: it maps one RunUnit to the
correct runtime condition, captures logs, and normalizes artifacts so
downstream v2 stages can read stable filenames.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import pickle
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import unit_status
from .manifest import manifest_hash
from .paths import BASE_DIR, experiment_root
from .schema import RunUnit


RUNTIME_DIR = BASE_DIR / "runtime"


def _load_runtime_core() -> Any:
    """Import v2/runtime experiment_core after making runtime importable.

    Runtime modules import `utils` as a top-level module, so the runtime
    directory must be on sys.path before importing experiment_core. The
    insertion is local and idempotent.
    """
    runtime_path = str(RUNTIME_DIR)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    for module_name in ["utils", "experiment_core"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", "") if module else ""
        if module_file and not str(module_file).startswith(runtime_path):
            del sys.modules[module_name]
    import experiment_core  # type: ignore[import-not-found]

    return experiment_core


def _condition_spec(runtime_core: Any, condition: str) -> Any:
    """Return the runtime ConditionSpec for a v2 condition name."""
    for spec in runtime_core.V2_CONDITION_SPECS:
        if spec.condition == condition:
            return spec
    raise ValueError(f"Runtime condition spec not found: {condition}")


def _family_hyperparams(manifest: dict[str, Any], family: str) -> dict[str, Any]:
    """Read common hyperparameters for one model family from the manifest."""
    key = "bert_common_hyperparams" if family == "BERT" else "roberta_common_hyperparams"
    return dict(manifest["benchmark"].get(key, {}))


def _build_runtime_config(runtime_core: Any, manifest: dict[str, Any], spec: Any, seed: int) -> Any:
    """Create an ExperimentConfig aligned with one v2 RunUnit."""
    config = runtime_core.ExperimentConfig()
    config.v2_enabled = True
    config.seeds = [int(seed)]
    config.attention_loss_alpha = float(manifest["benchmark"].get("attention_loss_alpha", 0.0))
    config.target_loss_beta = float(manifest["benchmark"].get("target_loss_beta", 0.0))

    for key, value in _family_hyperparams(manifest, spec.family).items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _condition_hyperparams(manifest: dict[str, Any], spec: Any) -> dict[str, Any]:
    """Map v2 condition flags to train_neural_model hyperparameters."""
    hyperparams = _family_hyperparams(manifest, spec.family)
    hyperparams["attention_loss_alpha"] = (
        float(manifest["benchmark"].get("attention_loss_alpha", 0.0))
        if spec.use_attention_loss
        else 0.0
    )
    hyperparams["target_loss_beta"] = (
        float(manifest["benchmark"].get("target_loss_beta", 0.0))
        if getattr(spec, "use_target_aux", False)
        else 0.0
    )
    return hyperparams


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return path


def _copy_checkpoint(source_value: str | None, unit: RunUnit) -> Path | None:
    """Move the runtime checkpoint into the v2 benchmark checkpoint directory.

    Runtime training first writes the best checkpoint under v2/checkpoints/.
    The v2 contract reads checkpoints from outputs/.../benchmark/checkpoints/.
    Moving instead of copying avoids keeping two large .pt files per run on the
    RunPod network volume.
    """
    if not source_value:
        return None
    source = Path(source_value)
    if not source.exists():
        return None
    target = experiment_root(unit.run_id) / "benchmark" / "checkpoints" / f"{unit.condition.lower()}_seed_{unit.seed}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(source), str(target))
    return target


def _xai_minimal_checkpoint_needed(manifest: dict[str, Any], unit: RunUnit) -> bool:
    """Return whether this checkpoint is needed by the current XAI stages.

    Primary XAI uses all seeds for its primary model pair. Deep/Ablation XAI in
    the current implementation use the middle manifest seed. This lets 5090
    runs avoid storing all 120 checkpoints while still preserving the planned
    XAI evidence path.
    """
    benchmark_seeds = [int(seed) for seed in manifest["benchmark"]["seeds"]]
    middle_seed = benchmark_seeds[len(benchmark_seeds) // 2]
    xai_config = manifest.get("xai", {})

    primary_models = set(xai_config.get("primary", {}).get("models", []))
    if unit.condition in primary_models:
        return True

    deep_models = set(xai_config.get("deep", {}).get("models", []))
    ablation_models = set(xai_config.get("ablation", {}).get("models", []))
    if unit.seed == middle_seed and unit.condition in (deep_models | ablation_models):
        return True
    return False


def _apply_checkpoint_retention(manifest: dict[str, Any], unit: RunUnit, checkpoint_path: Path | None) -> Path | None:
    """Apply CHECKPOINT_RETENTION to save network storage during long runs.

    Policies:
      keep-all     Keep every checkpoint.
      xai-minimal  Keep only checkpoints needed by configured XAI stages.
      none         Delete checkpoints after metrics/predictions are normalized.
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        return checkpoint_path

    policy = os.environ.get("CHECKPOINT_RETENTION", "keep-all").strip().lower()
    if policy in {"", "keep-all", "all"}:
        return checkpoint_path
    if policy not in {"xai-minimal", "none"}:
        raise ValueError(
            "CHECKPOINT_RETENTION must be one of keep-all, xai-minimal, none "
            f"(got {policy!r})"
        )
    if policy == "none" or not _xai_minimal_checkpoint_needed(manifest, unit):
        checkpoint_path.unlink()
        return None
    return checkpoint_path


def _normalize_predictions(prediction_artifact: str | None, unit: RunUnit) -> Path | None:
    """Convert the runtime predictions pickle into a CSV for v2 inference output."""
    if not prediction_artifact:
        return None
    source = Path(prediction_artifact)
    if not source.exists():
        return None

    with open(source, "rb") as handle:
        payload = pickle.load(handle)

    rows = []
    y_true = payload.get("y_true", [])
    y_pred = payload.get("y_pred", [])
    y_prob = payload.get("y_prob", [])
    for index, (true_label, predicted_label) in enumerate(zip(y_true, y_pred)):
        row = {
            "index": index,
            "true_label": int(true_label),
            "predicted_label": int(predicted_label),
        }
        if len(y_prob) > index:
            for label_index, probability in enumerate(y_prob[index]):
                row[f"prob_{label_index}"] = float(probability)
        rows.append(row)

    target = unit.run_dir / "predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = ["index", "true_label", "predicted_label", "prob_0", "prob_1", "prob_2"]
    with open(target, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return target


def _weighted_f1_from_predictions(prediction_artifact: str | None) -> float | None:
    """Compute weighted F1 when runtime metrics do not provide it directly."""
    if not prediction_artifact:
        return None
    source = Path(prediction_artifact)
    if not source.exists():
        return None
    with open(source, "rb") as handle:
        payload = pickle.load(handle)
    y_true = payload.get("y_true")
    y_pred = payload.get("y_pred")
    if y_true is None or y_pred is None:
        return None
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _write_run_config(
    manifest: dict[str, Any],
    unit: RunUnit,
    spec: Any,
    hyperparams: dict[str, Any],
) -> Path:
    """Write the per-unit config required by v2 resume/status checks."""
    return _write_json(
        unit.config_path,
        {
            "run_id": unit.run_id,
            "condition": unit.condition,
            "seed": unit.seed,
            "manifest_hash": manifest_hash(manifest),
            "model_name": spec.model_name,
            "family": spec.family,
            "use_attention_loss": spec.use_attention_loss,
            "use_sentiment": spec.use_vader,
            "use_target_aux": getattr(spec, "use_target_aux", False),
            "hyperparams": hyperparams,
        },
    )


def _write_normalized_metrics(
    manifest: dict[str, Any],
    unit: RunUnit,
    spec: Any,
    runtime_record: dict[str, Any],
    checkpoint_path: Path | None,
    predictions_path: Path | None,
) -> Path:
    """Write the flat metrics schema consumed by v2 aggregate/report stages."""
    runtime_metrics_path = unit.run_dir / "runtime_metrics.json"
    if unit.metrics_path.exists():
        shutil.copy2(unit.metrics_path, runtime_metrics_path)

    prediction_artifact = runtime_record.get("prediction_artifact")
    weighted_f1 = _weighted_f1_from_predictions(prediction_artifact)
    metrics = {
        "run_id": unit.run_id,
        "condition": unit.condition,
        "seed": unit.seed,
        "model_name": spec.model_name,
        "backbone": spec.family,
        "use_attention_loss": spec.use_attention_loss,
        "use_sentiment": spec.use_vader,
        "train_seconds": runtime_record.get("elapsed_seconds", ""),
        "best_epoch": runtime_record.get("best_epoch", ""),
        "macro_f1": runtime_record.get("macro_f1", ""),
        "weighted_f1": weighted_f1 if weighted_f1 is not None else "",
        "accuracy": runtime_record.get("accuracy", ""),
        "precision_macro": runtime_record.get("macro_precision", ""),
        "recall_macro": runtime_record.get("macro_recall", ""),
        "loss": runtime_record.get("loss", ""),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else runtime_record.get("checkpoint_path", ""),
        "metrics_path": str(unit.metrics_path),
        "predictions_path": str(predictions_path) if predictions_path else prediction_artifact or "",
        "runtime_metrics_path": str(runtime_metrics_path) if runtime_metrics_path.exists() else "",
    }
    return _write_json(unit.metrics_path, metrics)


def execute_run_unit(
    manifest: dict[str, Any],
    unit: RunUnit,
    *,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one v2 benchmark unit through the v2-local runtime code."""
    if resume and not force and unit_status(unit) == "completed":
        return {"status": "skipped", "condition": unit.condition, "seed": unit.seed, "run_dir": unit.run_dir}

    runtime_core = _load_runtime_core()
    spec = _condition_spec(runtime_core, unit.condition)
    config = _build_runtime_config(runtime_core, manifest, spec, unit.seed)
    hyperparams = _condition_hyperparams(manifest, spec)

    unit.run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = unit.run_dir / "stdout.log"
    stderr_path = unit.run_dir / "stderr.log"

    try:
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(stderr_path, "w", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
                record = runtime_core.train_neural_model(
                    model_name=spec.model_name,
                    display_name=spec.condition,
                    dataset_builder=lambda local_config, spec=spec: runtime_core._condition_dataset_builder(local_config, spec),
                    model_factory=runtime_core._condition_model_factory(spec),
                    config=config,
                    seed=unit.seed,
                    hyperparams=hyperparams,
                    output_root=unit.run_dir,
                    evaluate_test=True,
                )
    except Exception:
        with open(stderr_path, "a", encoding="utf-8") as stderr_handle:
            traceback.print_exc(file=stderr_handle)
        raise

    record["condition"] = spec.condition
    record["family"] = spec.family
    record["use_attention_loss"] = spec.use_attention_loss
    record["use_vader"] = spec.use_vader

    checkpoint_path = _copy_checkpoint(record.get("checkpoint_path"), unit)
    checkpoint_path = _apply_checkpoint_retention(manifest, unit, checkpoint_path)
    predictions_path = _normalize_predictions(record.get("prediction_artifact"), unit)
    _write_run_config(manifest, unit, spec, hyperparams)
    _write_normalized_metrics(manifest, unit, spec, record, checkpoint_path, predictions_path)

    return {
        "status": "completed",
        "condition": unit.condition,
        "seed": unit.seed,
        "run_dir": unit.run_dir,
        "metrics": unit.metrics_path,
        "checkpoint": checkpoint_path,
        "predictions": predictions_path,
    }
