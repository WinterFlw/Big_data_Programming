"""
학습/벤치마크 모듈 (model/train.py)
====================================
신경망 학습 루프, 평가, TF-IDF 베이스라인, 8조건 벤치마크, 하이퍼파라미터
튜닝, freeze study, 파이프라인 상태 점검을 담아요.
ucam의 `model/ucam.py`(train_model/model_compile)에 대응하는 학습 로직이에요.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.config import ExperimentConfig, get_config, save_config
from src.data import (
    TransformerTextDataset,
    HybridTextDataset,
    prepare_data,
    load_splits,
    load_processed_dataframe,
    _has_v2_columns,
)
from src.vader import extract_vader_features
from src.path import (
    BASE_DIR,
    RAW_DATASET_PATH,
    RAW_SPLIT_PATH,
    SPLITS_PICKLE_PATH,
    VADER_PICKLE_PATH,
    CONFIG_PATH,
    BEST_MODELS_PATH,
    BENCHMARK_RUNS_PATH,
    BENCHMARK_SUMMARY_PATH,
    BENCHMARK_MARKDOWN_PATH,
    FREEZE_STUDY_PATH,
    FREEZE_STUDY_MARKDOWN_PATH,
    TUNING_LOG_PATH,
    TUNING_SUMMARY_PATH,
    DATA_PROFILE_PATH,
    OUTPUT_DIR,
    REPORT_DIR,
    RUNS_DIR,
    TUNING_DIR,
    XAI_DIR,
    CHECKPOINT_DIR,
)
from model.models import (
    TransformerCLSClassifier,
    HybridSentimentClassifier,
    TransformerMLPClassifier,
    TransformerConditionClassifier,
    ConditionSpec,
    V2_CONDITION_SPECS,
)
from src.utils import (
    LABEL_NAMES,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    VADER_COLUMNS,
    EarlyStopping,
    aggregate_run_metrics,
    clear_device_cache,
    compute_class_weight_tensor,
    compute_factorial_anova,
    compute_metrics,
    compute_pairwise_significance,
    compute_subgroup_metrics,
    dataframe_to_markdown,
    ensure_dir,
    flatten_run_record,
    format_mean_std,
    get_device,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_metric_comparison,
    plot_per_class_heatmap,
    plot_split_distribution,
    primary_target_label,
    save_dataframe,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_text,
    set_seed,
    slugify,
    _artifact_current,
)


# ╔══════════════════════════════════════════════════════════╗
# ║  데이터 로딩 유틸리티 — 배치를 만드는 똑똑한 도우미들     ║
# ╚══════════════════════════════════════════════════════════╝

# ── collate 함수 ────────────────────────────────
# DataLoader가 여러 샘플을 하나의 배치로 묶을 때 이 함수를 사용해요.
# 길이가 다른 시퀀스들을 패딩으로 맞춰주는 역할이에요!
def _build_collate_fn(tokenizer) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    # HuggingFace의 DataCollatorWithPadding이 패딩을 자동으로 처리해줘요
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # input_ids와 attention_mask만 추출해서 패딩 처리
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = data_collator(model_features)
        # 라벨도 배치에 추가!
        batch["labels"] = torch.stack([feature["labels"] for feature in features])
        # 하이브리드 모델이면 VADER 피처도 추가해요
        if "vader" in features[0]:
            batch["vader"] = torch.stack([feature["vader"] for feature in features])
        if "rationale_mask" in features[0]:
            batch["rationale_mask"] = pad_sequence(
                [feature["rationale_mask"] for feature in features],
                batch_first=True,
                padding_value=0.0,
            )
        if "target_multilabel" in features[0]:
            batch["target_multilabel"] = torch.stack([feature["target_multilabel"] for feature in features])
        return batch

    return collate_fn


# ── DataLoader 생성 ─────────────────────────────
# 시드를 고정해서 셔플 순서도 재현 가능하게 만들어요!
def _make_loader(dataset: Dataset, tokenizer, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)  # 셔플 순서를 시드로 고정 (재현성!)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,       # train은 True, val/test는 False
        generator=generator,
        collate_fn=_build_collate_fn(tokenizer),
    )


# ── 데이터셋 빌더 함수들 ────────────────────────
# 모델 타입에 맞는 데이터셋을 만들어주는 팩토리 함수들이에요
def build_transformer_datasets(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    config: ExperimentConfig,
) -> tuple[Any, dict[str, Dataset]]:
    """트랜스포머 모델용 토크나이저와 데이터셋을 만들어요."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = {
        name: TransformerTextDataset(
            df["text"].tolist(),
            df["label"].to_numpy(),
            tokenizer,
            config.max_len,
            post_tokens=df["post_tokens"].tolist() if "post_tokens" in df.columns else None,
            rationale_masks=df["rationale_mask"].tolist() if "rationale_mask" in df.columns else None,
            target_vectors=df["target_multilabel"].tolist() if "target_multilabel" in df.columns else None,
        )
        for name, df in splits.items()
    }
    return tokenizer, datasets


def build_hybrid_datasets(
    model_name: str,
    splits: dict[str, pd.DataFrame],
    vader_features: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[Any, dict[str, Dataset]]:
    """트랜스포머 + VADER 하이브리드 모델용 데이터셋을 만들어요. VADER 피처가 추가돼요!"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = {
        name: HybridTextDataset(
            df["text"].tolist(),
            df["label"].to_numpy(),
            vader_features[name],  # 여기서 VADER 감성 피처를 같이 넘겨요
            tokenizer,
            config.max_len,
            post_tokens=df["post_tokens"].tolist() if "post_tokens" in df.columns else None,
            rationale_masks=df["rationale_mask"].tolist() if "rationale_mask" in df.columns else None,
            target_vectors=df["target_multilabel"].tolist() if "target_multilabel" in df.columns else None,
        )
        for name, df in splits.items()
    }
    return tokenizer, datasets


# ── 배치 순전파 헬퍼 ────────────────────────────
# baseline과 hybrid 모델 모두 호환되게 forward를 호출해요
def _forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    return_outputs: bool = False,
    output_attentions: bool = False,
) -> torch.Tensor | dict[str, Any]:
    """배치 데이터를 GPU/MPS로 보내고 모델에 통과시켜요. VADER 유무를 자동 감지!"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "vader" in batch:
        # 하이브리드 모델: VADER 피처도 같이 넘겨요
        kwargs["vader"] = batch["vader"].to(device)
    if return_outputs:
        try:
            return model(
                **kwargs,
                output_attentions=output_attentions,
                return_dict=True,
            )
        except TypeError:
            return {"logits": model(**kwargs), "target_logits": None, "attentions": None}
    return model(**kwargs)


def _extract_logits(outputs: torch.Tensor | dict[str, Any]) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs["logits"]
    return outputs


def _compute_attention_supervision_loss(outputs: dict[str, Any], batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """CLS attention 분포가 human rationale 위치에 정렬되도록 보조 손실을 계산합니다."""
    attentions = outputs.get("attentions")
    if not attentions or "rationale_mask" not in batch:
        return torch.tensor(0.0, device=device)

    last_attention = attentions[-1]
    cls_attention = last_attention.mean(dim=1)[:, 0, :]
    target = batch["rationale_mask"].to(device).float()
    attention_mask = batch["attention_mask"].to(device).float()
    has_rationale = (target.sum(dim=1) > 0).float().unsqueeze(1)
    valid_mask = attention_mask * has_rationale
    if valid_mask.sum() <= 0:
        return torch.tensor(0.0, device=device)
    loss = F.binary_cross_entropy(
        cls_attention.clamp(1e-6, 1.0 - 1e-6),
        target,
        reduction="none",
    )
    return (loss * valid_mask).sum() / valid_mask.sum()


def _compute_target_aux_loss(outputs: dict[str, Any], batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    target_logits = outputs.get("target_logits")
    if target_logits is None or "target_multilabel" not in batch:
        return torch.tensor(0.0, device=device)
    target = batch["target_multilabel"].to(device).float()
    return F.binary_cross_entropy_with_logits(target_logits, target)


# ╔══════════════════════════════════════════════════════════╗
# ║  모델 평가 — 실력을 측정하는 시험 시간이에요!             ║
# ╚══════════════════════════════════════════════════════════╝
def evaluate_neural_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """트랜스포머 모델을 평가해서 메트릭과 손실을 반환해요."""
    model.eval()  # 평가 모드로 전환 (Dropout 비활성화)
    losses = []
    predictions = []
    labels = []
    probabilities = []

    # 평가할 때는 gradient 계산이 필요 없어요 (메모리 절약!)
    with torch.no_grad():
        for batch in dataloader:
            batch_labels = batch["labels"].to(device)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, batch_labels)
            # softmax로 확률 변환 후, 가장 높은 확률의 클래스를 예측으로 선택
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            losses.append(loss.item())
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())

    # numpy 배열로 변환해서 메트릭 계산!
    y_true = np.asarray(labels)
    y_pred = np.asarray(predictions)
    y_prob = np.asarray(probabilities)
    metrics = compute_metrics(y_true, y_pred, y_prob)  # F1, precision, recall 등
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    # 나중에 혼동행렬 등에 활용할 수 있도록 원본 예측 결과도 함께 반환해요
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


# ╔══════════════════════════════════════════════════════════╗
# ║  🎯 여기서부터 실험의 심장부! 모델 학습 루프예요          ║
# ╚══════════════════════════════════════════════════════════╝
# 이 함수 하나가 데이터 준비 → 학습 → 평가 → 결과 저장까지 전부 해요.
# 논문의 Table 결과가 바로 여기서 나오는 거예요!
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
    """
    하나의 시드에 대해 트랜스포머 모델을 학습하고 모든 산출물을 저장해요.

    학습 흐름을 단계별로 보면:
      1단계: 데이터셋 구축 (토크나이즈해서 모델이 이해할 수 있게!)
      2단계: 클래스 가중치 계산 (데이터가 불균형하면 소수 클래스에 더 큰 가중치)
      3단계: AdamW 옵티마이저 + linear warmup 스케줄러를 세팅해요
      4단계: 에포크 루프 — 학습 -> 검증 -> 체크포인트 저장 -> early stopping 체크
      5단계: 가장 좋았던 체크포인트를 불러와서 테스트셋 최종 평가!
      6단계: 학습곡선, 혼동행렬, 메트릭 JSON 등 모든 결과물 저장
    """
    # ── Step 0: 설정 준비 ─────────────────────────
    # 기본 config에 하이퍼파라미터 오버라이드를 적용해요
    hyperparams = hyperparams or {}
    run_config = ExperimentConfig(**{**asdict(config), **hyperparams})
    set_seed(seed)  # 재현성을 위해 모든 랜덤 시드를 고정!
    device = get_device()  # GPU/MPS/CPU 중 사용 가능한 디바이스 선택
    print(
        f"[train] {display_name} | seed={seed} | device={device} | batch={run_config.batch_size} | "
        f"lr={run_config.learning_rate} | dropout={run_config.dropout} | epochs={run_config.epochs}",
        flush=True,
    )

    # ── Step 1: 데이터셋 구축 ─────────────────────
    # 텍스트를 토크나이즈하고 DataLoader를 만들어요
    tokenizer, datasets = dataset_builder(run_config)

    loaders = {
        "train": _make_loader(datasets["train"], tokenizer, run_config.batch_size, True, seed),   # 셔플 O
        "val": _make_loader(datasets["val"], tokenizer, run_config.batch_size, False, seed),       # 셔플 X
        "test": _make_loader(datasets["test"], tokenizer, run_config.batch_size, False, seed),     # 셔플 X
    }

    # ── Step 2: 모델 & 손실함수 준비 ────────────
    model = model_factory(run_config).to(device)  # 모델을 디바이스로 올려요
    # 클래스 불균형이 있으면 소수 클래스에 높은 가중치를 줘서 공정하게!
    class_weight_tensor, class_weight_meta = compute_class_weight_tensor(
        datasets["train"].labels.numpy(),
        imbalance_threshold=run_config.imbalance_threshold,
    )
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)
    # label_smoothing=0.1: hate/offensive 경계가 모호한 샘플에서
    # hard label(0 또는 1)보다 soft label이 일반화에 도움돼요.
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=0.1)

    # ── Step 3: 옵티마이저 & 스케줄러 세팅 ──────
    # AdamW: Adam에 L2 정규화(weight decay)를 올바르게 적용한 버전이에요
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    total_steps = len(loaders["train"]) * run_config.epochs
    warmup_steps = int(total_steps * run_config.warmup_ratio)
    # Linear warmup: 처음엔 lr을 천천히 올리고, 이후 서서히 줄여요
    # 갑자기 큰 lr로 시작하면 학습이 불안정해질 수 있거든요!
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Step 4: 체크포인트 & 조기 종료 준비 ─────
    run_dir = ensure_dir(output_root or (RUNS_DIR / slugify(display_name) / f"seed_{seed}"))
    checkpoint_path = CHECKPOINT_DIR / f"{slugify(display_name)}_seed_{seed}.pt"
    ensure_dir(checkpoint_path.parent)
    # Early Stopping: val loss가 개선되지 않으면 학습을 멈춰요
    early_stopping = EarlyStopping(
        patience=run_config.early_stopping_patience,
        mode="min",  # loss는 작을수록 좋으니까 "min"!
        min_delta=run_config.early_stopping_min_delta,
    )

    # ── Step 5: 학습 기록 초기화 ──────────────────
    history_rows = []         # 에포크별 loss, f1 등을 기록할 리스트
    best_val_macro_f1 = -1.0  # 지금까지의 최고 검증 F1 점수
    best_epoch = 0            # 최고 성능을 기록한 에포크 번호
    start_time = time.time()  # 학습 시간 측정 시작!

    # ╔══════════════════════════════════════════════════════╗
    # ║  에포크 루프 — 여기서 실제로 모델이 배워요!           ║
    # ╚══════════════════════════════════════════════════════╝
    for epoch in range(1, run_config.epochs + 1):
        # ── 학습 단계 (Training) ────────────────
        model.train()  # 학습 모드 (Dropout 활성화)
        train_losses = []
        train_attention_losses = []
        train_target_losses = []
        for batch in loaders["train"]:
            batch_labels = batch["labels"].to(device)
            # 순전파: 모델에 데이터를 넣어 예측값을 얻어요
            needs_aux_outputs = run_config.attention_loss_alpha > 0 or run_config.target_loss_beta > 0
            outputs = _forward_batch(
                model,
                batch,
                device,
                return_outputs=needs_aux_outputs,
                output_attentions=run_config.attention_loss_alpha > 0,
            )
            logits = _extract_logits(outputs)
            # 손실 계산: 예측과 정답의 차이를 수치화해요
            loss = criterion(logits, batch_labels)
            if isinstance(outputs, dict) and run_config.attention_loss_alpha > 0:
                attn_loss = _compute_attention_supervision_loss(outputs, batch, device)
                loss = loss + run_config.attention_loss_alpha * attn_loss
                train_attention_losses.append(float(attn_loss.detach().cpu()))
            if isinstance(outputs, dict) and run_config.target_loss_beta > 0:
                target_loss = _compute_target_aux_loss(outputs, batch, device)
                loss = loss + run_config.target_loss_beta * target_loss
                train_target_losses.append(float(target_loss.detach().cpu()))

            # 역전파: gradient를 계산하고 가중치를 업데이트해요
            optimizer.zero_grad()       # 이전 gradient 초기화
            loss.backward()             # 역전파로 gradient 계산
            # gradient clipping: gradient가 너무 커지는 걸 방지해요 (안정성!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()            # 가중치 업데이트
            scheduler.step()            # learning rate 스케줄 한 칸 진행
            train_losses.append(loss.item())

        # ── 검증 단계 (Validation) ──────────────
        # 한 에포크가 끝나면 검증 데이터로 성능을 체크해요
        val_metrics = evaluate_neural_model(model, loaders["val"], criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        # 학습 곡선을 그리기 위해 매 에포크 기록을 남겨요
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "train_attention_loss": float(np.mean(train_attention_losses)) if train_attention_losses else 0.0,
                "train_target_loss": float(np.mean(train_target_losses)) if train_target_losses else 0.0,
            }
        )
        print(
            f"[epoch] {display_name} | seed={seed} | epoch={epoch}/{run_config.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        # ── 최고 성능 갱신 & 체크포인트 저장 ────
        # val F1이 역대 최고면 모델 가중치를 저장해둬요 (나중에 불러올 거예요!)
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

        # ── Early Stopping 체크 ─────────────────
        # val loss가 계속 나빠지면 "더 학습해도 소용없다"고 판단해서 멈춰요
        if early_stopping.update(val_metrics["loss"]):
            print(
                f"[early-stop] {display_name} | seed={seed} | epoch={epoch} | "
                f"best_epoch={best_epoch} | best_val_macro_f1={best_val_macro_f1:.4f}",
                flush=True,
            )
            break

    # ── Step 6: 최고 체크포인트로 최종 평가 ──────
    # 학습 중 가장 좋았던 가중치를 다시 불러와서 테스트셋으로 최종 평가해요
    # (마지막 에포크가 아니라 "최고의 순간"을 사용하는 게 포인트!)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if evaluate_test:
        final_metrics = evaluate_neural_model(model, loaders["test"], criterion, device)
    else:
        # 튜닝 모드에서는 테스트셋 평가를 건너뛰어요 (데이터 오염 방지!)
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
    elapsed_seconds = time.time() - start_time  # 총 학습 시간 계산

    # ── Step 7: 결과물 저장 — 보고서에 쓸 자료들! ──
    # 학습 곡선 데이터와 시각화를 저장해요
    history_frame = pd.DataFrame(history_rows)
    save_dataframe(history_frame, run_dir / "history.csv")
    plot_learning_curves(history_frame, f"{display_name} (seed={seed})", run_dir / "learning_curve.png")
    # 테스트셋 예측 결과를 저장해요 (혼동행렬, 예측 확률 등)
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

    # 메트릭을 JSON으로 정리해서 저장 (나중에 보고서 작성할 때 필요!)
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

    # 벤치마크 집계에 쓸 결과 레코드를 만들어요
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

    # ── 마무리: 메모리 정리 ────────────────────────
    # 모델을 CPU로 옮기고 GPU/MPS 메모리를 해제해요 (다음 모델을 위해!)
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


# ── ML 모델 예측 헬퍼 ───────────────────────────
# TF-IDF 벡터화 + 모델 예측을 한 번에 해주는 편의 함수예요
def _predict_ml_bundle(bundle: dict[str, Any], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF 변환 후 ML 모델로 예측해요. 확률 출력도 함께 반환!"""
    features = bundle["tfidf"].transform(texts)      # 텍스트 -> TF-IDF 벡터
    predictions = bundle["model"].predict(features)   # 클래스 예측
    if hasattr(bundle["model"], "predict_proba"):
        probabilities = bundle["model"].predict_proba(features)  # 확률도 같이!
    else:
        probabilities = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    return predictions, probabilities


# ╔══════════════════════════════════════════════════════════╗
# ║  TF-IDF 베이스라인 — 전통 ML의 저력을 보여줄 시간!       ║
# ╚══════════════════════════════════════════════════════════╝
# 딥러닝만이 전부가 아니에요! TF-IDF + LR/SVM은 빠르고 해석 가능하며,
# 때로는 놀라울 정도로 좋은 성능을 보여준답니다.
# Davidson et al. (2017) 논문의 방법론을 참고했어요.
def run_tfidf_baselines(
    splits: dict[str, pd.DataFrame],
    config: ExperimentConfig,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    전통 ML 베이스라인: TF-IDF(1~3gram) + Logistic Regression / Linear SVM.
    C(정규화 강도) 파라미터를 검증셋 기준으로 선택하고 테스트셋으로 평가해요.
    """
    seeds = seeds or config.seeds
    # 각 분할에서 텍스트와 라벨을 추출해요
    train_texts = splits["train"]["text"].tolist()
    train_labels = splits["train"]["label"].to_numpy()
    val_texts = splits["val"]["text"].tolist()
    val_labels = splits["val"]["label"].to_numpy()
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].to_numpy()

    # 두 가지 ML 모델을 정의해요: LR과 SVM
    model_specs = {
        "TF-IDF + LR": {
            "factory": lambda c, seed, class_weight: LogisticRegression(
                C=c,
                max_iter=2000,
                class_weight=class_weight,
                random_state=seed,
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

    # 클래스 불균형 체크 — 필요하면 "balanced" 가중치를 사용해요
    _, class_weight_meta = compute_class_weight_tensor(train_labels, config.imbalance_threshold)
    use_class_weight = class_weight_meta["use_class_weight"]
    class_weight = "balanced" if use_class_weight else None

    results = []
    for seed in seeds:
        set_seed(seed)
        for display_name, spec in model_specs.items():
            print(f"[ml] {display_name} | seed={seed} | tuning C over {spec['candidates']}", flush=True)
            # TF-IDF: 단어 빈도를 가중치로 바꿔요 (1~3gram, 최대 50K 피처)
            vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 3), sublinear_tf=True)
            x_train = vectorizer.fit_transform(train_texts)
            x_val = vectorizer.transform(val_texts)

            # C 파라미터 최적값을 검증셋으로 찾아요 (간단한 그리드 서치!)
            best_bundle = None
            best_val_f1 = -1.0
            best_c = None

            for candidate_c in spec["candidates"]:
                print(f"[ml-candidate] {display_name} | seed={seed} | C={candidate_c}", flush=True)
                base_model = spec["factory"](candidate_c, seed, class_weight)
                if display_name.endswith("SVM"):
                    # SVM은 원래 확률을 못 주니까 CalibratedClassifierCV로 감싸요
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

            # 최적 C를 찾았으니, 이제 테스트셋으로 최종 평가!
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


# ── 헬퍼 함수들 ────────────────────────────────
def _ml_summary_rows(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """실행 기록을 보고서용 평탄한 행 형태로 변환해요."""
    rows = []
    for record in run_records:
        rows.append(flatten_run_record(record))
    return rows


# 아래 두 함수는 데이터셋 빌더를 간단하게 감싼 래퍼예요.
# train_neural_model에 넘겨줄 때 편하게 쓰려고 만들었어요!
def _transformer_dataset_builder(
    config: ExperimentConfig,
    model_name: str,
) -> tuple[Any, dict[str, Dataset]]:
    """순수 트랜스포머용 데이터셋 빌더 (VADER 없이)"""
    splits = load_splits()
    return build_transformer_datasets(model_name, splits, config)


def _hybrid_dataset_builder(
    config: ExperimentConfig,
    model_name: str,
) -> tuple[Any, dict[str, Dataset]]:
    """하이브리드(트랜스포머 + VADER)용 데이터셋 빌더"""
    splits = load_splits()
    vader_features = extract_vader_features(splits)
    return build_hybrid_datasets(model_name, splits, vader_features, config)


def _condition_dataset_builder(config: ExperimentConfig, spec: ConditionSpec) -> tuple[Any, dict[str, Dataset]]:
    if spec.use_vader:
        return _hybrid_dataset_builder(config, spec.model_name)
    return _transformer_dataset_builder(config, spec.model_name)


def _condition_hyperparams(spec: ConditionSpec, base_config: ExperimentConfig, tuned_values: dict[str, Any]) -> dict[str, Any]:
    hyperparams = dict(tuned_values)
    if spec.use_attention_loss:
        if "attention_loss_alpha" in hyperparams:
            hyperparams["attention_loss_alpha"] = float(hyperparams["attention_loss_alpha"])
        elif base_config.attention_loss_alpha > 0:
            hyperparams["attention_loss_alpha"] = float(base_config.attention_loss_alpha)
        else:
            hyperparams["attention_loss_alpha"] = 0.3
    else:
        hyperparams["attention_loss_alpha"] = 0.0
    if spec.use_target_aux:
        default_beta = next((beta for beta in base_config.beta_grid if beta > 0), 0.1)
        if "target_loss_beta" in hyperparams:
            hyperparams["target_loss_beta"] = float(hyperparams["target_loss_beta"])
        elif base_config.target_loss_beta > 0:
            hyperparams["target_loss_beta"] = float(base_config.target_loss_beta)
        else:
            hyperparams["target_loss_beta"] = float(default_beta)
    else:
        hyperparams["target_loss_beta"] = 0.0
    return hyperparams


def _condition_tuned_values(spec: ConditionSpec, tuned_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """조건별 튜닝값을 찾고, B_B alpha 전파 규칙을 fallback에서도 보장합니다."""
    if spec.condition in tuned_params:
        return dict(tuned_params[spec.condition])

    if spec.family == "BERT" and spec.use_vader:
        base_key = "BERT+VADER"
    elif spec.family == "BERT":
        base_key = "BERT+MLP" if "BERT+MLP" in tuned_params else "BERT-base"
    elif spec.use_vader:
        base_key = "RoBERTa+VADER"
    else:
        base_key = "A_R" if "A_R" in tuned_params else "RoBERTa+VADER"

    values = dict(tuned_params.get(base_key, {}))
    if spec.use_attention_loss and "B_B" in tuned_params:
        alpha = tuned_params["B_B"].get("attention_loss_alpha")
        if alpha is not None:
            values["attention_loss_alpha"] = alpha
    return values


def _condition_model_factory(spec: ConditionSpec) -> Callable[[ExperimentConfig], nn.Module]:
    def factory(local_config: ExperimentConfig) -> nn.Module:
        return TransformerConditionClassifier(
            model_name=spec.model_name,
            use_vader=spec.use_vader,
            dropout=local_config.dropout,
            hidden_dim=local_config.mlp_hidden,
            num_targets=len(local_config.target_labels) if spec.use_target_aux else 0,
        )

    return factory


# ╔══════════════════════════════════════════════════════════╗
# ║  트랜스포머 벤치마크 — 본격적인 모델 비교 실험!           ║
# ╚══════════════════════════════════════════════════════════╝
# 3개 모델(BERT-base, BERT+VADER, RoBERTa+VADER)을
# 여러 시드로 반복 실험해서 평균과 표준편차를 구해요.
# 이렇게 해야 "우연히 잘 나온 건 아닌지" 확인할 수 있어요!
def run_transformer_benchmark(
    config: ExperimentConfig | None = None,
    tuned_params: dict[str, dict[str, Any]] | None = None,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """보고서에 맞춰 트랜스포머 모델들의 반복 실험을 돌려요."""
    config = config or get_config()
    seeds = seeds or config.seeds
    tuned_params = tuned_params or load_tuned_hyperparams()

    if config.v2_enabled:
        records = []
        for seed in seeds:
            for spec in V2_CONDITION_SPECS:
                tuned_values = _condition_tuned_values(spec, tuned_params)
                model_hparams = _condition_hyperparams(spec, config, tuned_values)
                record = train_neural_model(
                    model_name=spec.model_name,
                    display_name=spec.condition,
                    dataset_builder=lambda local_config, spec=spec: _condition_dataset_builder(local_config, spec),
                    model_factory=_condition_model_factory(spec),
                    config=config,
                    seed=seed,
                    hyperparams=model_hparams,
                )
                record["condition"] = spec.condition
                record["family"] = spec.family
                record["use_attention_loss"] = spec.use_attention_loss
                record["use_vader"] = spec.use_vader
                record["use_target_aux"] = spec.use_target_aux
                record["is_aux_experiment"] = False
                record["benchmark_scope"] = "main_8_condition"
                records.append(record)

        aux_spec = ConditionSpec("D_B+Target", "BERT", "bert-base-uncased", True, True, True)
        aux_tuned_values = (
            tuned_params.get(aux_spec.condition)
            or _condition_tuned_values(next(item for item in V2_CONDITION_SPECS if item.condition == "D_B"), tuned_params)
        )
        aux_hyperparams = _condition_hyperparams(aux_spec, config, aux_tuned_values)
        if aux_hyperparams.get("target_loss_beta", 0.0) > 0:
            for seed in seeds:
                record = train_neural_model(
                    model_name=aux_spec.model_name,
                    display_name=aux_spec.condition,
                    dataset_builder=lambda local_config, spec=aux_spec: _condition_dataset_builder(local_config, spec),
                    model_factory=_condition_model_factory(aux_spec),
                    config=config,
                    seed=seed,
                    hyperparams=aux_hyperparams,
                )
                record["condition"] = aux_spec.condition
                record["family"] = aux_spec.family
                record["use_attention_loss"] = aux_spec.use_attention_loss
                record["use_vader"] = aux_spec.use_vader
                record["use_target_aux"] = aux_spec.use_target_aux
                record["is_aux_experiment"] = True
                record["benchmark_scope"] = "target_aux"
                records.append(record)
        return records

    # 실험할 모델 4가지를 정의해요 (ablation 포함!)
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
            # Ablation: VADER 없이 MLP만 키운 모델 — 파라미터 수 교란 변수 통제용!
            "display_name": "BERT+MLP",
            "model_name": "bert-base-uncased",
            "builder": lambda local_config: _transformer_dataset_builder(local_config, "bert-base-uncased"),
            "factory": lambda local_config: TransformerMLPClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
            ),
            "tuning_key": "BERT+MLP",
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

    # 모든 시드 x 모든 모델 조합을 돌려요 (3시드 x 3모델 = 9번!)
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


# ╔══════════════════════════════════════════════════════════╗
# ║  Freeze Study — 인코더를 얼리면 어떻게 될까?             ║
# ╚══════════════════════════════════════════════════════════╝
# 흥미로운 실험이에요! BERT의 가중치를 동결(freeze)하면 어떨까요?
# - 동결 시: BERT는 그냥 고정된 피처 추출기 역할만 하고,
#   MLP 분류 헤드 + VADER 피처만으로 학습해요
# - 미세조정 시: BERT도 함께 업데이트되어 과제에 맞게 적응해요
# 이 비교를 통해 BERT fine-tuning의 가치를 정량적으로 보여줄 수 있어요!
def run_freeze_study(
    config: ExperimentConfig | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """
    Encoder Freeze Study: BERT+VADER에서 encoder 동결 vs 미세조정을 비교해요.
    VADER 피처의 독립적 기여도와 BERT 미세조정의 시너지 효과를 분석할 수 있어요!
    """
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


# ── 최고 성능 런 선택 ───────────────────────────
# 여러 시드 중 검증 F1이 가장 높았던 런을 모델별로 골라요
def _select_best_runs(run_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """각 모델에서 val macro F1이 가장 높았던 런을 선택해요."""
    best_runs: dict[str, dict[str, Any]] = {}
    for record in run_records:
        current_best = best_runs.get(record["model"])
        if current_best is None or record["best_val_macro_f1"] > current_best["best_val_macro_f1"]:
            best_runs[record["model"]] = record
    return best_runs


# ── 벤치마크 결과 저장 ──────────────────────────
# 모든 실험 결과를 CSV, 마크다운, 시각화로 저장하는 함수예요.
# 보고서에 바로 쓸 수 있는 형태로 만들어줘요!
def save_benchmark_artifacts(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    """벤치마크 결과를 CSV, 마크다운, 시각화로 저장해요."""
    flat_runs = [flatten_run_record(record) for record in run_records]
    run_frame = pd.DataFrame(flat_runs)
    save_dataframe(run_frame, BENCHMARK_RUNS_PATH)  # 개별 런 기록

    summary_frame = aggregate_run_metrics(flat_runs)  # 평균 +/- 표준편차 요약
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

    # ── 통계적 유의성 검정 ──────────────────────────
    # 모든 모델 쌍에 대해 paired t-test 수행 (같은 시드끼리 비교!)
    sig_frame = compute_pairwise_significance(flat_runs, metric_key="macro_f1")
    if not sig_frame.empty:
        save_dataframe(sig_frame, REPORT_DIR / "significance_tests.csv")
        save_text(
            "# Statistical Significance Tests\n\n"
            "Paired t-test on macro F1 across seeds (alpha=0.05).\n"
            "⚠️ 3-seed 반복은 검정력이 낮으므로 해석에 주의가 필요합니다.\n\n"
            + dataframe_to_markdown(sig_frame),
            REPORT_DIR / "significance_tests.md",
        )

    if {"family", "use_attention_loss", "use_vader"}.issubset(run_frame.columns):
        factorial_frame = run_frame[run_frame["family"].isin(["BERT", "RoBERTa"])].copy()
        if "is_aux_experiment" in factorial_frame.columns:
            factorial_frame = factorial_frame[~factorial_frame["is_aux_experiment"].fillna(False).astype(bool)]
        anova_3way = compute_factorial_anova(
            factorial_frame,
            metric_key="macro_f1",
            factors=["family", "use_attention_loss", "use_vader"],
        )
        if not anova_3way.empty:
            save_dataframe(anova_3way, REPORT_DIR / "anova_3way.csv")
            save_text(
                "# 3-way ANOVA\n\n"
                "Macro F1 ~ family × attention loss × VADER.\n\n"
                + dataframe_to_markdown(anova_3way),
                REPORT_DIR / "anova_3way.md",
            )

        bert_frame = factorial_frame[factorial_frame["family"] == "BERT"]
        anova_2way = compute_factorial_anova(
            bert_frame,
            metric_key="macro_f1",
            factors=["use_attention_loss", "use_vader"],
        )
        if not anova_2way.empty:
            save_dataframe(anova_2way, REPORT_DIR / "anova_2way_bert.csv")
            save_text(
                "# BERT Family 2-way ANOVA\n\n"
                "Macro F1 ~ attention loss × VADER within BERT conditions.\n\n"
                + dataframe_to_markdown(anova_2way),
                REPORT_DIR / "anova_2way_bert.md",
            )

    return summary_frame


# ── 튜닝 결과 로드 ──────────────────────────────
def load_tuned_hyperparams() -> dict[str, dict[str, Any]]:
    """튜닝으로 찾은 최적 하이퍼파라미터를 불러와요. 없으면 기본값(빈 딕셔너리) 반환!"""
    if not TUNING_SUMMARY_PATH.exists():
        return {
            "BERT-base": {},
            "BERT+MLP": {},
            "BERT+VADER": {},
            "RoBERTa+VADER": {},
        }
    with open(TUNING_SUMMARY_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ╔══════════════════════════════════════════════════════════╗
# ║  하이퍼파라미터 튜닝 — 최적의 레시피를 찾아요!            ║
# ╚══════════════════════════════════════════════════════════╝
# 순차적 탐색(Sequential Search) 전략을 사용해요:
# lr를 먼저 최적화 -> 그 값을 고정하고 batch size 최적화 -> ... 이런 식으로!
# Grid Search보다 훨씬 효율적이에요 (모든 조합을 안 해도 되니까).
_TUNING_KEY_TO_MODEL_NAME = {
    "BERT-base": "bert-base-uncased",
    "BERT+MLP": "bert-base-uncased",
    "BERT+VADER": "bert-base-uncased",
    "RoBERTa+VADER": "roberta-base",
}


def _tune_single_model(
    tuning_key: str,
    base_config: ExperimentConfig,
    dataset_builder: Callable[[ExperimentConfig], tuple[Any, dict[str, Dataset]]],
    model_factory: Callable[[ExperimentConfig], nn.Module],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """한 모델의 하이퍼파라미터를 순차적으로 탐색해요: lr -> batch -> dropout -> epochs"""
    search_history: list[dict[str, Any]] = []  # 모든 시도 기록
    # 현재까지 찾은 최적값 (시작은 기본값으로)
    tuned_values = {
        "learning_rate": base_config.learning_rate,
        "batch_size": base_config.batch_size,
        "dropout": base_config.dropout,
        "epochs": base_config.epochs,
    }

    # 탐색 순서: lr -> batch -> dropout -> epochs
    # 앞에서 찾은 최적값을 뒤 탐색에 반영하는 게 순차 탐색의 핵심!
    search_plan = [
        ("learning_rate", base_config.tune_learning_rates),
        ("batch_size", base_config.tune_batch_sizes),
        ("dropout", base_config.tune_dropouts),
        ("epochs", base_config.tune_epochs),
    ]

    print(f"[tuning-model] {tuning_key} | start", flush=True)
    for parameter_name, candidates in search_plan:
        # 후보가 1개뿐이면 탐색할 것도 없으니 바로 고정해요
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
            # 지금까지 찾은 최적값 + 이번 후보를 조합해서 학습!
            hyperparams = dict(tuned_values)
            hyperparams[parameter_name] = candidate
            print(
                f"[tuning-candidate] {tuning_key} | parameter={parameter_name} | candidate={candidate}",
                flush=True,
            )
            tuning_result = train_neural_model(
                model_name=_TUNING_KEY_TO_MODEL_NAME.get(tuning_key, tuning_key.lower()),
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


def _tune_attention_alpha(base_config: ExperimentConfig, base_hyperparams: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """v2.1 규칙: B_B 조건에서 alpha를 결정하고 B/D 조건에 전파합니다."""
    spec = next(item for item in V2_CONDITION_SPECS if item.condition == "B_B")
    best_alpha = base_config.alpha_grid[0]
    best_score = -1.0
    history = []
    for alpha in base_config.alpha_grid:
        hyperparams = {**base_hyperparams, "attention_loss_alpha": alpha, "target_loss_beta": 0.0}
        result = train_neural_model(
            model_name=spec.model_name,
            display_name=f"B_B Alpha Tuning [alpha={alpha}]",
            dataset_builder=lambda local_config, spec=spec: _condition_dataset_builder(local_config, spec),
            model_factory=_condition_model_factory(spec),
            config=base_config,
            seed=base_config.tuning_seed,
            hyperparams=hyperparams,
            output_root=TUNING_DIR / "b_b_alpha" / f"alpha_{alpha}",
            evaluate_test=False,
        )
        score = result["best_val_macro_f1"]
        history.append({
            "model": "B_B",
            "parameter": "attention_loss_alpha",
            "candidate": alpha,
            "val_macro_f1": score,
            "seed": base_config.tuning_seed,
        })
        if score > best_score:
            best_score = score
            best_alpha = alpha
    selected = {**base_hyperparams, "attention_loss_alpha": best_alpha, "target_loss_beta": 0.0}
    return selected, history


def _tune_target_beta(base_config: ExperimentConfig, base_hyperparams: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """v2.1 부가 실험: D_B target auxiliary loss의 beta를 검증셋 기준으로 고릅니다."""
    spec = ConditionSpec("D_B+Target", "BERT", "bert-base-uncased", True, True, True)
    best_beta = base_config.beta_grid[0]
    best_score = -1.0
    history = []
    for beta in base_config.beta_grid:
        hyperparams = {**base_hyperparams, "target_loss_beta": beta}
        result = train_neural_model(
            model_name=spec.model_name,
            display_name=f"D_B+Target Beta Tuning [beta={beta}]",
            dataset_builder=lambda local_config, spec=spec: _condition_dataset_builder(local_config, spec),
            model_factory=_condition_model_factory(spec),
            config=base_config,
            seed=base_config.tuning_seed,
            hyperparams=hyperparams,
            output_root=TUNING_DIR / "d_b_target_beta" / f"beta_{beta}",
            evaluate_test=False,
        )
        score = result["best_val_macro_f1"]
        history.append({
            "model": "D_B+Target",
            "parameter": "target_loss_beta",
            "candidate": beta,
            "val_macro_f1": score,
            "seed": base_config.tuning_seed,
        })
        if score > best_score:
            best_score = score
            best_beta = beta
    selected = {**base_hyperparams, "target_loss_beta": best_beta}
    return selected, history


# ── 전체 하이퍼파라미터 튜닝 실행 ────────────────
# 모든 모델에 대해 순차 탐색을 돌리고 결과를 저장해요
def run_hyperparameter_tuning(
    config: ExperimentConfig | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """모든 모델의 하이퍼파라미터를 순차 탐색으로 최적화하고 결과를 저장해요."""
    config = config or get_config()

    # 이미 튜닝 결과가 있으면 바로 반환 (시간 절약!)
    if TUNING_SUMMARY_PATH.exists() and not force_refresh:
        print("[tuning] 기존 튜닝 결과를 불러올게요 (다시 하려면 --force를 써주세요)", flush=True)
        return load_tuned_hyperparams()

    # 데이터와 VADER 피처가 준비되어 있는지 확인!
    prepare_data(config)
    extract_vader_features(force_refresh=False)

    # 튜닝할 모델 3개를 정의해요
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
            "BERT+MLP",
            lambda local_config: _transformer_dataset_builder(local_config, "bert-base-uncased"),
            lambda local_config: TransformerMLPClassifier(
                model_name="bert-base-uncased",
                dropout=local_config.dropout,
                hidden_dim=local_config.mlp_hidden,
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

    all_history = []  # 모든 모델의 탐색 기록을 모아요
    tuned_summary: dict[str, dict[str, Any]] = {}

    # 각 모델별로 순차 탐색을 실행!
    for tuning_key, dataset_builder, model_factory in tuning_specs:
        print(f"[tuning] Running sequential search for {tuning_key}", flush=True)
        tuned_values, history = _tune_single_model(tuning_key, config, dataset_builder, model_factory)
        all_history.extend(history)
        tuned_summary[tuning_key] = tuned_values

    if config.v2_enabled:
        alpha_base = tuned_summary.get("BERT+MLP") or tuned_summary.get("BERT-base", {})
        alpha_values, alpha_history = _tune_attention_alpha(config, alpha_base)
        all_history.extend(alpha_history)
        for condition in ["B_B", "D_B", "B_R", "D_R"]:
            tuned_summary[condition] = dict(alpha_values)
        beta_values, beta_history = _tune_target_beta(config, tuned_summary["D_B"])
        all_history.extend(beta_history)
        tuned_summary["D_B+Target"] = dict(beta_values)

    history_frame = pd.DataFrame(all_history)
    save_dataframe(history_frame, TUNING_LOG_PATH)
    save_json(tuned_summary, TUNING_SUMMARY_PATH)
    save_text(
        "# Hyperparameter Tuning Summary\n\n"
        + dataframe_to_markdown(pd.DataFrame([{"model": key, **value} for key, value in tuned_summary.items()])),
        TUNING_DIR / "transformer_tuning_best.md",
    )
    return tuned_summary


# ╔══════════════════════════════════════════════════════════╗
# ║  전체 벤치마크 실행 — 모든 모델을 한 번에!                ║
# ╚══════════════════════════════════════════════════════════╝
# TF-IDF 베이스라인 + 트랜스포머 모델을 모두 학습하고 비교해요.
# 이 함수 하나만 호출하면 전체 실험이 끝나요!
def run_benchmark(config: ExperimentConfig | None = None) -> pd.DataFrame:
    """전체 벤치마크를 실행해요: TF-IDF 베이스라인 + 트랜스포머 모델 모두!"""
    config = config or get_config()
    prepare_data(config)
    extract_vader_features()

    # Step 1: TF-IDF 베이스라인 (LR + SVM)
    tfidf_runs = run_tfidf_baselines(load_splits(), config)
    # Step 2: 트랜스포머 모델들 (BERT, BERT+VADER, RoBERTa+VADER)
    transformer_runs = run_transformer_benchmark(config=config)
    # Step 3: 모든 결과를 합쳐서 보고서용 산출물 저장!
    all_runs = tfidf_runs + transformer_runs
    return save_benchmark_artifacts(all_runs)


# ── 파이프라인 상태 확인 ────────────────────────
# 각 단계가 완료되었는지 한눈에 보여주는 함수예요.
# "지금 어디까지 진행됐지?" 할 때 유용해요!


def _tuning_matches_config(config: ExperimentConfig) -> bool:
    if not TUNING_SUMMARY_PATH.exists():
        return False
    if not config.v2_enabled:
        return True
    tuned_params = load_tuned_hyperparams()
    required_keys = {"BERT+MLP", "B_B", "D_B", "B_R", "D_R", "D_B+Target"}
    return required_keys.issubset(tuned_params.keys())


def _benchmark_matches_config(config: ExperimentConfig) -> bool:
    if not BENCHMARK_RUNS_PATH.exists() or not BENCHMARK_SUMMARY_PATH.exists() or not BEST_MODELS_PATH.exists():
        return False
    if not config.v2_enabled:
        return True
    try:
        run_frame = pd.read_csv(BENCHMARK_RUNS_PATH)
    except Exception:
        return False
    if "condition" not in run_frame.columns:
        return False
    required_conditions = {spec.condition for spec in V2_CONDITION_SPECS}
    observed_conditions = set(run_frame["condition"].dropna().astype(str))
    return required_conditions.issubset(observed_conditions)


def _xai_matches_config(config: ExperimentConfig) -> bool:
    summary_path = XAI_DIR / "xai_summary.json"
    details_path = XAI_DIR / "xai_4axis_metrics.json"
    if not summary_path.exists():
        return False
    if not config.v2_enabled:
        return True
    if not details_path.exists():
        return False
    try:
        summary = load_json(summary_path)
    except Exception:
        return False
    required_keys = {
        "baseline_ci",
        "improved_ci",
        "baseline_mss",
        "improved_mss",
        "baseline_interaction_strength",
        "improved_interaction_strength",
        "baseline_rollout_entropy",
        "improved_rollout_entropy",
    }
    return required_keys.issubset(summary.keys())


def describe_status() -> dict[str, Any]:
    """현재 파이프라인의 진행 상태를 딕셔너리로 반환해요."""
    config = get_config()
    data_ready = SPLITS_PICKLE_PATH.exists()
    if data_ready:
        try:
            splits = pd.read_pickle(SPLITS_PICKLE_PATH)
            data_ready = all(_has_v2_columns(frame) for frame in splits.values()) if config.v2_enabled else True
        except Exception:
            data_ready = False

    vader_ready = _artifact_current(VADER_PICKLE_PATH, [SPLITS_PICKLE_PATH])
    tuning_ready = _tuning_matches_config(config)
    benchmark_ready = _benchmark_matches_config(config)
    freeze_study_ready = FREEZE_STUDY_PATH.exists() and (
        not config.v2_enabled or (benchmark_ready and _artifact_current(FREEZE_STUDY_PATH, [BENCHMARK_SUMMARY_PATH]))
    )
    xai_ready = _xai_matches_config(config)
    dashboard_ready = (OUTPUT_DIR / "dashboard" / "index.html").exists() and (
        not config.v2_enabled or (benchmark_ready and xai_ready)
    )

    return {
        "config_exists": CONFIG_PATH.exists(),          # 설정 파일 존재?
        "data_ready": data_ready,                       # v2 데이터 계약 완료?
        "vader_ready": vader_ready,                     # 현재 data split 기준 VADER 피처?
        "tuning_ready": tuning_ready,                   # v2 alpha/beta 튜닝 완료?
        "benchmark_ready": benchmark_ready,             # v2 8조건 벤치마크 완료?
        "freeze_study_ready": freeze_study_ready,       # v2 benchmark 이후 프리즈 스터디 완료?
        "xai_ready": xai_ready,                         # v2 4축 XAI 완료?
        "dashboard_ready": dashboard_ready,             # v2 결과 기준 대시보드 생성?
    }
