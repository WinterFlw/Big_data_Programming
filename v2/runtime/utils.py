"""
공통 유틸리티 모듈 (utils.py)
==============================

이 파일은 프로젝트 전체에서 공유하는 도구 모음이에요.
여기에 있는 함수들은 다른 모든 파이썬 파일에서 import해서 씁니다.

크게 나누면 이런 역할들을 해요:
  1. 디바이스 감지 — 우리 M3 Max의 MPS를 자동으로 잡아줌
  2. 시드 고정 — 실험을 여러 번 돌려도 같은 결과가 나오게
  3. 성능 지표 계산 — Accuracy, Macro F1, AUROC 등을 한 번에
  4. 파일 I/O — JSON, pickle, CSV 저장/로드를 깔끔하게
  5. 시각화 — 혼동행렬, 학습곡선, 모델 비교 차트 자동 생성
  6. Early Stopping — 검증 성능이 더 안 오르면 학습을 멈추는 장치
"""

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

# ──────────────────────────────────────────────
# matplotlib 백엔드 설정
# "Agg"는 화면 없이 이미지 파일로 저장만 하는 모드예요.
# 서버나 터미널에서 돌릴 때 GUI 없어도 그래프를 그릴 수 있게 해줍니다.
# ──────────────────────────────────────────────
RUNTIME_DIR = Path(__file__).resolve().parent
BASE_DIR = RUNTIME_DIR.parent

MPLCONFIGDIR = BASE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
CACHE_DIR = BASE_DIR / ".cache"
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

# ──────────────────────────────────────────────
# 프로젝트 디렉토리 구조
# 각 폴더가 어떤 역할인지 한눈에 볼 수 있도록 정리했어요.
# 새로운 산출물을 추가할 때는 여기 경로를 참고하면 됩니다.
# ──────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"              # HateXplain 원본 JSON (GitHub에서 자동 다운로드)
OUTPUT_DIR = BASE_DIR / "outputs"          # 모든 실험 결과의 최상위 폴더
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 학습된 모델 가중치 (.pt 파일, 개당 ~420MB)
REPORT_DIR = OUTPUT_DIR / "reports"        # 보고서에 바로 쓸 수 있는 표/그래프
RUNS_DIR = OUTPUT_DIR / "runs"             # seed별 반복 실험 (각각 history, confusion matrix 포함)
TUNING_DIR = OUTPUT_DIR / "tuning"         # 하이퍼파라미터 탐색 기록 (lr, dropout 등)
XAI_DIR = OUTPUT_DIR / "xai"              # SHAP/LIME 분석 결과 (Overlap@5, 케이스 비교 등)

# ──────────────────────────────────────────────
# 라벨 체계
# Davidson et al. (2017)이 제안한 3-class 분류를 따릅니다.
#   0: hatespeech — 특정 집단을 겨냥한 혐오 발화
#   1: offensive  — 거친 표현이지만 특정 집단 대상은 아닌 것
#   2: normal     — 문제 없는 일반 텍스트
# 이 구분이 모호한 게 바로 우리 연구의 핵심 문제!
# ──────────────────────────────────────────────
LABEL_NAMES = ["hatespeech", "offensive", "normal"]
LABEL2ID = {name: index for index, name in enumerate(LABEL_NAMES)}
ID2LABEL = {index: name for name, index in LABEL2ID.items()}
NUM_LABELS = len(LABEL_NAMES)

# VADER 감성 분석이 뱉어주는 4개 점수 (Hutto & Gilbert, 2014)
#   pos: 긍정 비율 (0~1), neg: 부정 비율 (0~1),
#   neu: 중립 비율 (0~1), compound: 종합 점수 (-1~+1)
VADER_COLUMNS = ["pos", "neg", "neu", "compound"]

# 필요한 디렉토리가 없으면 자동으로 만들어줍니다 (처음 실행할 때 편하게!)
for directory in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, REPORT_DIR, RUNS_DIR, TUNING_DIR, XAI_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ======================================================================
#  디바이스 & 재현성 관련
# ======================================================================

def get_device() -> torch.device:
    """
    학습/추론에 쓸 디바이스를 자동으로 골라줍니다.
    우선순위: MPS(Apple Silicon) > CUDA(NVIDIA) > CPU
    우리 M3 Max에서는 항상 MPS가 잡힐 거예요.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    모든 난수 생성기의 시드를 고정해요.
    같은 시드로 돌리면 항상 동일한 결과가 나옵니다 (실험 재현성!).

    고정하는 대상: Python random, NumPy, PyTorch (CPU + GPU + MPS)
    기본 시드 42는 '은하수를 여행하는 히치하이커를 위한 안내서'에서 따온 거예요 :)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 15 seed paired test 재현성 보장을 위해 cudnn 결정성 강제.
        # 약 5~10% 속도 저하가 있을 수 있지만 시드 변동 측정이 본 연구 핵심 메시지라 절대 끄지 않습니다.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # MPS 시드 — PyTorch 버전에 따라 이 함수가 없을 수도 있어서 체크합니다
    if hasattr(torch.mps, "manual_seed") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def clear_device_cache() -> None:
    """
    GPU/MPS 메모리 캐시를 비워줍니다.
    모델을 바꿔가며 실험할 때 이전 모델의 찌꺼기가 남아있으면
    메모리가 부족해질 수 있거든요. 모델 전환 시점마다 호출하면 안전해요.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ======================================================================
#  파일 경로 & I/O 유틸리티
# ======================================================================

def slugify(value: str) -> str:
    """
    텍스트를 파일명에 쓸 수 있는 형태로 바꿔줍니다.
    예: "BERT+VADER (Fine-tuned)" → "bert_vader_fine_tuned"
    특수문자, 공백 다 밑줄로 치환해요.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "artifact"


def ensure_dir(path: Path | str) -> Path:
    """폴더가 없으면 만들어줍니다. 이미 있으면 아무것도 안 해요."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_serializable(obj: Any) -> Any:
    """
    JSON으로 저장할 수 없는 타입들을 변환해주는 내부 헬퍼.
    numpy 배열 → list, Path → str, dataclass → dict 등으로 바꿔줘요.
    save_json()에서 내부적으로 쓰입니다.
    """
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
    """데이터를 JSON 파일로 저장. 한글도 깨지지 않게 ensure_ascii=False로 처리해요."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(_to_serializable(data), handle, ensure_ascii=False, indent=2)
    return target


def load_json(filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Any:
    """JSON 파일 불러오기."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(data: Any, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """
    파이썬 객체를 pickle로 저장.
    데이터 분할(data_splits.pkl)이나 VADER 피처 같은 중간 산출물을 저장할 때 씁니다.
    pickle은 파이썬 전용이지만 속도가 빠르고 어떤 객체든 저장할 수 있어요.
    """
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "wb") as handle:
        pickle.dump(data, handle)
    return target


def load_pickle(filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Any:
    """pickle 파일 불러오기."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    with open(target, "rb") as handle:
        return pickle.load(handle)


def save_dataframe(frame: pd.DataFrame, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """DataFrame을 CSV로 저장. 엑셀이나 구글 시트에서 바로 열 수 있어요."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)
    return target


def save_text(text: str, filename: str | Path, directory: Path | str = OUTPUT_DIR) -> Path:
    """텍스트 파일(마크다운 보고서 등) 저장."""
    target = Path(filename)
    if not target.is_absolute():
        target = Path(directory) / target
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    return target


def remove_tree(path: Path | str) -> None:
    """폴더 전체를 삭제. clean 명령에서 outputs/와 checkpoints/를 날릴 때 씁니다."""
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)


# ======================================================================
#  클래스 가중치 & 성능 지표
# ======================================================================

def compute_class_weight_tensor(labels: Iterable[int], imbalance_threshold: float = 0.10) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """
    클래스 불균형을 감지하고, 필요하면 가중치 텐서를 만들어줍니다.

    HateXplain의 클래스 분포는 이래요:
      hate: 29.5%  |  offensive: 27.2%  |  normal: 38.8%
    비율 차이가 크지 않아서 (소수 클래스 27.2% > threshold 10%)
    실제로는 가중치를 안 쓰게 됩니다.

    만약 극단적으로 불균형한 데이터라면 자동으로 balanced 가중치를 적용해요.
    CrossEntropyLoss(weight=...)에 넘겨주면 됩니다.
    """
    label_array = np.asarray(list(labels))
    counts = np.bincount(label_array, minlength=NUM_LABELS)
    ratios = counts / max(counts.sum(), 1)
    minority_ratio = float(ratios.min()) if len(ratios) else 0.0
    use_weights = minority_ratio < imbalance_threshold

    # 메타데이터는 나중에 JSON으로 저장해서 어떤 판단이 내려졌는지 기록해둡니다
    metadata = {
        "counts": {LABEL_NAMES[index]: int(count) for index, count in enumerate(counts)},
        "ratios": {LABEL_NAMES[index]: float(ratio) for index, ratio in enumerate(ratios)},
        "minority_ratio": minority_ratio,
        "use_class_weight": use_weights,
        "threshold": imbalance_threshold,
    }

    if not use_weights:
        return None, metadata  # 불균형이 심하지 않으면 가중치 없이 진행

    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_LABELS), y=label_array)
    metadata["weights"] = {LABEL_NAMES[index]: float(weight) for index, weight in enumerate(weights)}
    return torch.tensor(weights, dtype=torch.float32), metadata


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    """
    분류 성능 지표를 한 번에 계산해서 딕셔너리로 돌려줍니다.

    계산하는 지표들:
      - accuracy: 전체 정답률
      - macro_f1: 3개 클래스 F1의 단순 평균 (불균형 데이터에서 공정한 지표)
      - macro_precision / macro_recall: 정밀도와 재현율
      - auroc: ROC 곡선 아래 면적 (확률 예측이 있을 때만)
      - per_class_*: 클래스별 precision, recall, F1 (hatespeech F1이 핵심!)
      - confusion_matrix: 혼동행렬 (어떤 클래스끼리 헷갈리는지 한눈에)

    이 함수 하나로 보고서 표를 다 채울 수 있어요 :)
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # 클래스별 지표 — 특히 hatespeech F1이 낮은 게 핵심 문제
    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0,
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0,
    )

    # AUROC — 확률 예측이 있어야 계산 가능 (TF-IDF+SVM은 CalibratedClassifierCV로 확보)
    auroc = None
    if y_prob is not None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            auroc = None  # 클래스가 하나만 있으면 AUROC 계산 불가

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
    """평균 ± 표준편차 형태로 포맷. 보고서 표에 넣을 때 쓰입니다."""
    if mean_value is None:
        return "N/A"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Iterable[Any],
    group_name: str = "source",
) -> pd.DataFrame:
    """그룹 변수(source/target 등)별로 분류 성능을 분리 보고하는 헬퍼.

    명세서 v2.1: stratified split은 라벨 단일이지만, 평가 시 source-별 / target-별
    Subgroup Macro F1, Precision, Recall, Accuracy를 분리해서 보고하여
    간접 leakage 또는 subgroup robustness를 검증합니다.

    Args:
        y_true: 정답 라벨 배열 (int)
        y_pred: 예측 라벨 배열 (int)
        groups: 각 샘플의 그룹 라벨 (예: ["gab", "twitter", ...])
        group_name: 결과 DataFrame의 그룹 컬럼명

    Returns:
        DataFrame[group_name, n_samples, macro_f1, macro_precision, macro_recall, accuracy]
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    groups_arr = np.asarray(list(groups), dtype=object)

    if len(y_true_arr) != len(y_pred_arr) or len(y_true_arr) != len(groups_arr):
        raise ValueError(
            f"length mismatch: y_true={len(y_true_arr)} y_pred={len(y_pred_arr)} groups={len(groups_arr)}"
        )

    rows = []
    unique_groups = sorted({str(group) for group in groups_arr.tolist()})
    for group_value in unique_groups:
        mask = np.array([str(item) == group_value for item in groups_arr], dtype=bool)
        if mask.sum() == 0:
            continue
        sub_true = y_true_arr[mask]
        sub_pred = y_pred_arr[mask]
        rows.append({
            group_name: group_value,
            "n_samples": int(mask.sum()),
            "macro_f1": float(f1_score(sub_true, sub_pred, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(sub_true, sub_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(sub_true, sub_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(sub_true, sub_pred)),
        })
    return pd.DataFrame(rows)


def primary_target_label(targets: list[str]) -> str:
    """multi-label target 리스트에서 대표 카테고리를 뽑는 헬퍼.

    Subgroup 분석 시 단일 그룹 변수로 사용. None만 있으면 'None',
    비-None target이 있으면 첫 번째를 대표로 사용 (학습 supervision 영향 없음).
    """
    non_none = [item for item in (targets or []) if item and str(item) != "None"]
    if not non_none:
        return "None"
    return str(non_none[0])


def aggregate_run_metrics(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    seed별 반복 실험 결과를 모아서 모델별 평균/표준편차 요약표를 만듭니다.
    예: 같은 조건을 여러 seed로 돌린 결과 → 평균 ± std로 한 줄 요약
    """
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

        # 클래스별 F1도 모델별로 평균 내줍니다
        for metric_name in ["per_class_f1", "per_class_precision", "per_class_recall"]:
            for label in LABEL_NAMES:
                values = group[f"{metric_name}.{label}"].to_numpy(dtype=float)
                row[f"{metric_name}.{label}_mean"] = float(values.mean())
                row[f"{metric_name}.{label}_std"] = float(values.std(ddof=0))

        grouped_rows.append(row)

    # Macro F1 기준 내림차순 정렬 — 가장 좋은 모델이 맨 위로!
    return pd.DataFrame(grouped_rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


# ======================================================================
#  Early Stopping
# ======================================================================

class EarlyStopping:
    """
    Early Stopping — 과적합을 막아주는 착한 파수꾼이에요.

    어떻게 동작하나요?
      매 에포크마다 검증 손실(val_loss)을 확인합니다.
      patience 에포크 동안 손실이 안 줄어들면 → "이제 그만 학습하자!" 신호를 보내요.

    왜 필요하나요?
      BERT 같은 큰 모델은 너무 오래 학습하면 훈련 데이터에 과적합되거든요.
      검증 성능이 더 이상 오르지 않는 시점에서 멈추는 게 최적이에요.

    사용법:
      stopper = EarlyStopping(patience=2)
      for epoch in range(max_epochs):
          ... 학습 ...
          if stopper.update(val_loss):  # True가 오면 멈추기!
              break
    """

    def __init__(self, patience: int = 2, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode          # "min": 작을수록 좋음(loss), "max": 클수록 좋음(F1)
        self.min_delta = min_delta  # 이것보다 적게 개선되면 개선으로 안 침
        self.best_value: float | None = None
        self.counter = 0          # 개선 안 된 연속 에포크 수
        self.should_stop = False

    def update(self, value: float) -> bool:
        """새 검증 값을 받아서 멈출지 판단. True가 오면 학습을 중단하세요!"""
        if self.best_value is None:
            self.best_value = value  # 첫 에포크는 무조건 기록
            return False

        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0  # 개선됐으니 카운터 리셋!
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True  # patience만큼 참았는데 안 돼서 포기

        return self.should_stop


# ======================================================================
#  시각화 함수들
#  보고서에 넣을 그래프를 자동으로 만들어줍니다.
#  전부 PNG로 저장되고, 화면에 띄우지 않아요 (Agg 백엔드).
# ======================================================================

def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path, labels: list[str] | None = None) -> None:
    """
    혼동행렬 히트맵을 그려서 저장합니다.
    어떤 클래스를 어떤 클래스로 잘못 예측했는지 한눈에 보여요.
    특히 hate↔offensive 혼동이 많은지 확인하는 데 핵심적인 그래프!
    """
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
    """
    학습곡선을 그려서 저장합니다.
    왼쪽: train_loss vs val_loss (과적합 여부 확인)
    오른쪽: val_macro_f1 (에포크별 성능 추이)

    이상적인 모습:
      - train_loss와 val_loss가 함께 내려가다가
      - val_loss만 올라가기 시작하면 → 그게 early stopping 포인트!
    """
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 왼쪽: 손실 곡선
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="train_loss")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], marker="o", label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # 오른쪽: 검증 F1 곡선
    axes[1].plot(history_frame["epoch"], history_frame["val_macro_f1"], marker="o", color="#dd8452")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Validation Macro F1")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_split_distribution(splits: dict[str, pd.DataFrame], output_path: Path) -> None:
    """
    train/val/test 각 분할의 라벨 분포를 막대그래프로 그립니다.
    stratified split이 잘 됐는지 확인하는 용도예요.
    세 막대의 비율이 비슷해야 정상!
    """
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
    """
    모델들의 성능을 한눈에 비교하는 4칸 차트를 그립니다.
    Macro F1 / Precision / Recall / Accuracy를 에러바(±std)와 함께 표시해요.
    보고서의 '모델 비교' 절에 넣기 딱 좋은 그래프!
    """
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
        # 막대 위에 수치 표시
        for index, value in enumerate(summary_frame[mean_col]):
            axis.text(index, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Comparison Across Repeated Runs")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_per_class_heatmap(summary_frame: pd.DataFrame, output_path: Path) -> None:
    """
    클래스별 F1을 히트맵으로 그립니다.
    행: 모델, 열: hatespeech / offensive / normal

    이 그래프에서 핵심적으로 볼 것:
      - offensive 열이 다른 열보다 확 낮으면 → hate↔offensive 혼동 문제!
      - VADER 추가 후 offensive 셀이 밝아지면 → 우리 개선이 효과 있다는 증거!
    """
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


# ======================================================================
#  보고서 포맷팅 헬퍼
# ======================================================================

def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """
    DataFrame을 마크다운 표로 변환합니다.
    보고서 .md 파일에 바로 복붙할 수 있는 형태로 만들어줘요.
    """
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
    """
    중첩된 딕셔너리를 1단계로 펼쳐줍니다.
    예: {"per_class_f1": {"hatespeech": 0.75}} → {"per_class_f1.hatespeech": 0.75}
    이래야 DataFrame으로 만들 때 컬럼이 깔끔하게 나와요.
    """
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat[f"{key}.{nested_key}"] = nested_value
        else:
            flat[key] = value
    return flat


# ╔══════════════════════════════════════════════════════════╗
# ║  통계적 유의성 검정 — "이 차이가 진짜인가요?"             ║
# ╚══════════════════════════════════════════════════════════╝
# 모델 A가 모델 B보다 성능이 좋다고 해서 그게 '진짜' 차이인지는
# 통계적으로 검증해야 해요. 같은 시드에서의 성능 차이를 비교하는
# paired t-test를 사용합니다!
#
# ⚠️ 참고: seed 수가 적을수록 통계적 검정력(power)이 낮아요.
#    v2 정식 실험은 15 seed를 기준으로 하고, smoke run 결과는
#    통계 결론이 아니라 실행 계약 검증으로만 해석합니다.

def compute_pairwise_significance(
    run_records: list[dict[str, Any]],
    metric_key: str = "macro_f1",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    모든 모델 쌍에 대해 paired t-test를 수행해요.

    같은 시드에서의 성능 차이를 비교하기 때문에 paired(대응) 검정이 맞아요.
    예: seed=42에서의 BERT F1 vs seed=42에서의 BERT+VADER F1

    Returns:
        DataFrame with columns: model_a, model_b, metric, mean_diff,
        t_statistic, p_value, significant, cohens_d
    """
    from itertools import combinations
    from scipy.stats import ttest_rel

    # 모델별로 시드 → 메트릭값 매핑 만들기
    model_seed_scores: dict[str, dict[int, float]] = {}
    for record in run_records:
        model_name = record.get("model", "")
        seed = record.get("seed")
        score = record.get(metric_key)
        if model_name and seed is not None and score is not None:
            model_seed_scores.setdefault(model_name, {})[seed] = float(score)

    model_names = sorted(model_seed_scores.keys())
    rows = []

    for model_a, model_b in combinations(model_names, 2):
        scores_a = model_seed_scores[model_a]
        scores_b = model_seed_scores[model_b]
        # 같은 시드만 비교 (paired test니까!)
        common_seeds = sorted(set(scores_a.keys()) & set(scores_b.keys()))
        if len(common_seeds) < 2:
            continue

        vals_a = [scores_a[s] for s in common_seeds]
        vals_b = [scores_b[s] for s in common_seeds]
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        mean_diff = sum(diffs) / len(diffs)

        # paired t-test
        t_stat, p_value = ttest_rel(vals_a, vals_b)

        # Cohen's d (효과 크기) — 차이가 얼마나 의미있게 큰지 보여줘요
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs) - 1, 1)) ** 0.5
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric_key,
            "n_seeds": len(common_seeds),
            "mean_diff": round(mean_diff, 6),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < alpha,
            "cohens_d": round(cohens_d, 4),
        })

    return pd.DataFrame(rows)


def compute_factorial_anova(frame: pd.DataFrame, metric_key: str, factors: list[str]) -> pd.DataFrame:
    """범주형 factor들의 주효과/상호작용을 OLS nested-model 비교로 검정합니다."""
    from itertools import combinations
    from scipy.stats import f as f_dist

    required = [metric_key, *factors]
    data = frame.dropna(subset=required).copy()
    if data.empty or len(data) <= 2:
        return pd.DataFrame()

    for factor in factors:
        data[factor] = data[factor].astype(str)

    terms: list[tuple[str, ...]] = []
    for order in range(1, len(factors) + 1):
        terms.extend(combinations(factors, order))

    def _term_matrix(term: tuple[str, ...]) -> np.ndarray:
        matrices = []
        for factor in term:
            dummies = pd.get_dummies(data[factor], prefix=factor, drop_first=True, dtype=float)
            if dummies.empty:
                dummies = pd.DataFrame({f"{factor}_const": np.ones(len(data))})
            matrices.append(dummies.to_numpy(dtype=float))
        matrix = matrices[0]
        for next_matrix in matrices[1:]:
            products = [
                (matrix[:, left] * next_matrix[:, right]).reshape(-1, 1)
                for left in range(matrix.shape[1])
                for right in range(next_matrix.shape[1])
            ]
            matrix = np.hstack(products) if products else np.empty((len(data), 0))
        return matrix

    term_matrices = {term: _term_matrix(term) for term in terms}

    def _design(excluded: tuple[str, ...] | None = None) -> np.ndarray:
        columns = [np.ones((len(data), 1))]
        for term, matrix in term_matrices.items():
            if excluded is not None and term == excluded:
                continue
            if matrix.size:
                columns.append(matrix)
        return np.hstack(columns)

    y = data[metric_key].to_numpy(dtype=float)
    full_x = _design()
    _, full_residuals, full_rank, _ = np.linalg.lstsq(full_x, y, rcond=None)
    full_sse = float(full_residuals[0]) if full_residuals.size else float(np.sum((y - full_x @ np.linalg.lstsq(full_x, y, rcond=None)[0]) ** 2))
    full_df = max(len(y) - full_rank, 1)

    rows = []
    for term in terms:
        reduced_x = _design(excluded=term)
        _, reduced_residuals, reduced_rank, _ = np.linalg.lstsq(reduced_x, y, rcond=None)
        reduced_sse = (
            float(reduced_residuals[0])
            if reduced_residuals.size
            else float(np.sum((y - reduced_x @ np.linalg.lstsq(reduced_x, y, rcond=None)[0]) ** 2))
        )
        df_num = max(full_rank - reduced_rank, 1)
        ss_term = max(reduced_sse - full_sse, 0.0)
        ms_term = ss_term / df_num
        ms_error = full_sse / full_df if full_df > 0 else 0.0
        f_value = ms_term / ms_error if ms_error > 0 else np.inf
        p_value = float(f_dist.sf(f_value, df_num, full_df)) if np.isfinite(f_value) else 0.0
        rows.append({
            "term": " * ".join(term),
            "metric": metric_key,
            "df_num": int(df_num),
            "df_den": int(full_df),
            "sum_sq": round(ss_term, 6),
            "f_value": round(float(f_value), 6) if np.isfinite(f_value) else "inf",
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        })

    return pd.DataFrame(rows)
