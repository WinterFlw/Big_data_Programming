"""
# ╔══════════════════════════════════════════════════════════╗
# ║  experiment_core.py — 우리 실험의 핵심 엔진이에요!       ║
# ╚══════════════════════════════════════════════════════════╝
#
# 안녕하세요! 이 파일은 혐오표현 탐지 실험의 모든 핵심 로직을 담고 있어요.
# 데이터 준비부터 모델 학습, 평가까지 전체 파이프라인이 여기에 있답니다.
#
# 이 모듈이 하는 일을 친절하게 정리하면:
#   1. 데이터 준비: HateXplain 데이터셋을 다운로드하고, 3명의 어노테이터가
#      다수결로 라벨을 정해요. 그리고 70/10/20 비율로 나눠요 (train/val/test)
#   2. VADER 감성 피처 추출: 텍스트의 감성 점수 4가지를 뽑아요
#      (긍정 pos, 부정 neg, 중립 neu, 종합 compound)
#   3. 모델 정의 — 세 가지 분류기를 만들어요:
#      - TransformerCLSClassifier: BERT의 [CLS] 토큰으로 바로 분류하는 기본 모델
#      - TransformerMLPClassifier: BERT [CLS] + MLP 헤드 (VADER 없음) — ablation 통제용
#      - HybridSentimentClassifier: BERT [CLS] + VADER 감성 점수를 합쳐서
#        더 똑똑하게 분류하는 우리의 개선 모델이에요!
#   4. 학습 루프: AdamW 옵티마이저 + 워밍업 + 조기 종료로 안정적으로 학습해요
#   5. 벤치마크: TF-IDF(LR/SVM) 전통 모델과 Transformer 모델을 여러 시드로
#      반복 실험해서 신뢰할 수 있는 결과를 얻어요
#   6. 하이퍼파라미터 탐색: lr → batch → dropout → epochs 순서로 최적값을 찾아요
#   7. Freeze Study: 인코더를 얼리면 vs 풀면 성능이 어떻게 달라지는지 비교해요
#
# 화이팅! 코드를 천천히 읽어보면 분명 이해할 수 있을 거예요 :)
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  임포트(import) — 필요한 도구들을 불러오는 구간이에요    ║
# ╚══════════════════════════════════════════════════════════╝

from __future__ import annotations  # 타입 힌트를 깔끔하게 쓸 수 있게 해줘요

# ── Python 기본 라이브러리 ──────────────────────
# JSON 파싱, 파일 경로, 정규식, 시간 측정 등 기본적인 도구들이에요
import json
import os
import re
import time
from collections import Counter          # 투표 결과를 셀 때 아주 유용해요!
from dataclasses import asdict, dataclass, field  # 설정값을 깔끔하게 관리해요
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlretrieve    # 데이터셋을 인터넷에서 다운로드할 때 사용해요

# ── 데이터 과학 & 딥러닝 핵심 라이브러리 ──────────
import numpy as np                        # 수치 계산의 기본! 배열 연산을 빠르게 해줘요
import pandas as pd                       # 데이터프레임으로 표 형태 데이터를 다뤄요
import torch                              # PyTorch — 딥러닝의 핵심 엔진이에요
import torch.nn as nn                     # 신경망 레이어들이 여기에 있어요

# ── sklearn: 전통 머신러닝 도구들 ──────────────────
# TF-IDF + LR/SVM 베이스라인 모델을 만들 때 사용해요
from sklearn.calibration import CalibratedClassifierCV  # SVM에 확률 출력을 추가해줘요
from sklearn.feature_extraction.text import TfidfVectorizer  # 텍스트 → 숫자 벡터 변환!
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split   # 데이터를 train/val/test로 나눠요
from sklearn.svm import LinearSVC

# ── PyTorch 데이터 로딩 & HuggingFace Transformers ──
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

# ── VADER 감성 분석기 ────────────────────────────
# 규칙 기반 감성 분석 도구예요. BERT가 놓칠 수 있는 감성 뉘앙스를 보완해줘요!
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── 우리 프로젝트의 유틸리티 함수들 ──────────────
# utils.py에서 경로, 상수, 헬퍼 함수들을 가져와요
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
    compute_pairwise_significance,
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


# ╔══════════════════════════════════════════════════════════╗
# ║  실험 설정 (ExperimentConfig) — 실험의 모든 설정을 여기서! ║
# ╚══════════════════════════════════════════════════════════╝
# 이 데이터클래스 하나로 실험의 모든 하이퍼파라미터를 관리해요.
# 마치 요리 레시피처럼, 여기 값만 바꾸면 실험 전체가 달라진답니다!
@dataclass
class ExperimentConfig:
    """
    실험 전체에서 공유되는 설정값이에요.
    run.sh 또는 코드에서 자유롭게 변경할 수 있고,
    outputs/experiment_config.json에 자동으로 저장되니까 재현성도 걱정 없어요!
    """

    # ── 데이터 분할 비율 (합계 = 1.0이어야 해요!) ──
    # 70%는 학습, 10%는 검증(하이퍼파라미터 조정용), 20%는 최종 테스트용이에요
    split_train: float = 0.70     # 학습 데이터 — 모델이 열심히 배우는 부분!
    split_val: float = 0.10       # 검증 데이터 — 학습 중간중간 실력 체크용
    split_test: float = 0.20      # 테스트 데이터 — 최종 성적표를 매기는 부분

    # ── 학습 하이퍼파라미터 — 모델 학습의 핵심 레시피 ──
    max_len: int = 128            # 토큰 최대 길이 (HateXplain은 대부분 128이면 충분해요)
    batch_size: int = 64          # 한 번에 처리하는 샘플 수 (MPS에서 64면 ~6GB로 넉넉해요)
    epochs: int = 5               # 최대 에포크 수 (early stopping이 알아서 멈춰줘요)
    learning_rate: float = 2e-5   # 학습률 — BERT fine-tuning의 황금 비율이에요
    warmup_ratio: float = 0.10    # 처음 10% 스텝은 천천히 워밍업해요 (급하면 발산해요!)
    weight_decay: float = 0.01    # L2 정규화 — 과적합을 살짝 막아줘요
    dropout: float = 0.10         # 드롭아웃 — 뉴런을 랜덤하게 쉬게 해서 일반화 능력 UP
    mlp_hidden: int = 256         # 하이브리드 모델 MLP 은닉층 크기

    # ── Early Stopping — 더 나아지지 않으면 멈추는 현명한 전략 ──
    early_stopping_patience: int = 2      # 2 에포크 연속 개선 없으면 "그만!" 해요
    early_stopping_min_delta: float = 1e-4  # 이 정도는 되어야 "개선됐다"고 인정해요

    # ── 클래스 가중치 — 불균형한 데이터를 공정하게! ──
    imbalance_threshold: float = 0.10  # 소수 클래스가 10% 미만이면 가중치를 줘요

    # ── 반복 실험 시드 — 3번 돌려서 평균 ± 표준편차로 신뢰도를 높여요 ──
    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])

    # ── 하이퍼파라미터 탐색 후보 — 어떤 값이 최적일까? ──
    tune_learning_rates: list[float] = field(default_factory=lambda: [1e-5, 2e-5, 3e-5])
    tune_batch_sizes: list[int] = field(default_factory=lambda: [64])
    tune_dropouts: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    tune_epochs: list[int] = field(default_factory=lambda: [5])
    tuning_seed: int = 42         # 튜닝할 때 쓰는 고정 시드 (재현성!)
    tuning_max_epochs: int = 5    # 튜닝 시 최대 에포크

    # ── XAI(설명가능 AI) 분석 설정 — 모델의 판단 이유를 들여다봐요 ──
    xai_sample_size: int = 24     # SHAP/LIME 분석할 샘플 수 (너무 많으면 느려요)
    lime_num_features: int = 5    # LIME이 보여줄 중요 단어 Top-K개
    lime_num_samples: int = 500   # LIME이 텍스트를 얼마나 변형해볼지
    shap_max_evals: int = 300     # SHAP 최대 평가 횟수
    shap_batch_size: int = 32     # SHAP 배치 크기


# ── 기본 설정 인스턴스 ──────────────────────────
# 아무것도 커스터마이즈하지 않을 때 쓰는 기본값이에요
DEFAULT_CONFIG = ExperimentConfig()

# ╔══════════════════════════════════════════════════════════╗
# ║  경로 상수들 — 파일이 어디에 저장되는지 한눈에 보여요     ║
# ╚══════════════════════════════════════════════════════════╝
# 모든 입출력 파일의 경로를 여기서 한 번에 관리해요.
# 나중에 경로를 바꾸고 싶으면 여기만 수정하면 돼요!
RAW_DATASET_PATH = DATA_DIR / "dataset.json"             # 원본 HateXplain 데이터
RAW_SPLIT_PATH = DATA_DIR / "post_id_divisions.json"     # 원본 split 정보
SPLITS_PICKLE_PATH = OUTPUT_DIR / "data_splits.pkl"      # 전처리된 train/val/test 분할
VADER_PICKLE_PATH = OUTPUT_DIR / "vader_features.pkl"    # VADER 감성 피처 캐시
CONFIG_PATH = OUTPUT_DIR / "experiment_config.json"      # 실험 설정 저장 경로
BEST_MODELS_PATH = REPORT_DIR / "best_models.json"       # 각 모델의 베스트 체크포인트 정보
BENCHMARK_RUNS_PATH = REPORT_DIR / "benchmark_runs.csv"  # 벤치마크 전체 실행 기록
BENCHMARK_SUMMARY_PATH = REPORT_DIR / "benchmark_summary.csv"  # 벤치마크 요약 통계
BENCHMARK_MARKDOWN_PATH = REPORT_DIR / "benchmark_summary.md"  # 보고서용 마크다운 표
FREEZE_STUDY_PATH = REPORT_DIR / "freeze_study.csv"      # 프리즈 스터디 결과
FREEZE_STUDY_MARKDOWN_PATH = REPORT_DIR / "freeze_study.md"
TUNING_LOG_PATH = TUNING_DIR / "transformer_tuning_log.csv"    # 하이퍼파라미터 탐색 로그
TUNING_SUMMARY_PATH = TUNING_DIR / "transformer_tuning_best.json"  # 최적 하이퍼파라미터
DATA_PROFILE_PATH = REPORT_DIR / "data_profile.json"     # 데이터셋 프로필 요약


# ── 설정 로드 & 저장 ───────────────────────────
# 실험 설정을 디스크에 저장하고 불러오는 함수들이에요.
# 한 번 저장해두면 나중에 동일한 실험을 재현할 수 있어서 아주 중요해요!
def get_config() -> ExperimentConfig:
    """저장된 설정이 있으면 불러오고, 없으면 기본값을 만들어서 저장해요."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return ExperimentConfig(**data)
    # 처음 실행이라면 기본 설정을 저장해둘게요
    save_json(asdict(DEFAULT_CONFIG), CONFIG_PATH)
    return DEFAULT_CONFIG


def save_config(config: ExperimentConfig) -> None:
    """현재 실험 설정을 JSON 파일로 저장해요. 재현성의 첫걸음!"""
    save_json(asdict(config), CONFIG_PATH)


# ── 데이터 다운로드 ─────────────────────────────
# HateXplain 데이터셋이 없으면 GitHub에서 자동으로 받아와요.
# 이미 있으면 건너뛰니까 걱정 마세요!
def ensure_raw_hatexplain(force_download: bool = False) -> None:
    """HateXplain 원본 데이터가 없으면 GitHub에서 다운로드해요."""
    base_url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url_name, destination in [
        ("dataset.json", RAW_DATASET_PATH),           # 본문 + 어노테이션 전체
        ("post_id_divisions.json", RAW_SPLIT_PATH),   # 원본 train/val/test ID
    ]:
        if force_download or not destination.exists():
            print(f"Downloading {url_name}...")
            urlretrieve(base_url + url_name, destination)


# ── 다수결 투표 로직 ────────────────────────────
# HateXplain은 각 게시물에 3명의 어노테이터가 라벨을 달아요.
# 의견이 갈릴 수 있으니까, 2명 이상 같은 의견이면 그걸 정답으로 삼아요.
# 민주주의처럼 다수결이에요! 3명 다 다른 의견이면 제외한답니다.
def _majority_label(sample: dict[str, Any]) -> int | None:
    """
    3명 어노테이터의 다수결 투표로 라벨을 결정해요.
    2표 이상 일치하지 않으면 None을 반환해서 분석에서 제외해요.
    (HateXplain에서 약 919건이 이렇게 제외돼요)
    """
    # 각 어노테이터의 라벨을 모아서...
    annotator_labels = [annotator["label"] for annotator in sample["annotators"]]
    # Counter로 투표 결과를 세요!
    label_counts = Counter(annotator_labels)
    majority_label, majority_count = label_counts.most_common(1)[0]
    if majority_count < 2:
        return None  # 의견이 갈려서 결정 불가 (undecided)
    return LABEL2ID.get(majority_label)  # 문자열 라벨 → 숫자 ID로 변환


# ── 대상 커뮤니티 수집 ──────────────────────────
# 혐오표현이 누구를 향한 것인지 수집해요 (예: African, Islam, Jewish 등)
def _collect_targets(sample: dict[str, Any]) -> list[str]:
    """어노테이터들이 표시한 대상 커뮤니티를 모아서 중복 제거 후 반환해요."""
    targets: list[str] = []
    for annotator in sample["annotators"]:
        if "target" in annotator:
            targets.extend(annotator["target"])
    return sorted(set(targets)) if targets else ["None"]


# ╔══════════════════════════════════════════════════════════╗
# ║  데이터 전처리 — 원본 JSON을 깔끔한 DataFrame으로!       ║
# ╚══════════════════════════════════════════════════════════╝
def load_processed_dataframe(force_download: bool = False) -> pd.DataFrame:
    """HateXplain 원본 JSON을 다운로드하고 전처리해서 DataFrame으로 만들어요."""
    ensure_raw_hatexplain(force_download=force_download)

    # Step 1: 원본 JSON 파일을 읽어요
    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as handle:
        raw_dataset = json.load(handle)

    # Step 2: 각 게시물을 하나씩 처리해요
    records = []
    for post_id, sample in raw_dataset.items():
        # 다수결 투표로 라벨을 정해요
        label = _majority_label(sample)
        if label is None:
            continue  # 투표 결과가 애매하면 건너뛰어요
        # 토큰들을 합쳐서 원본 텍스트를 복원하고, @멘션은 익명화해요
        text = " ".join(sample["post_tokens"])
        text = re.sub(r"@\S+", "<user>", text)  # 개인정보 보호!
        records.append(
            {
                "post_id": post_id,
                "text": text,
                "label": label,              # 숫자 라벨 (0, 1, 2)
                "label_name": ID2LABEL[label],  # 사람이 읽을 수 있는 라벨명
                "targets": _collect_targets(sample),  # 대상 커뮤니티
            }
        )

    # Step 3: DataFrame으로 변환하고 중복 텍스트 제거!
    frame = pd.DataFrame(records)
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return frame


# ╔══════════════════════════════════════════════════════════╗
# ║  데이터 준비 (prepare_data) — 실험의 첫 번째 단계!       ║
# ╚══════════════════════════════════════════════════════════╝
# 전체 데이터를 train/val/test로 나누고, 분포 통계도 저장해요.
# stratified split을 써서 각 분할에 라벨 비율이 골고루 들어가게 해요!
def prepare_data(config: ExperimentConfig | None = None, force_refresh: bool = False, force_download: bool = False) -> dict[str, pd.DataFrame]:
    """70/10/20 stratified split을 만들고 데이터셋 진단 결과를 저장해요."""
    config = config or get_config()
    save_config(config)

    # 이미 분할된 데이터가 있으면 바로 불러와요 (시간 절약!)
    if SPLITS_PICKLE_PATH.exists() and not force_refresh:
        return pd.read_pickle(SPLITS_PICKLE_PATH)

    frame = load_processed_dataframe(force_download=force_download)

    # Step 1: 먼저 test 20%를 떼어놓고...
    train_val_df, test_df = train_test_split(
        frame,
        test_size=config.split_test,
        random_state=config.tuning_seed,
        stratify=frame["label"],  # 라벨 비율을 유지하면서 나눠요!
    )
    # Step 2: 나머지에서 val을 분리해요 (상대적 비율로 계산)
    val_relative_size = config.split_val / (config.split_train + config.split_val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=config.tuning_seed,
        stratify=train_val_df["label"],
    )

    # Step 3: 깔끔하게 딕셔너리로 정리!
    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }

    # Step 4: 데이터셋 프로필(통계 요약)을 만들어요
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

    # Step 5: 결과물을 저장해요 — pickle, JSON, 시각화까지!
    pd.to_pickle(splits, SPLITS_PICKLE_PATH)
    save_json(profile, DATA_PROFILE_PATH)
    plot_split_distribution(splits, REPORT_DIR / "split_distribution.png")

    # Step 6: 보고서용 CSV와 마크다운 표도 만들어요
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


# ── 분할 데이터 로드 ────────────────────────────
def load_splits() -> dict[str, pd.DataFrame]:
    """전처리된 train/val/test 분할을 불러와요. 없으면 자동으로 만들어줘요!"""
    if not SPLITS_PICKLE_PATH.exists():
        return prepare_data()
    return pd.read_pickle(SPLITS_PICKLE_PATH)


# ╔══════════════════════════════════════════════════════════╗
# ║  VADER 감성 피처 추출 — 텍스트의 감정 온도를 재요!        ║
# ╚══════════════════════════════════════════════════════════╝
# VADER(Valence Aware Dictionary and sEntiment Reasoner)는
# 규칙 기반 감성 분석 도구예요. 각 텍스트에 대해 4가지 점수를 줘요:
#   - pos: 긍정 점수 (0~1)
#   - neg: 부정 점수 (0~1) ← 혐오표현은 이게 높겠죠?
#   - neu: 중립 점수 (0~1)
#   - compound: 종합 점수 (-1~+1) ← 가장 직관적인 지표!
def extract_vader_features(
    splits: dict[str, pd.DataFrame] | None = None,
    force_refresh: bool = False,
) -> dict[str, np.ndarray]:
    """모든 분할(train/val/test)에 대해 VADER 감성 점수를 계산해요."""
    splits = splits or load_splits()
    # 이미 계산해둔 게 있으면 재사용해요 (VADER는 결정적이라 결과가 항상 같아요)
    if VADER_PICKLE_PATH.exists() and not force_refresh:
        return pd.read_pickle(VADER_PICKLE_PATH)

    analyzer = SentimentIntensityAnalyzer()
    features: dict[str, np.ndarray] = {}

    for split_name, frame in splits.items():
        rows = []
        for text in frame["text"].tolist():
            # 각 텍스트의 감성 점수 4개를 추출해요
            scores = analyzer.polarity_scores(text)
            rows.append([scores[column] for column in VADER_COLUMNS])
        features[split_name] = np.asarray(rows, dtype=np.float32)

        # 분석 결과를 CSV로도 저장해요 (보고서에 활용 가능!)
        save_dataframe(
            pd.DataFrame(features[split_name], columns=VADER_COLUMNS),
            REPORT_DIR / f"vader_{split_name}.csv",
        )

    pd.to_pickle(features, VADER_PICKLE_PATH)
    return features


# ╔══════════════════════════════════════════════════════════╗
# ║  PyTorch Dataset 클래스들 — 데이터를 모델에 먹여줘요!     ║
# ╚══════════════════════════════════════════════════════════╝

# ── 트랜스포머 전용 데이터셋 ────────────────────
# BERT/RoBERTa에 넣을 수 있도록 텍스트를 토큰 ID로 변환해요.
# DataLoader가 이 클래스에서 배치 단위로 데이터를 꺼내갑니다!
class TransformerTextDataset(Dataset):
    """트랜스포머 전용 데이터셋. 텍스트 -> input_ids + attention_mask + labels"""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_len: int) -> None:
        # 토크나이저가 텍스트를 숫자 시퀀스로 바꿔줘요
        self.encodings = tokenizer(
            texts,
            truncation=True,    # max_len보다 긴 텍스트는 잘라요
            max_length=max_len,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],       # 토큰 ID들
            "attention_mask": self.encodings["attention_mask"][index],  # 실제 토큰=1, 패딩=0
            "labels": self.labels[index],                           # 정답 라벨
        }


# ── 하이브리드 데이터셋 (트랜스포머 + VADER) ────
# 기본 데이터셋에 VADER 감성 점수 4개를 추가로 포함해요.
# 이 추가 피처가 모델의 판단력을 높여주는 비밀 무기예요!
class HybridTextDataset(Dataset):
    """트랜스포머 + VADER 하이브리드용 데이터셋. VADER 4차원 감성 피처를 함께 전달해요."""

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        vader_features: np.ndarray,  # shape: (N, 4) — pos, neg, neu, compound
        tokenizer,
        max_len: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.vader = torch.tensor(vader_features, dtype=torch.float32)  # VADER 피처도 텐서로!

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
            "vader": self.vader[index],  # 여기가 기본 데이터셋과 다른 점!
        }


# ╔══════════════════════════════════════════════════════════╗
# ║  모델 정의 — 우리 실험의 두뇌를 만드는 구간이에요!        ║
# ╚══════════════════════════════════════════════════════════╝

# ── Baseline 모델: TransformerCLSClassifier ─────
# BERT의 [CLS] 토큰 벡터(768차원)를 바로 선형 분류기에 넣는 가장 기본적인 구조예요.
# [CLS]는 BERT가 문장 전체를 요약한 특별한 토큰이에요!
class TransformerCLSClassifier(nn.Module):
    """
    Baseline 분류기: Transformer [CLS] -> Dropout -> Linear -> 3-class

    구조를 그림으로 보면:
      입력 텍스트 -> BERT encoder -> [CLS] 토큰 (768차원)
                                      -> Dropout (과적합 방지)
                                      -> Linear(768, 3) -> 3개 클래스 확률

    freeze_encoder=True로 하면 BERT 가중치를 얼려서 분류 헤드만 학습해요.
    (Freeze Study에서 사용해요)
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        # HuggingFace에서 사전학습된 모델을 불러와요
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            # 인코더의 모든 파라미터를 동결! 역전파 시 업데이트 안 해요
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size  # BERT-base는 768
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # 768 -> 3

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # BERT에 토큰을 넣어서 출력을 받아요
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output이 있으면 쓰고, 없으면 첫 번째 토큰([CLS]) 직접 가져와요
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)  # logits 반환 (softmax 전!)


# ── 개선 모델: HybridSentimentClassifier ────────
# 이게 우리 논문의 핵심 모델이에요!
# BERT의 [CLS] 벡터에 VADER 감성 점수를 합쳐서 MLP로 분류해요.
# BERT가 문맥을 이해하고, VADER가 감성의 강도를 보완하는 협업 구조!
class HybridSentimentClassifier(nn.Module):
    """
    우리의 개선 모델: Transformer [CLS] + VADER 감성 피처를 결합해요!

    구조를 그림으로 보면:
      Transformer encoder -> [CLS] (768차원)  ─┐
      VADER(pos,neg,neu,compound) (4차원)      ─┤-> concat (772차원)
                                                -> Dropout
                                                -> Linear(772, 256) -> ReLU
                                                -> Linear(256, 3) -> 3개 클래스!

    핵심 아이디어: BERT가 놓칠 수 있는 감성적 뉘앙스를
    VADER의 명시적 감성 점수로 보완하는 거예요.
    예를 들어, hate speech는 neg 점수가 높고 compound가 낮고,
    offensive는 중간 수준의 부정성을 보여요.
    """

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
        hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(dropout)
        # 768(BERT) + 4(VADER) = 772차원을 256차원으로 압축하는 은닉층
        self.hidden = nn.Linear(hidden_size + len(VADER_COLUMNS), hidden_dim)
        self.relu = nn.ReLU()  # 비선형 활성화 — MLP의 표현력을 높여줘요
        self.out = nn.Linear(hidden_dim, num_labels)  # 256 -> 3 (최종 분류)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, vader: torch.Tensor) -> torch.Tensor:
        # BERT 인코더에 텍스트를 넣어요
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        # 여기가 핵심! [CLS] 벡터와 VADER 피처를 이어붙여요 (concat)
        combined = torch.cat([pooled_output, vader], dim=1)  # [batch, 772]
        combined = self.dropout(combined)
        combined = self.hidden(combined)   # [batch, 256]
        combined = self.relu(combined)     # 비선형 변환
        return self.out(combined)          # [batch, 3] logits


# ╔══════════════════════════════════════════════════════════╗
# ║  Ablation 모델 — MLP 용량 효과를 분리해요!               ║
# ╚══════════════════════════════════════════════════════════╝
# BERT+VADER가 좋아진 게 VADER 4차원 덕분인지, 아니면 단순히 MLP가 커서인지
# 알아보기 위한 대조 실험 모델이에요!
#
# 구조: [CLS](768d) → Dropout → Linear(768, 256) → ReLU → Linear(256, 3)
#
# HybridSentimentClassifier와 동일한 MLP 구조이지만 VADER 입력이 없어요.
# 이 모델이 BERT+VADER와 비슷한 성능을 낸다면 → MLP 크기 효과
# 이 모델보다 BERT+VADER가 확실히 낫다면 → VADER의 실질적 기여 입증!

class TransformerMLPClassifier(nn.Module):
    """
    Ablation 분류기: Transformer [CLS] → MLP → 3-class (VADER 없이).

    BERT+VADER와 동일한 MLP 용량을 가지되 VADER 입력을 제거하여,
    성능 향상이 MLP 크기 때문인지 VADER 피처 때문인지 분리합니다.

    구조:
      [CLS](768d) → Dropout → Linear(768, hidden_dim) → ReLU → Linear(hidden_dim, 3)
    """

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
        # VADER 없이 768d에서 바로 MLP로! (Hybrid는 772d → 256)
        self.hidden = nn.Linear(hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = self.relu(pooled_output)
        return self.out(pooled_output)


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
    """트랜스포머 + VADER 하이브리드 모델용 데이터셋을 만들어요. VADER 피처가 추가돼요!"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    datasets = {
        name: HybridTextDataset(
            df["text"].tolist(),
            df["label"].to_numpy(),
            vader_features[name],  # 여기서 VADER 감성 피처를 같이 넘겨요
            tokenizer,
            config.max_len,
        )
        for name, df in splits.items()
    }
    return tokenizer, datasets


# ── 배치 순전파 헬퍼 ────────────────────────────
# baseline과 hybrid 모델 모두 호환되게 forward를 호출해요
def _forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """배치 데이터를 GPU/MPS로 보내고 모델에 통과시켜요. VADER 유무를 자동 감지!"""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if "vader" in batch:
        # 하이브리드 모델: VADER 피처도 같이 넘겨요
        vader = batch["vader"].to(device)
        return model(input_ids=input_ids, attention_mask=attention_mask, vader=vader)
    # 기본 트랜스포머 모델
    return model(input_ids=input_ids, attention_mask=attention_mask)


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
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)  # 가중 교차 엔트로피 손실

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
        for batch in loaders["train"]:
            batch_labels = batch["labels"].to(device)
            # 순전파: 모델에 데이터를 넣어 예측값을 얻어요
            logits = _forward_batch(model, batch, device)
            # 손실 계산: 예측과 정답의 차이를 수치화해요
            loss = criterion(logits, batch_labels)

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
            x_test = vectorizer.transform(test_texts)

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


# ── 전체 하이퍼파라미터 튜닝 실행 ────────────────
# 모든 모델에 대해 순차 탐색을 돌리고 결과를 저장해요
def run_hyperparameter_tuning(config: ExperimentConfig | None = None) -> dict[str, dict[str, Any]]:
    """모든 모델의 하이퍼파라미터를 순차 탐색으로 최적화하고 결과를 저장해요."""
    config = config or get_config()
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
def describe_status() -> dict[str, Any]:
    """현재 파이프라인의 진행 상태를 딕셔너리로 반환해요."""
    return {
        "config_exists": CONFIG_PATH.exists(),          # 설정 파일 존재?
        "data_ready": SPLITS_PICKLE_PATH.exists(),      # 데이터 분할 완료?
        "vader_ready": VADER_PICKLE_PATH.exists(),      # VADER 피처 추출 완료?
        "tuning_ready": TUNING_LOG_PATH.exists(),       # 하이퍼파라미터 튜닝 완료?
        "benchmark_ready": BENCHMARK_SUMMARY_PATH.exists(),  # 벤치마크 완료?
        "freeze_study_ready": FREEZE_STUDY_PATH.exists(),    # 프리즈 스터디 완료?
        "xai_ready": (XAI_DIR / "xai_summary.json").exists(),  # XAI 분석 완료?
        "dashboard_ready": (OUTPUT_DIR / "dashboard" / "index.html").exists(),  # 대시보드 생성 완료?
    }
