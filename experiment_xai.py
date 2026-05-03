# ╔══════════════════════════════════════════════════════════╗
# ║  experiment_xai.py                                      ║
# ║  XAI (설명 가능한 AI) 분석 파이프라인                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
XAI (설명 가능한 AI) 분석 파이프라인.

안녕하세요! 이 파일은 우리 혐오표현 탐지 연구에서 가장 흥미로운 부분이에요.
단순히 "모델이 맞았다/틀렸다"를 넘어서, "왜 그렇게 판단했는지"를 들여다보는 거죠.
XAI(eXplainable AI)를 통해 모델의 판단 근거를 시각적으로 확인할 수 있답니다!

연구의 핵심 흐름 (v2.1 사후 검증):
  기준 조건 분석: A_B(BERT + MLP) 또는 v1 fallback 모델에 SHAP/LIME 적용
  비교 조건 분석: D_B/D_R 또는 v1 VADER fallback 모델에 동일 분석 적용
  목적: 사전에 정의한 ablation 조건 간 설명 패턴 차이를 사후 정량화

  참고: XAI는 사전에 세운 가설(VADER가 감성 맥락을 보완)의 "사후 검증" 역할이에요.
  XAI 결과를 모델 설계에 되먹임하지 않고, 가설 → 통제 실험 → 사후 해석 순서로만 사용합니다.

분석 산출물 (이 파이프라인이 만들어내는 결과물들):
  - Overlap@5: SHAP Top-5 와 LIME Top-5의 일치도 (>=60%이면 높은 신뢰!)
    --> 두 XAI 기법이 같은 토큰을 중요하다고 보면 신뢰할 수 있겠죠?
  - 케이스 비교: 기준 조건과 비교 조건의 SHAP attribution 차이
    --> 각 ablation 조건이 어떤 토큰에 주목했는지 해석 근거를 줍니다
  - xai_summary.md: 보고서에 바로 복붙 가능한 요약 마크다운

참고: SHAP는 CPU에서 실행됩니다 (Apple MPS에서 DeepExplainer가 호환되지 않아요).
     시간이 좀 걸릴 수 있지만, 정확한 분석을 위해 조금만 기다려주세요!
"""

# ╔══════════════════════════════════════════════════════════╗
# ║  1. 라이브러리 임포트 및 환경 설정                         ║
# ╚══════════════════════════════════════════════════════════╝

# ── Python 기본 라이브러리 ──────────────────────────────────
# from __future__는 Python 버전 호환성을 위한 것이에요.
# 예를 들어 `list[str]` 같은 타입 힌트를 Python 3.9 이하에서도 쓸 수 있게 해줍니다.
from __future__ import annotations

from dataclasses import dataclass  # 데이터 클래스: 깔끔한 구조체를 만들 때 유용해요
import os
from pathlib import Path           # 파일 경로를 OS에 상관없이 다루는 멋진 도구!
from typing import Any

# ── 환경 변수 설정 (matplotlib 캐시 등) ───────────────────
# matplotlib이 폰트 캐시 같은 걸 저장할 때 프로젝트 안에 저장하도록 설정해요.
# 이렇게 하면 다른 환경에서도 깔끔하게 동작합니다.
MODULE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = MODULE_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
CACHE_DIR = MODULE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

# ── 시각화 라이브러리 ──────────────────────────────────────
# "Agg" 백엔드는 GUI 없이 파일로만 그래프를 저장할 때 사용해요.
# 서버 환경이나 Jupyter 없이 돌릴 때 필수!
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt    # 그래프 그리기의 국민 라이브러리

# ── 데이터 / ML 핵심 라이브러리 ────────────────────────────
import numpy as np                 # 수치 연산의 기본! 배열 처리에 필수
import pandas as pd                # 데이터프레임으로 테이블 형태 데이터를 다루는 라이브러리
import shap                        # SHAP: Shapley value 기반 XAI 라이브러리 (Lundberg & Lee, 2017)
import torch                       # PyTorch: 딥러닝 프레임워크

# ── XAI / NLP 관련 라이브러리 ──────────────────────────────
from lime.lime_text import LimeTextExplainer  # LIME: 로컬 해석 가능한 모델 (Ribeiro et al., 2016)
from transformers import AutoTokenizer         # HuggingFace 토크나이저 (BERT, RoBERTa 등)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # VADER 감성 분석기

# ── 프로젝트 내부 모듈 (experiment_core) ───────────────────
# 우리가 직접 만든 모듈에서 필요한 것들을 가져옵니다.
# experiment_core.py에는 모델 정의와 실험 설정이 들어있어요.
from experiment_core import (
    BENCHMARK_SUMMARY_PATH,        # 벤치마크 요약 CSV 경로
    BEST_MODELS_PATH,              # 최고 성능 모델 레지스트리 JSON 경로
    ExperimentConfig,              # 실험 설정을 담는 dataclass
    HybridSentimentClassifier,     # BERT/RoBERTa + VADER 하이브리드 모델
    RAW_DATASET_PATH,              # dataset.json 경로 (human rationale 포함)
    SPLITS_PICKLE_PATH,            # train/val/test 분할 데이터 경로
    TransformerConditionClassifier,
    TransformerCLSClassifier,      # 순수 Transformer 분류기 (baseline)
    get_config,                    # 실험 설정 로드 함수
)

# ── 프로젝트 내부 모듈 (utils) ─────────────────────────────
# 유틸리티 함수들이에요. 저장, 시각화, 시드 설정 등 공통 기능을 모아둔 곳!
from utils import (
    LABEL_NAMES,                   # ["hate", "offensive", "normal"] 라벨 이름
    NUM_LABELS,                    # 3 (클래스 수)
    VADER_COLUMNS,                 # VADER 감성 점수 컬럼명 (neg, neu, pos, compound)
    XAI_DIR,                       # XAI 결과물 저장 디렉토리
    clear_device_cache,            # GPU/MPS 메모리 정리
    compute_metrics,               # 정확도, F1 등 성능 지표 계산
    dataframe_to_markdown,         # DataFrame을 마크다운 표로 변환
    ensure_dir,                    # 디렉토리가 없으면 생성
    load_json,                     # JSON 파일 로드
    load_pickle,                   # Pickle 파일 로드
    plot_confusion_matrix,         # 혼동 행렬 시각화
    save_dataframe,                # DataFrame 저장 (CSV)
    save_json,                     # JSON 저장
    save_text,                     # 텍스트 파일 저장
    set_seed,                      # 재현성을 위한 랜덤 시드 설정
    slugify,                       # 문자열을 파일명에 안전한 형태로 변환
)


# ╔══════════════════════════════════════════════════════════╗
# ║  2. 데이터 구조 및 상수 정의                              ║
# ╚══════════════════════════════════════════════════════════╝

@dataclass
class LoadedModelBundle:
    """
    모델 번들 (Model Bundle) - XAI 분석에 필요한 모든 것을 한데 모은 꾸러미!

    하나의 모델을 XAI로 분석하려면 모델 자체뿐 아니라 토크나이저, 디바이스 정보 등이
    함께 필요하거든요. 이걸 매번 따로 넘기면 코드가 복잡해지니까,
    dataclass로 깔끔하게 묶어서 들고 다닙니다.
    마치 여행 갈 때 옷, 세면도구, 충전기를 캐리어 하나에 넣는 것처럼요!
    """

    display_name: str          # 화면에 표시할 이름 (예: "BERT-base", "RoBERTa+VADER")
    model_type: str            # "transformer" 또는 "hybrid" (VADER 사용 여부 구분)
    model_name: str            # HuggingFace 모델 이름 (예: "bert-base-uncased")
    model: torch.nn.Module     # 학습된 PyTorch 모델 객체
    tokenizer: Any             # 텍스트를 토큰으로 쪼개는 토크나이저
    device: torch.device       # 연산 장치 (XAI에서는 항상 CPU)


# v2 결과가 있으면 A_B vs D_B/D_R를 우선 사용하고, 없으면 v1 checkpoint로 fallback합니다.
BASELINE_MODEL_NAMES = ["A_B", "BERT+MLP", "BERT-base"]
IMPROVED_MODEL_NAMES = ["D_B", "D_R", "RoBERTa+VADER", "BERT+VADER"]


# ╔══════════════════════════════════════════════════════════╗
# ║  3. 모델 로딩 헬퍼 함수들                                 ║
# ╚══════════════════════════════════════════════════════════╝

# ── 벤치마크 레지스트리 로드 ───────────────────────────────
# 벤치마크 실행 결과가 JSON으로 저장되어 있는데, 그걸 읽어오는 함수예요.
# 아직 벤치마크를 안 돌렸다면 친절하게 에러를 알려줍니다.
def _load_best_registry() -> dict[str, dict[str, Any]]:
    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError("Benchmark results are missing. Run the benchmark first.")
    return load_json(BEST_MODELS_PATH)


# ── 최고 성능 개선 모델 선택 ───────────────────────────────
# v2 D 조건 또는 v1 VADER 모델 중 저장된 결과에서 최고 성능 모델을 고릅니다.
def _select_best_improved_model_name(registry: dict[str, dict[str, Any]]) -> str:
    """Select the strongest improved model from saved benchmark artifacts."""

    # [전략 1] 벤치마크 요약 CSV가 있으면 --> 거기서 가장 높은 F1 점수로 선택
    # macro_f1이 같으면 precision, recall, accuracy 순으로 비교해요 (타이브레이커!)
    if BENCHMARK_SUMMARY_PATH.exists():
        summary_frame = pd.read_csv(BENCHMARK_SUMMARY_PATH)
        candidate_frame = summary_frame[summary_frame["model"].isin(IMPROVED_MODEL_NAMES)].copy()
        if not candidate_frame.empty:
            candidate_frame = candidate_frame.sort_values(
                by=["macro_f1_mean", "macro_precision_mean", "macro_recall_mean", "accuracy_mean"],
                ascending=[False, False, False, False],
            )
            return str(candidate_frame.iloc[0]["model"])

    # [전략 2] CSV가 없으면 --> 레지스트리에서 val F1 기준으로 선택 (폴백)
    available_candidates = [
        name
        for name in IMPROVED_MODEL_NAMES
        if name in registry
    ]
    if not available_candidates:
        raise KeyError("No improved-model registry entries were found for XAI.")

    # max()에 key를 넘겨서 가장 높은 validation F1을 가진 모델명을 반환!
    return max(
        available_candidates,
        key=lambda name: registry[name].get("best_val_macro_f1", float("-inf")),
    )


def _select_baseline_model_name(registry: dict[str, dict[str, Any]]) -> str:
    """v2 조건명이 있으면 A_B를 기준점으로, 없으면 v1 BERT-base를 사용합니다."""
    for name in BASELINE_MODEL_NAMES:
        if name in registry:
            return name
    raise KeyError("No baseline registry entry was found for XAI.")


# ── 모델 인스턴스화 (체크포인트 --> 실행 가능한 모델) ──────
# 저장된 체크포인트 파일(.pt)을 읽어서 실제로 추론할 수 있는 모델 번들을 만드는 함수예요.
# "냉동 보관된 모델을 해동해서 바로 쓸 수 있게 준비하는 과정"이라고 생각하면 됩니다!
def _instantiate_bundle(display_name: str, record: dict[str, Any]) -> LoadedModelBundle:
    # 1) 체크포인트 파일 로드 (CPU에 올립니다 - XAI는 CPU에서 돌아가니까요)
    checkpoint_path = Path(record["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 2) 하이퍼파라미터 복원 (학습 때 사용한 설정값을 그대로 재현해야 해요!)
    hyperparams = record.get("hyperparams", {})
    dropout = float(hyperparams.get("dropout") or 0.1)
    hidden_dim = int(hyperparams.get("mlp_hidden") or 256)

    # 3) 모델 아키텍처 생성 (display_name에 따라 적절한 클래스를 골라줍니다)
    if display_name == "BERT-base":
        # Baseline: 순수 BERT만 사용하는 분류기
        model = TransformerCLSClassifier(
            model_name="bert-base-uncased",
            dropout=dropout,
        )
        model_type = "transformer"
        model_name = "bert-base-uncased"
    elif display_name == "BERT+MLP":
        model = TransformerConditionClassifier(
            model_name="bert-base-uncased",
            use_vader=False,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model_type = "transformer"
        model_name = "bert-base-uncased"
    elif display_name == "BERT+VADER":
        # 개선 모델 A: BERT + VADER 감성 점수를 결합한 하이브리드
        model = HybridSentimentClassifier(
            model_name="bert-base-uncased",
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model_type = "hybrid"
        model_name = "bert-base-uncased"
    elif display_name == "RoBERTa+VADER":
        # 개선 모델 B: RoBERTa + VADER 감성 점수를 결합한 하이브리드
        model = HybridSentimentClassifier(
            model_name="roberta-base",
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        model_type = "hybrid"
        model_name = "roberta-base"
    elif display_name in {"A_B", "B_B", "C_B", "D_B", "A_R", "B_R", "C_R", "D_R"}:
        model_name = "roberta-base" if display_name.endswith("_R") else "bert-base-uncased"
        use_vader = display_name.startswith(("C_", "D_"))
        target_labels = hyperparams.get("target_labels") or []
        model = TransformerConditionClassifier(
            model_name=model_name,
            use_vader=use_vader,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_targets=len(target_labels) if record.get("use_target_aux") else 0,
        )
        model_type = "hybrid" if use_vader else "transformer"
    else:
        raise ValueError(f"Unsupported XAI model: {display_name}")

    # 4) 학습된 가중치를 모델에 로드하고, 평가 모드(eval)로 전환
    #    eval() 모드에서는 Dropout이 비활성화되어 일관된 결과를 줍니다.
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 5) 토크나이저도 함께 준비해서 번들로 묶어 반환!
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return LoadedModelBundle(
        display_name=display_name,
        model_type=model_type,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),  # XAI 분석은 항상 CPU에서!
    )


# ── 기준 조건 + 비교 조건 모델 동시 로드 ───────────────────
# v2 결과가 있으면 A_B와 D_B/D_R를, 없으면 v1 fallback checkpoint를 준비합니다.
def load_bundles_for_xai() -> tuple[LoadedModelBundle, LoadedModelBundle]:
    """Load the reference and comparison models used in the XAI report."""
    registry = _load_best_registry()
    baseline_model_name = _select_baseline_model_name(registry)
    baseline = _instantiate_bundle(baseline_model_name, registry[baseline_model_name])
    improved_model_name = _select_best_improved_model_name(registry)
    improved = _instantiate_bundle(improved_model_name, registry[improved_model_name])
    return baseline, improved


# ╔══════════════════════════════════════════════════════════╗
# ║  4. 예측 및 전처리 함수들                                 ║
# ╚══════════════════════════════════════════════════════════╝

# ── VADER 감성 점수 계산 ───────────────────────────────────
# VADER(Valence Aware Dictionary and sEntiment Reasoner)는 규칙 기반 감성 분석 도구예요.
# 각 텍스트에 대해 neg, neu, pos, compound 점수를 계산해서 numpy 배열로 반환합니다.
# Hybrid 모델은 이 점수를 BERT 임베딩과 결합해서 판단한답니다!
def _compute_vader_array(texts: list[str]) -> np.ndarray:
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        rows.append([scores[column] for column in VADER_COLUMNS])
    return np.asarray(rows, dtype=np.float32)


# ── 확률 예측 함수 (핵심!) ─────────────────────────────────
# 이 함수는 XAI 파이프라인에서 가장 많이 호출되는 함수예요!
# SHAP과 LIME 모두 이 함수를 통해 모델의 예측 확률을 받아갑니다.
# 텍스트 리스트를 넣으면 (샘플 수 x 3) 형태의 확률 배열이 나와요.
#   예: [[0.8, 0.15, 0.05], ...]  --> [hate 80%, offensive 15%, normal 5%]
def predict_probabilities(bundle: LoadedModelBundle, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """텍스트 리스트에 대해 3-class 확률 예측. LIME의 predict_fn으로도 사용됨."""

    # 빈 입력이면 빈 배열 반환 (에지 케이스 처리는 항상 중요해요!)
    if not texts:
        return np.zeros((0, NUM_LABELS), dtype=np.float32)

    device = bundle.device
    bundle.model.to(device)
    all_probabilities = []

    # Hybrid 모델이면 VADER 감성 점수를 미리 계산해둡니다
    vader_array = _compute_vader_array(texts) if bundle.model_type == "hybrid" else None

    # torch.no_grad(): 추론 시에는 gradient 계산이 필요 없어요. 메모리 절약!
    with torch.no_grad():
        # 배치 단위로 처리 (한 번에 너무 많이 넣으면 메모리 터질 수 있으니까요)
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            # 토크나이저가 텍스트를 숫자(토큰 ID)로 변환해줍니다
            encoded = bundle.tokenizer(
                batch_texts,
                truncation=True,       # max_len 넘으면 자르기
                padding=True,          # 배치 내 최대 길이에 맞춰 패딩
                max_length=get_config().max_len,
                return_tensors="pt",   # PyTorch 텐서로 반환
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # 모델 타입에 따라 forward pass가 달라요
            if bundle.model_type == "hybrid":
                # Hybrid: 텍스트 토큰 + VADER 감성 점수를 함께 입력
                vader = torch.tensor(vader_array[start : start + batch_size], dtype=torch.float32).to(device)
                logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask, vader=vader)
            else:
                # Transformer: 텍스트 토큰만 입력
                logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)

            # softmax로 logit을 확률로 변환 (각 클래스의 확률 합 = 1.0)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probabilities.append(probabilities)

    # 메모리 정리: 모델을 CPU로 옮기고 캐시 비우기
    bundle.model.to("cpu")
    clear_device_cache()
    return np.vstack(all_probabilities)  # 모든 배치 결과를 하나로 합치기


# ── 토큰 정규화 ───────────────────────────────────────────
# BERT는 "##ing" 같은 서브워드를, RoBERTa는 "Ġhello" 같은 형태를 사용해요.
# 이런 접두사/접미사를 제거해서 SHAP과 LIME 토큰을 공정하게 비교할 수 있게 합니다.
def _normalize_token(token: str) -> str:
    return token.lower().replace("##", "").replace("Ġ", "").strip()


def _is_subword_continuation(token: str) -> bool:
    """서브워드가 이전 단어의 연속인지 판별.

    BERT 계열은 '##' prefix로, RoBERTa 계열은 'Ġ' prefix로 단어 시작을 표시합니다.
    - BERT: "playing" -> ["play", "##ing"] -- "##"이 붙으면 연속
    - RoBERTa: "playing" -> ["Ġplay", "ing"] -- "Ġ"이 없으면 연속 (첫 토큰 제외)
    """
    if token.startswith("##"):
        return True  # BERT 서브워드 연속
    # RoBERTa: Ġ로 시작하면 새 단어, 아니면 연속
    # 단, 특수토큰이나 숫자 등은 Ġ 없이도 새 단어일 수 있으므로 보수적으로 판단
    return False


def _aggregate_subword_scores(
    tokens: list[str],
    scores: np.ndarray,
    special_tokens: set[str],
) -> list[dict[str, Any]]:
    """서브워드 토큰의 SHAP 점수를 원래 단어 단위로 합산(sum).

    SHAP은 토크나이저 단위로 점수를 매기므로 "niggers" -> ["nig", "##ger", "##s"]처럼
    의미 없는 파편이 Top-5에 올라오는 문제가 있습니다.
    이 함수는 서브워드 점수를 원래 단어로 합산하여 해석 가능한 단위로 만들어줍니다.

    합산 방식: sum (SHAP의 additive 속성상 부분 기여도의 합 = 전체 기여도)
    """
    if len(tokens) == 0 or len(scores) == 0:
        return []

    # Step 1: 서브워드를 단어 단위로 그룹화
    word_groups: list[dict[str, Any]] = []  # [{"word": str, "score_sum": float}, ...]
    current_word = ""
    current_score = 0.0
    has_bert_subword = any(t.startswith("##") for t in tokens)

    for token, score in zip(tokens, scores):
        # 특수 토큰은 건너뛰기
        if token in special_tokens:
            continue
        normalized = _normalize_token(token)
        if not normalized:
            continue

        if has_bert_subword:
            # BERT 토크나이저: ## prefix로 연속 판별
            if token.startswith("##"):
                # 이전 단어에 이어붙이기
                current_word += normalized
                current_score += float(score)
            else:
                # 새 단어 시작 -- 이전 단어 저장
                if current_word:
                    word_groups.append({"word": current_word, "score": current_score, "abs_score": abs(current_score)})
                current_word = normalized
                current_score = float(score)
        else:
            # RoBERTa 토크나이저: Ġ prefix로 새 단어 시작 판별
            if token.startswith("Ġ") or not word_groups and not current_word:
                # 새 단어 시작
                if current_word:
                    word_groups.append({"word": current_word, "score": current_score, "abs_score": abs(current_score)})
                current_word = normalized
                current_score = float(score)
            else:
                # 이전 단어에 이어붙이기
                current_word += normalized
                current_score += float(score)

    # 마지막 단어 저장
    if current_word:
        word_groups.append({"word": current_word, "score": current_score, "abs_score": abs(current_score)})

    return word_groups


# ── SHAP 점수 추출 (차원 처리) ─────────────────────────────
# SHAP의 결과 배열은 상황에 따라 차원이 달라질 수 있어서, 여기서 안전하게 처리해요.
# - 1D: 이미 토큰별 점수 --> 그대로 반환
# - 2D (tokens x labels): 행이 토큰, 열이 라벨 --> 예측 라벨 열만 추출
# - 2D (labels x tokens): 행이 라벨, 열이 토큰 --> 예측 라벨 행만 추출
# 어떤 형태가 오더라도 "토큰별 SHAP 기여도" 1D 배열을 반환하는 게 목표!
def _extract_shap_scores(values: np.ndarray, tokens: list[str], predicted_label: int) -> np.ndarray:
    if values.ndim == 1:
        return values                            # 이미 1D면 바로 OK
    if values.ndim != 2:
        return np.zeros(len(tokens), dtype=float) # 3D 이상은 예상 밖 -> 안전하게 0 반환
    if values.shape[0] == len(tokens):
        return values[:, predicted_label]         # (tokens, labels) 형태
    if values.shape[1] == len(tokens):
        return values[predicted_label, :]         # (labels, tokens) 형태
    return np.zeros(len(tokens), dtype=float)     # 매칭 안 되면 안전하게 0 반환


# ╔══════════════════════════════════════════════════════════╗
# ║  5. XAI 설명 생성 함수들 (SHAP & LIME)                   ║
# ╚══════════════════════════════════════════════════════════╝

# ── SHAP 설명 생성 ────────────────────────────────────────
def run_shap_explanations(
    bundle: LoadedModelBundle,
    texts: list[str],
    predicted_labels: list[int],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    """
    SHAP 설명 생성 (Lundberg & Lee, 2017).
    각 토큰의 예측 기여도(Shapley value)를 계산하여 Top-5 중요 토큰 추출.
    CPU에서 실행됨 -- MPS 미지원.

    쉽게 말하면: "이 문장에서 어떤 단어가 예측에 가장 큰 영향을 줬을까?"
    를 게임 이론의 Shapley value로 정량화하는 거예요!
    """

    # Step 1: SHAP Explainer 생성
    # predict_probabilities를 wrapping해서 SHAP이 호출할 수 있는 형태로 만듭니다.
    # tokenizer를 넘겨주면 SHAP이 알아서 토큰 단위로 기여도를 계산해줘요.
    explainer = shap.Explainer(
        lambda batch: predict_probabilities(bundle, list(batch)),
        bundle.tokenizer,
        output_names=LABEL_NAMES,
    )

    # Step 2: SHAP 값 계산 (시간이 좀 걸릴 수 있어요 -- 커피 한 잔 하고 오세요!)
    # max_evals: 함수 호출 횟수 제한, batch_size: 한 번에 처리할 샘플 수
    shap_values = explainer(
        texts,
        max_evals=config.shap_max_evals,
        batch_size=config.shap_batch_size,
    )

    # Step 3: 각 샘플별로 Top-5 중요 토큰 추출 (서브워드 → 단어 aggregation 적용)
    results = []
    special_tokens = set(bundle.tokenizer.all_special_tokens)  # [CLS], [SEP] 등 제외용
    for index, text in enumerate(texts):
        # SHAP이 반환한 토큰 데이터 가져오기 (문자열일 수도, 리스트일 수도 있어요)
        raw_tokens = shap_values.data[index]
        tokens = [str(token) for token in raw_tokens] if not isinstance(raw_tokens, str) else bundle.tokenizer.tokenize(raw_tokens)

        # 예측 라벨에 해당하는 SHAP 점수만 추출
        token_scores = _extract_shap_scores(np.asarray(shap_values.values[index]), tokens, predicted_labels[index])

        # 서브워드 점수를 단어 단위로 합산 (A-1 개선)
        # SHAP의 additive 속성상 서브워드 점수의 합 = 원래 단어의 기여도
        # "nig" + "##ger" + "##s" = "niggers" (하나의 해석 가능한 단위!)
        word_pairs = _aggregate_subword_scores(tokens, token_scores, special_tokens)
        word_pairs = sorted(word_pairs, key=lambda item: item["abs_score"], reverse=True)

        # Top-5 단어와 점수를 결과에 담기
        results.append(
            {
                "text": text,
                "top_tokens": [item["word"] for item in word_pairs[:5]],
                "top_scores": [item["score"] for item in word_pairs[:5]],
                "token_details": word_pairs,  # 전체 단어 정보도 보관
            }
        )
    return results


# ── LIME 설명 생성 ────────────────────────────────────────
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

    SHAP이 게임 이론이라면, LIME은 "단어를 하나씩 빼보면서 예측이 얼마나 변하는지"
    관찰하는 방식이에요. 서로 다른 원리이므로, 둘이 같은 토큰을 가리키면 더 믿을 수 있죠!
    """

    # Step 1: LIME Explainer 생성 (공백 기준으로 텍스트를 분리)
    explainer = LimeTextExplainer(class_names=LABEL_NAMES, split_expression=r"\s+")
    results = []

    # predict_fn: LIME이 perturbation된 텍스트를 예측할 때 호출하는 함수
    def predict_fn(batch_texts: list[str]) -> np.ndarray:
        return predict_probabilities(bundle, list(batch_texts))

    # Step 2: 각 텍스트에 대해 LIME 설명 생성 (샘플별로 하나씩!)
    for text, predicted_label in zip(texts, predicted_labels):
        # explain_instance: 이 텍스트 주변에서 로컬 선형 모델을 학습시킵니다
        # num_samples: perturbation 횟수 (많을수록 정확하지만 느려요)
        explanation = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_fn,
            labels=[0, 1, 2],  # hatespeech, offensive, normal
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
        )

        # 예측 라벨에 대한 피처 가중치를 추출 (어떤 단어가 얼마나 기여했는지!)
        pred_label = int(predicted_label)
        # LIME이 해당 라벨의 설명을 생성하지 못한 경우 안전하게 처리
        available_labels = list(explanation.local_exp.keys())
        if pred_label not in available_labels and available_labels:
            pred_label = available_labels[0]
        feature_weights = explanation.as_list(label=pred_label)
        results.append(
            {
                "text": text,
                "top_tokens": [token for token, _ in feature_weights[:5]],
                "top_scores": [float(weight) for _, weight in feature_weights[:5]],
                "feature_weights": [{"token": token, "score": float(weight)} for token, weight in feature_weights],
            }
        )
    return results


# ╔══════════════════════════════════════════════════════════╗
# ║  6. Overlap@5 및 시각화 함수들                            ║
# ╚══════════════════════════════════════════════════════════╝

# ── Overlap@5 계산 ────────────────────────────────────────
# SHAP Top-5 토큰과 LIME Top-5 토큰이 얼마나 겹치는지 계산하는 함수!
# 이 수치가 높을수록(>=60%) 두 기법이 "이 단어가 중요하다"고 의견이 일치한다는 뜻이에요.
# 연구에서 XAI 결과의 "신뢰도"를 판단하는 핵심 지표입니다.
def _compute_overlap_at_5(
    shap_results: list[dict[str, Any]],
    lime_results: list[dict[str, Any]],
) -> list[float]:
    overlaps = []
    for shap_result, lime_result in zip(shap_results, lime_results):
        # 각각의 Top-5 토큰을 정규화해서 집합(set)으로 만들기
        shap_top = {_normalize_token(token) for token in shap_result["top_tokens"] if _normalize_token(token)}
        lime_top = {_normalize_token(token) for token in lime_result["top_tokens"] if _normalize_token(token)}

        # 퍼지 매칭: BERT의 서브워드("##ing")와 LIME의 전체 단어("running")가
        # 부분 문자열 관계면 매칭으로 인정해줍니다 (좀 더 공정한 비교를 위해!)
        matched_shap_tokens = set()
        for shap_token in shap_top:
            if shap_token in lime_top:
                matched_shap_tokens.add(shap_token)
                continue
            for lime_token in lime_top:
                if shap_token in lime_token or lime_token in shap_token:
                    matched_shap_tokens.add(shap_token)
                    break

        # 겹치는 토큰 수 / 5 = Overlap@5 비율 (0.0 ~ 1.0)
        # matched_shap_tokens에는 exact + fuzzy 매칭 결과가 모두 들어있으므로
        # lime_top과 다시 교집합하면 fuzzy 매칭이 날아가버려요! 그래서 그대로 사용합니다.
        overlaps.append(len(matched_shap_tokens) / 5.0)
    return overlaps


# ── Overlap@5 박스플롯 시각화 ──────────────────────────────
# 기준 조건과 비교 조건의 Overlap@5 분포를 나란히 비교하는 차트입니다.
# 빨간 점선이 신뢰 임계값(0.6)인데, 이 위에 있으면 "설명을 믿을 만하다"는 뜻이에요.
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


# ── 케이스별 SHAP 비교 차트 ────────────────────────────────
# 기준 조건과 비교 조건의 SHAP Top-5를 나란히 보여주는 가로 막대 그래프입니다.
# XAI 결과는 설계 입력이 아니라 학습 완료 후 조건 간 차이 해석에만 사용합니다.
def _plot_case_comparison(
    case_index: int,
    text: str,
    baseline_result: dict[str, Any],
    improved_result: dict[str, Any],
    output_path: Path,
    baseline_name: str = "BERT-base",
    improved_name: str = "Improved",
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 좌: Baseline, 우: Improved
    for axis, title, result in [
        (axes[0], f"{baseline_name} SHAP", baseline_result),
        (axes[1], f"{improved_name} SHAP", improved_result),
    ]:
        tokens = result["top_tokens"][:5]
        scores = result["top_scores"][:5]
        axis.barh(range(len(tokens)), scores, color="#4c72b0")
        axis.set_yticks(range(len(tokens)))
        axis.set_yticklabels(tokens)
        axis.invert_yaxis()  # 가장 중요한 토큰이 맨 위에 오도록!
        axis.set_title(title)
    fig.suptitle(f"Case {case_index + 1}: {text[:90]}...")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  6-1. Human Rationale 비교 (설명 타당성 평가)             ║
# ╚══════════════════════════════════════════════════════════╝
#
# HateXplain 데이터셋에는 annotator들이 "왜 혐오표현인지" 근거 토큰을 표시한
# word-level rationale이 포함되어 있어요. 이걸 활용해서:
#   - 모델이 중요하다고 본 토큰(SHAP/LIME Top-5)이 인간의 판단 근거와 얼마나 일치하는지 측정
#   - Overlap@5(안정성)와 별개로 "설명 타당성"을 평가하는 독립적인 지표
#
# 이는 "두 XAI 기법이 서로 동의한다"(안정성)와
# "모델이 인간과 같은 근거를 본다"(타당성)는 전혀 다른 질문에 답합니다.

import json as _json
import math as _math


def _load_human_rationales() -> dict[str, list[str]]:
    """
    dataset.json에서 human rationale을 로드하여 post_id -> rationale 토큰 리스트로 반환.

    각 annotator의 binary mask를 majority vote로 합산해서,
    과반수(ceil(n/2)) 이상이 표시한 토큰만 rationale로 인정합니다.
    이렇게 하면 개별 annotator의 주관적 편향을 줄일 수 있어요.
    """
    if not RAW_DATASET_PATH.exists():
        return {}

    with open(RAW_DATASET_PATH, "r", encoding="utf-8") as f:
        raw = _json.load(f)

    rationale_map: dict[str, list[str]] = {}
    for post_id, sample in raw.items():
        rats = sample.get("rationales", [])
        tokens = sample.get("post_tokens", [])
        if not rats or not tokens:
            continue

        # annotator 수에서 과반수 기준 결정
        n_annotators = len(rats)
        threshold = _math.ceil(n_annotators / 2)

        # majority vote: 과반수 이상이 표시한 토큰만 rationale로 인정
        rationale_tokens = []
        for tok_idx in range(len(tokens)):
            votes = sum(r[tok_idx] for r in rats if tok_idx < len(r))
            if votes >= threshold:
                rationale_tokens.append(tokens[tok_idx].lower())

        if rationale_tokens:
            rationale_map[post_id] = rationale_tokens

    return rationale_map


def _compute_rationale_overlap(
    xai_results: list[dict[str, Any]],
    post_ids: list[str],
    rationale_map: dict[str, list[str]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Model Top-k 토큰과 Human rationale 토큰의 overlap을 계산.

    "인간 rationale 토큰 중 모델이 커버한 비율"을 측정합니다:
      overlap = |covered human tokens| / |Human Rationale|

    human 토큰 기준으로 세기 때문에 서브워드 퍼지 매칭에서도
    0.0~1.0 범위가 보장됩니다.

    반환값: 샘플별 overlap 정보 리스트
    """
    results = []
    for xai_result, pid in zip(xai_results, post_ids):
        human_tokens = rationale_map.get(pid, [])
        if not human_tokens:
            # rationale이 없는 샘플(normal 등)은 건너뛰기
            continue

        # 모델의 Top-k 토큰을 정규화
        model_top = {_normalize_token(t) for t in xai_result["top_tokens"][:k] if _normalize_token(t)}
        human_set = set(human_tokens)

        # human 토큰 기준: 각 human 토큰이 model top-k에 의해 커버되는지 확인
        # 퍼지 매칭: 서브워드와 전체 단어 간 부분 문자열 관계도 인정
        covered_human = set()
        for human_token in human_set:
            if human_token in model_top:
                covered_human.add(human_token)
                continue
            for model_token in model_top:
                if model_token in human_token or human_token in model_token:
                    covered_human.add(human_token)
                    break

        overlap = len(covered_human) / len(human_set) if human_set else 0.0

        results.append({
            "post_id": pid,
            "model_top_tokens": list(model_top),
            "human_rationale_tokens": human_tokens,
            "matched_tokens": list(covered_human),
            "overlap": round(overlap, 4),
            "human_rationale_count": len(human_set),
        })

    return results


# ╔══════════════════════════════════════════════════════════╗
# ║  6-A. 마스킹 기반 충실도 검증 (XAI 4축 중 Faithfulness)      ║
# ╚══════════════════════════════════════════════════════════╝
#
# SHAP과 LIME은 "어떤 토큰이 중요한가"를 측정하지만,
# 그 토큰이 정말로 예측에 인과적 영향을 미치는지는 별개의 질문입니다.
# 마스킹 검증은 토큰을 실제로 제거/유지하여 예측 변화를 관찰함으로써
# XAI 설명의 "충실도(faithfulness)"를 검증합니다 (DeYoung et al., 2020).


def _mask_tokens_in_text(text: str, tokens_to_mask: set[str]) -> str:
    """텍스트에서 지정된 토큰(단어)을 [MASK]로 치환.

    공백 기준으로 단어를 분리하고, 정규화된 형태가 mask 대상에 포함되면 치환합니다.
    퍼지 매칭도 지원: mask 대상이 단어의 부분 문자열이거나 그 반대인 경우도 치환.
    """
    words = text.split()
    masked = []
    for word in words:
        word_lower = word.lower().strip(".,!?;:'\"()[]{}#@")
        should_mask = False
        for target in tokens_to_mask:
            if word_lower == target or target in word_lower or word_lower in target:
                should_mask = True
                break
        masked.append("[MASK]" if should_mask else word)
    return " ".join(masked)


def _compute_masking_metrics(
    bundle: LoadedModelBundle,
    xai_results: list[dict[str, Any]],
    texts: list[str],
    predicted_labels: list[int],
    k: int = 5,
) -> dict[str, Any]:
    """Comprehensiveness와 Sufficiency를 계산 (DeYoung et al., 2020).

    마스킹 기반 인과적 검증의 핵심 메트릭 두 가지:

    Comprehensiveness (포괄성):
      Top-k 토큰을 제거했을 때 예측 확률이 얼마나 떨어지는가?
      높을수록 "XAI가 지목한 토큰이 진짜 중요했다" = 설명이 충실

    Sufficiency (충분성):
      Top-k 토큰만 남기고 나머지를 제거했을 때 예측이 유지되는가?
      낮을수록 "Top-k만으로 충분히 예측 가능" = 집중된 설명

    반환값: 샘플별 메트릭과 전체 평균을 담은 딕셔너리
    """
    comp_scores = []   # 포괄성 점수 (sample-level)
    suff_scores = []   # 충분성 점수 (sample-level)
    details = []       # 상세 결과 (디버깅 및 분석용)

    for idx, (xai_result, text, pred_label) in enumerate(zip(xai_results, texts, predicted_labels)):
        top_tokens = {t.lower() for t in xai_result["top_tokens"][:k]}
        if not top_tokens:
            continue

        # 원본 예측 확률
        original_prob = predict_probabilities(bundle, [text])[0]
        original_score = float(original_prob[pred_label])

        # Comprehensiveness: Top-k 제거
        text_removed = _mask_tokens_in_text(text, top_tokens)
        removed_prob = predict_probabilities(bundle, [text_removed])[0]
        removed_score = float(removed_prob[pred_label])
        comp = original_score - removed_score  # 높을수록 좋음

        # Sufficiency: Top-k만 유지 (나머지 제거)
        words = text.split()
        kept_words = []
        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"()[]{}#@")
            is_top = False
            for target in top_tokens:
                if word_lower == target or target in word_lower or word_lower in target:
                    is_top = True
                    break
            kept_words.append(word if is_top else "[MASK]")
        text_kept = " ".join(kept_words)

        kept_prob = predict_probabilities(bundle, [text_kept])[0]
        kept_score = float(kept_prob[pred_label])
        suff = original_score - kept_score  # 낮을수록 좋음

        comp_scores.append(comp)
        suff_scores.append(suff)
        details.append({
            "sample_idx": idx,
            "text": text,
            "top_tokens": list(top_tokens),
            "original_prob": round(original_score, 4),
            "removed_prob": round(removed_score, 4),
            "kept_prob": round(kept_score, 4),
            "comprehensiveness": round(comp, 4),
            "sufficiency": round(suff, 4),
        })

    return {
        "comprehensiveness_mean": round(float(np.mean(comp_scores)), 4) if comp_scores else None,
        "sufficiency_mean": round(float(np.mean(suff_scores)), 4) if suff_scores else None,
        "comprehensiveness_std": round(float(np.std(comp_scores)), 4) if comp_scores else None,
        "sufficiency_std": round(float(np.std(suff_scores)), 4) if suff_scores else None,
        "sample_count": len(comp_scores),
        "details": details,
    }


# ── 비하어(Slur) 사전 ────────────────────────────────────────
# Slur-Free Prediction에서 마스킹할 명시적 비하어 목록.
# HateXplain 데이터셋에서 빈도가 높은 표면 비하어를 수집.
# 이 목록은 완벽할 필요 없이, "명시적 비하어 의존도"를 측정하는 데 충분하면 됩니다.
EXPLICIT_SLURS = {
    # 인종 관련
    "nigger", "niggers", "nigga", "niggas", "negro", "negros",
    "chink", "chinks", "gook", "gooks", "spic", "spics", "wetback",
    "kike", "kikes", "coon", "coons", "darkie", "jap", "japs",
    # 성별/성적지향 관련
    "faggot", "faggots", "fag", "fags", "dyke", "dykes",
    "tranny", "trannies", "homo", "homos",
    # 종교 관련
    "raghead", "ragheads", "towelhead", "towelheads", "sandnigger",
    # 여성 비하
    "bitch", "bitches", "whore", "whores", "slut", "sluts", "cunt", "cunts",
    # 장애 관련
    "retard", "retards", "retarded",
    # 일반 비하
    "trash", "scum", "vermin", "subhuman", "mongrel",
}


def _compute_slur_free_prediction(
    bundle: LoadedModelBundle,
    texts: list[str],
    labels: np.ndarray,
    predicted_labels: list[int],
) -> dict[str, Any]:
    """비하어를 마스킹한 뒤에도 모델이 혐오를 탐지하는지 검증 (Slur-Free Prediction).

    핵심 질문: "모든 명시적 비하어를 제거해도 모델이 맥락만으로 혐오를 잡아내는가?"

    이 실험은 모델의 "맥락 이해 능력"을 직접적으로 측정합니다:
    - 비하어 의존도가 높으면: 비하어 제거 시 예측 확률이 크게 하락
    - 맥락 이해가 좋으면: 비하어 없이도 주변 단어 조합으로 혐오를 탐지

    대상: hatespeech/offensive 클래스만 (normal은 비하어가 없으므로 제외)
    """
    results = []
    hate_offensive_mask = labels < 2  # 0=hate, 1=offensive만 대상

    for idx in range(len(texts)):
        if not hate_offensive_mask[idx]:
            continue

        text = texts[idx]
        true_label = int(labels[idx])
        pred_label = predicted_labels[idx]

        # 텍스트에 비하어가 있는지 확인
        words_lower = {w.lower().strip(".,!?;:'\"()[]{}#@") for w in text.split()}
        slurs_found = words_lower & EXPLICIT_SLURS
        if not slurs_found:
            continue  # 비하어가 없는 문장은 이 실험에 해당하지 않음

        # 원본 예측
        original_prob = predict_probabilities(bundle, [text])[0]
        original_hate_prob = float(original_prob[true_label])

        # 비하어 마스킹 후 예측
        masked_text = _mask_tokens_in_text(text, slurs_found)
        masked_prob = predict_probabilities(bundle, [masked_text])[0]
        masked_hate_prob = float(masked_prob[true_label])
        masked_pred = int(masked_prob.argmax())

        # 예측 유지 여부: 마스킹 후에도 올바른 라벨을 예측하는가?
        prediction_retained = masked_pred == true_label

        results.append({
            "sample_idx": idx,
            "text": text,
            "true_label": true_label,
            "slurs_found": list(slurs_found),
            "slur_count": len(slurs_found),
            "original_prob": round(original_hate_prob, 4),
            "masked_prob": round(masked_hate_prob, 4),
            "prob_drop": round(original_hate_prob - masked_hate_prob, 4),
            "original_pred": pred_label,
            "masked_pred": masked_pred,
            "prediction_retained": prediction_retained,
        })

    if not results:
        return {
            "slur_free_accuracy": None,
            "mean_prob_drop": None,
            "sample_count": 0,
            "retention_count": 0,
            "details": [],
        }

    retention_count = sum(1 for r in results if r["prediction_retained"])
    prob_drops = [r["prob_drop"] for r in results]

    return {
        "slur_free_accuracy": round(retention_count / len(results), 4),
        "mean_prob_drop": round(float(np.mean(prob_drops)), 4),
        "std_prob_drop": round(float(np.std(prob_drops)), 4),
        "sample_count": len(results),
        "retention_count": retention_count,
        "details": results,
    }


def _gini(values: list[float]) -> float | None:
    array = np.asarray([abs(value) for value in values if value is not None], dtype=float)
    if array.size == 0 or float(array.sum()) == 0.0:
        return None
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * array) / (n * np.sum(array))) - ((n + 1) / n))


def _top_token_candidates(xai_result: dict[str, Any], max_tokens: int = 12) -> list[str]:
    details = xai_result.get("token_details") or []
    if details:
        ranked = sorted(details, key=lambda item: float(item.get("abs_score", 0.0)), reverse=True)
        return [str(item.get("word") or item.get("token")) for item in ranked[:max_tokens] if item.get("word") or item.get("token")]
    return [str(token) for token in xai_result.get("top_tokens", [])[:max_tokens]]


def _keep_only_tokens(text: str, tokens_to_keep: set[str]) -> str:
    kept_words = []
    for word in text.split():
        word_lower = word.lower().strip(".,!?;:'\"()[]{}#@")
        should_keep = any(word_lower == token or token in word_lower or word_lower in token for token in tokens_to_keep)
        kept_words.append(word if should_keep else "[MASK]")
    return " ".join(kept_words)


def _compute_ci_scores(xai_results: list[dict[str, Any]]) -> list[float]:
    scores = []
    for result in xai_results:
        details = result.get("token_details") or []
        values = [float(item.get("score", 0.0)) for item in details]
        if not values:
            values = [float(score) for score in result.get("top_scores", [])]
        gini_value = _gini(values)
        if gini_value is not None:
            scores.append(gini_value)
    return scores


def _compute_mss_scores(
    bundle: LoadedModelBundle,
    xai_results: list[dict[str, Any]],
    texts: list[str],
    predicted_labels: list[int],
    threshold: float,
) -> list[int]:
    mss_scores = []
    for xai_result, text, pred_label in zip(xai_results, texts, predicted_labels):
        candidates = _top_token_candidates(xai_result)
        if not candidates:
            continue
        original_score = float(predict_probabilities(bundle, [text])[0][pred_label])
        target_score = threshold * original_score
        found_size = len(candidates)
        for size in range(1, len(candidates) + 1):
            kept_text = _keep_only_tokens(text, {token.lower() for token in candidates[:size]})
            kept_score = float(predict_probabilities(bundle, [kept_text])[0][pred_label])
            if kept_score >= target_score:
                found_size = size
                break
        mss_scores.append(found_size)
    return mss_scores


def _compute_loo_scores(
    bundle: LoadedModelBundle,
    xai_results: list[dict[str, Any]],
    texts: list[str],
    predicted_labels: list[int],
    k: int = 5,
) -> list[float]:
    loo_scores = []
    for xai_result, text, pred_label in zip(xai_results, texts, predicted_labels):
        candidates = _top_token_candidates(xai_result, max_tokens=k)
        if not candidates:
            continue
        original_score = float(predict_probabilities(bundle, [text])[0][pred_label])
        drops = []
        for token in candidates:
            masked_text = _mask_tokens_in_text(text, {token.lower()})
            masked_score = float(predict_probabilities(bundle, [masked_text])[0][pred_label])
            drops.append(original_score - masked_score)
        if drops:
            loo_scores.append(float(np.mean(drops)))
    return loo_scores


def _compute_interaction_strength(
    bundle: LoadedModelBundle,
    xai_results: list[dict[str, Any]],
    texts: list[str],
    predicted_labels: list[int],
    max_pairs: int,
) -> list[float]:
    from itertools import combinations

    scores = []
    used_pairs = 0
    for xai_result, text, pred_label in zip(xai_results, texts, predicted_labels):
        candidates = _top_token_candidates(xai_result, max_tokens=6)
        pair_scores = []
        original_score = float(predict_probabilities(bundle, [text])[0][pred_label])
        for token_a, token_b in combinations(candidates, 2):
            if used_pairs >= max_pairs:
                break
            token_a = token_a.lower()
            token_b = token_b.lower()
            score_a = float(predict_probabilities(bundle, [_mask_tokens_in_text(text, {token_a})])[0][pred_label])
            score_b = float(predict_probabilities(bundle, [_mask_tokens_in_text(text, {token_b})])[0][pred_label])
            score_ab = float(predict_probabilities(bundle, [_mask_tokens_in_text(text, {token_a, token_b})])[0][pred_label])
            pair_scores.append(abs(original_score - score_a - score_b + score_ab))
            used_pairs += 1
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
        if used_pairs >= max_pairs:
            break
    return scores


def _compute_attention_rollout_entropy(
    bundle: LoadedModelBundle,
    texts: list[str],
    max_samples: int,
) -> list[float]:
    model = bundle.model.to(torch.device("cpu"))
    entropies = []
    with torch.no_grad():
        for text in texts[:max_samples]:
            encoded = bundle.tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=get_config().max_len,
                return_tensors="pt",
            )
            try:
                outputs = model.encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    output_attentions=True,
                    return_dict=True,
                )
            except Exception:
                continue
            attentions = getattr(outputs, "attentions", None)
            if not attentions:
                continue
            seq_len = attentions[0].shape[-1]
            rollout = torch.eye(seq_len)
            for layer_attention in attentions:
                attention = layer_attention[0].mean(dim=0)
                attention = 0.5 * attention + 0.5 * torch.eye(seq_len)
                attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                rollout = attention @ rollout
            cls_flow = rollout[0]
            valid = encoded["attention_mask"][0].bool()
            cls_flow = cls_flow[valid]
            cls_flow = cls_flow / cls_flow.sum().clamp_min(1e-12)
            entropy = float(-(cls_flow * torch.log(cls_flow.clamp_min(1e-12))).sum())
            entropies.append(entropy)
    model.to(torch.device("cpu"))
    clear_device_cache()
    return entropies


def _summarize_metric(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "count": int(array.size),
    }


def _compute_v2_xai_metrics(
    bundle: LoadedModelBundle,
    xai_results: list[dict[str, Any]],
    texts: list[str],
    predicted_labels: list[int],
    config: ExperimentConfig,
) -> dict[str, Any]:
    sample_cap = min(len(texts), config.xai_context_sample_size)
    metric_texts = texts[:sample_cap]
    metric_results = xai_results[:sample_cap]
    metric_preds = predicted_labels[:sample_cap]
    return {
        "ci": _summarize_metric(_compute_ci_scores(metric_results)),
        "mss": _summarize_metric(_compute_mss_scores(bundle, metric_results, metric_texts, metric_preds, config.xai_mss_threshold)),
        "loo": _summarize_metric(_compute_loo_scores(bundle, metric_results, metric_texts, metric_preds)),
        "interaction_strength": _summarize_metric(
            _compute_interaction_strength(bundle, metric_results, metric_texts, metric_preds, config.xai_interaction_pairs)
        ),
        "attention_rollout_entropy": _summarize_metric(
            _compute_attention_rollout_entropy(bundle, metric_texts, sample_cap)
        ),
    }


def _plot_masking_comparison(
    baseline_masking: dict[str, Any],
    improved_masking: dict[str, Any],
    baseline_name: str,
    improved_name: str,
    output_path: Path,
) -> None:
    """기준 조건과 비교 조건의 마스킹 메트릭 비교 막대 그래프."""
    ensure_dir(output_path.parent)

    metrics = ["Comprehensiveness", "Sufficiency"]
    baseline_vals = [
        baseline_masking.get("comprehensiveness_mean", 0) or 0,
        baseline_masking.get("sufficiency_mean", 0) or 0,
    ]
    improved_vals = [
        improved_masking.get("comprehensiveness_mean", 0) or 0,
        improved_masking.get("sufficiency_mean", 0) or 0,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, baseline_vals, width, label=baseline_name, color="#5DADE2")
    bars2 = ax.bar(x + width / 2, improved_vals, width, label=improved_name, color="#F39C12")

    ax.set_ylabel("Score")
    ax.set_title("Masking Verification: Comprehensiveness & Sufficiency")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_slur_free_comparison(
    baseline_slur: dict[str, Any],
    improved_slur: dict[str, Any],
    baseline_name: str,
    improved_name: str,
    output_path: Path,
) -> None:
    """기준 조건과 비교 조건의 Slur-Free Prediction 비교 차트."""
    ensure_dir(output_path.parent)

    metrics = ["Slur-Free Accuracy", "Mean Prob Drop"]
    baseline_vals = [
        baseline_slur.get("slur_free_accuracy", 0) or 0,
        baseline_slur.get("mean_prob_drop", 0) or 0,
    ]
    improved_vals = [
        improved_slur.get("slur_free_accuracy", 0) or 0,
        improved_slur.get("mean_prob_drop", 0) or 0,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, baseline_vals, width, label=baseline_name, color="#5DADE2")
    ax.bar(x + width / 2, improved_vals, width, label=improved_name, color="#F39C12")

    ax.set_ylabel("Score")
    ax.set_title("Slur-Free Prediction: Context Understanding Ability")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_rationale_comparison(
    baseline_overlaps: list[dict[str, Any]],
    improved_overlaps: list[dict[str, Any]],
    baseline_name: str,
    improved_name: str,
    output_path: Path,
) -> None:
    """기준 조건과 비교 조건의 Human Rationale Overlap 분포를 비교하는 박스플롯."""
    ensure_dir(output_path.parent)

    rows = []
    for item in baseline_overlaps:
        rows.append({"model": baseline_name, "overlap": item["overlap"]})
    for item in improved_overlaps:
        rows.append({"model": improved_name, "overlap": item["overlap"]})

    if not rows:
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="overlap", by="model", ax=ax)
    ax.axhline(0.5, color="red", linestyle="--", label="Alignment threshold (0.5)")
    ax.set_title("Model Top-5 vs Human Rationale Overlap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Overlap (precision@5)")
    plt.suptitle("")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════╗
# ║  7. 분석 샘플 선정 전략                                   ║
# ╚══════════════════════════════════════════════════════════╝
#
# XAI 분석은 시간이 오래 걸리기 때문에 테스트셋 전체를 분석하긴 어려워요.
# 그래서 "가장 인사이트가 풍부한" 샘플을 전략적으로 골라야 합니다!
#
# 샘플 선정 우선순위:
#   1순위: fixed_error        - 기준 조건은 틀렸고 비교 조건은 맞힌 것
#   2순위: consistently_correct - 둘 다 맞힌 것 (안정적인 판단의 근거 확인)
#   3순위: model_disagreement  - 두 모델의 의견이 다른 것 (왜 다르게 봤을까?)
#   4순위: fallback_sample     - 나머지 (빈자리 채우기)
#
def _select_analysis_samples(
    texts: list[str],
    labels: np.ndarray,
    baseline_preds: np.ndarray,
    improved_preds: np.ndarray,
    sample_size: int,
    post_ids: list[str] | None = None,
) -> pd.DataFrame:
    # 카테고리별로 분류할 리스트들
    fixed_rows = []          # 오분류 --> 정분류 전환 (가장 가치 있는 샘플!)
    stable_rows = []         # 둘 다 정답
    disagreement_rows = []   # 두 모델이 서로 다르게 예측
    fallback_rows = []       # 그 외 나머지
    columns = [
        "index",
        "post_id",
        "text",
        "true_label",
        "true_label_name",
        "baseline_pred",
        "baseline_pred_name",
        "improved_pred",
        "improved_pred_name",
        "category",
    ]

    # post_ids가 없으면 빈 문자열로 채우기 (하위 호환)
    if post_ids is None:
        post_ids = [""] * len(texts)

    # 각 샘플을 분류하기
    for index, (text, label, baseline_pred, improved_pred, pid) in enumerate(
        zip(texts, labels, baseline_preds, improved_preds, post_ids)
    ):
        row = {
            "index": index,
            "post_id": pid,
            "text": text,
            "true_label": int(label),
            "true_label_name": LABEL_NAMES[int(label)],
            "baseline_pred": int(baseline_pred),
            "baseline_pred_name": LABEL_NAMES[int(baseline_pred)],
            "improved_pred": int(improved_pred),
            "improved_pred_name": LABEL_NAMES[int(improved_pred)],
        }
        if baseline_pred != label and improved_pred == label:
            row["category"] = "fixed_error"        # VADER 덕분에 고쳐진 케이스!
            fixed_rows.append(row)
        elif baseline_pred == label and improved_pred == label:
            row["category"] = "consistently_correct"  # 둘 다 잘 맞힘
            stable_rows.append(row)
        elif baseline_pred != improved_pred:
            row["category"] = "model_disagreement"    # 의견 불일치
            disagreement_rows.append(row)
        else:
            row["category"] = "fallback_sample"       # 기타
            fallback_rows.append(row)

    # 우선순위에 따라 sample_size만큼 채우기 (절반은 fixed_error로!)
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


# ╔══════════════════════════════════════════════════════════╗
# ║  8. 메인 파이프라인: run_xai()                            ║
# ║     여기가 이 파일의 심장부입니다!                           ║
# ╚══════════════════════════════════════════════════════════╝
#
# 이 함수 하나를 호출하면 XAI 분석의 모든 과정이 자동으로 진행됩니다.
# 마치 요리 레시피처럼 Step-by-Step으로 따라가 보세요!
#
def run_xai(config: ExperimentConfig | None = None) -> dict[str, Any]:
    """
    XAI 전체 파이프라인 실행: 기준 조건 vs 비교 조건의 v2.1 4축 사후 검증.

    흐름 (한눈에 보기):
      Step 1. 테스트셋 예측 (기준 조건 + 비교 조건)
      Step 2. 분석 대상 샘플 선정 (오분류->정분류 전환 우선)
      Step 3. SHAP + LIME 설명 생성 (각 모델별, 서브워드 aggregation 적용)
      Step 4. Overlap@5 계산 및 시각화 (Attribution 안정성)
      Step 5. Human Rationale 비교 (Plausibility 보조)
      Step 6. 마스킹 검증 -- Comprehensiveness/Sufficiency + Slur-Free (Faithfulness)
      Step 7. CI/MSS/LOO/IS/Attention Rollout 자동 메트릭 (Context Learning)
      Step 8. 케이스별 SHAP attribution 비교 차트 생성
      Step 9. xai_summary.json + xai_summary.md 저장

    반환값: 분석 요약이 담긴 dictionary (JSON 저장 후 반환)
    """

    # ── Step 0: 초기 설정 ──────────────────────────────────
    # 실험 설정 로드, 랜덤 시드 고정 (재현성!), 출력 디렉토리 준비
    config = config or get_config()
    set_seed(config.tuning_seed)
    ensure_dir(XAI_DIR)

    # ── Step 1: 테스트 데이터 로드 ─────────────────────────
    # 데이터 분할(train/val/test)이 아직 안 되어 있으면 에러!
    # experiment_data.py를 먼저 실행해야 해요.
    if not SPLITS_PICKLE_PATH.exists():
        raise FileNotFoundError("Data split artifact is missing. Run data preparation first.")

    splits = load_pickle(SPLITS_PICKLE_PATH)
    test_df = splits["test"]
    texts = test_df["text"].tolist()       # 테스트 텍스트 목록
    labels = test_df["label"].to_numpy()   # 정답 라벨 (0=hate, 1=offensive, 2=normal)
    post_ids = test_df["post_id"].tolist() if "post_id" in test_df.columns else [""] * len(texts)

    # ── Step 2: 두 모델로 테스트셋 전체 예측 ──────────────
    # 기준 조건(A_B 우선)과 비교 조건(D_B/D_R 우선)을 로드하고 테스트셋 전체에 대해 예측합니다.
    baseline_bundle, improved_bundle = load_bundles_for_xai()
    baseline_prob = predict_probabilities(baseline_bundle, texts)   # (N, 3) 확률 배열
    improved_prob = predict_probabilities(improved_bundle, texts)   # (N, 3) 확률 배열
    baseline_preds = baseline_prob.argmax(axis=1)  # 가장 높은 확률의 클래스 = 예측 라벨
    improved_preds = improved_prob.argmax(axis=1)

    # ── Step 3: 성능 지표 계산 및 혼동 행렬 시각화 ─────────
    # 두 모델의 정확도, F1, 혼동 행렬을 계산하고 그래프로 저장합니다.
    # 보고서의 "성능 비교" 섹션에 사용될 거예요!
    baseline_metrics = compute_metrics(labels, baseline_preds, baseline_prob)
    improved_metrics = compute_metrics(labels, improved_preds, improved_prob)

    # 각 모델별 결과 디렉토리 생성 및 혼동 행렬 저장
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

    # ── Step 4: 분석 대상 샘플 선정 ───────────────────────
    # 테스트셋 전체를 XAI로 분석하면 시간이 너무 오래 걸리니까,
    # 가장 인사이트가 풍부한 샘플만 골라서 분석합니다.
    # (fixed_error 우선! --> "VADER 추가로 이렇게 좋아졌다"를 보여줄 수 있으니까요)
    sample_frame = _select_analysis_samples(
        texts=texts,
        labels=labels,
        baseline_preds=baseline_preds,
        improved_preds=improved_preds,
        sample_size=config.xai_sample_size,
        post_ids=post_ids,
    )
    save_dataframe(sample_frame, XAI_DIR / "analysis_samples.csv")

    # ── Step 4-1: 빈 샘플 예외 처리 ──────────────────────
    # 만약 선정된 샘플이 하나도 없다면? (극히 드물지만 안전하게 처리!)
    # 빈 결과물을 만들고 일찍 반환합니다.
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
            "baseline_rationale_shap_mean": None,
            "improved_rationale_shap_mean": None,
            "baseline_rationale_lime_mean": None,
            "improved_rationale_lime_mean": None,
            "baseline_rationale_ge_50": 0,
            "improved_rationale_ge_50": 0,
            "rationale_sample_count": 0,
            "baseline_comprehensiveness": None,
            "improved_comprehensiveness": None,
            "baseline_sufficiency": None,
            "improved_sufficiency": None,
            "masking_sample_count": 0,
            "baseline_slur_free_accuracy": None,
            "improved_slur_free_accuracy": None,
            "baseline_slur_prob_drop": None,
            "improved_slur_prob_drop": None,
            "slur_free_sample_count": 0,
            "baseline_ci": None,
            "improved_ci": None,
            "baseline_mss": None,
            "improved_mss": None,
            "baseline_loo": None,
            "improved_loo": None,
            "baseline_interaction_strength": None,
            "improved_interaction_strength": None,
            "baseline_rollout_entropy": None,
            "improved_rollout_entropy": None,
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
                "baseline_rationale_shap": [],
                "improved_rationale_shap": [],
                "baseline_rationale_lime": [],
                "improved_rationale_lime": [],
            },
            XAI_DIR / "xai_details.json",
        )
        return summary

    # ── Step 5: SHAP + LIME 설명 생성 (가장 시간이 오래 걸리는 부분!) ──
    # 선정된 샘플에 대해 기준 조건과 비교 조건 각각에 SHAP, LIME을 적용합니다.
    # 총 4번의 XAI 분석이 수행돼요:
    #   1) 기준 조건 + SHAP    2) 기준 조건 + LIME
    #   3) 비교 조건 + SHAP    4) 비교 조건 + LIME
    sample_texts = sample_frame["text"].tolist()
    baseline_sample_preds = sample_frame["baseline_pred"].tolist()
    improved_sample_preds = sample_frame["improved_pred"].tolist()

    baseline_shap = run_shap_explanations(baseline_bundle, sample_texts, baseline_sample_preds, config)
    baseline_lime = run_lime_explanations(baseline_bundle, sample_texts, baseline_sample_preds, config)
    improved_shap = run_shap_explanations(improved_bundle, sample_texts, improved_sample_preds, config)
    improved_lime = run_lime_explanations(improved_bundle, sample_texts, improved_sample_preds, config)

    # ── Step 6: Overlap@5 계산 ────────────────────────────
    # SHAP과 LIME이 같은 토큰을 중요하다고 봤는지 체크!
    # 두 기법이 일치할수록(>=60%) 설명의 신뢰도가 높다고 볼 수 있어요.
    baseline_overlap = _compute_overlap_at_5(baseline_shap, baseline_lime)
    improved_overlap = _compute_overlap_at_5(improved_shap, improved_lime)

    # Overlap 결과를 DataFrame으로 정리하고 CSV + 박스플롯으로 저장
    overlap_rows = []
    for index, overlap in enumerate(baseline_overlap):
        overlap_rows.append({"model": baseline_bundle.display_name, "sample_id": int(sample_frame.iloc[index]["index"]), "overlap_at_5": overlap})
    for index, overlap in enumerate(improved_overlap):
        overlap_rows.append({"model": improved_bundle.display_name, "sample_id": int(sample_frame.iloc[index]["index"]), "overlap_at_5": overlap})
    overlap_frame = pd.DataFrame(overlap_rows)
    save_dataframe(overlap_frame, XAI_DIR / "overlap_at_5.csv")
    _plot_overlap_summary(overlap_rows, XAI_DIR / "overlap_at_5.png")

    # ── Step 6-1: Human Rationale 비교 (설명 타당성 평가) ──
    # 모델의 Top-5 토큰이 인간 annotator의 판단 근거(rationale)와 얼마나 일치하는지 측정합니다.
    # Overlap@5(안정성)와 독립적인 "설명 타당성" 지표예요.
    rationale_map = _load_human_rationales()
    sample_post_ids = sample_frame["post_id"].tolist()

    # SHAP 기준으로 기준 조건/비교 조건 각각의 rationale overlap 계산
    baseline_rat_shap = _compute_rationale_overlap(baseline_shap, sample_post_ids, rationale_map)
    improved_rat_shap = _compute_rationale_overlap(improved_shap, sample_post_ids, rationale_map)
    # LIME 기준으로도 동일하게 계산
    baseline_rat_lime = _compute_rationale_overlap(baseline_lime, sample_post_ids, rationale_map)
    improved_rat_lime = _compute_rationale_overlap(improved_lime, sample_post_ids, rationale_map)

    # 결과를 CSV로 저장
    rat_rows = []
    for item in baseline_rat_shap:
        rat_rows.append({"model": baseline_bundle.display_name, "xai_method": "SHAP", "post_id": item["post_id"],
                         "overlap": item["overlap"], "matched": ", ".join(item["matched_tokens"]),
                         "model_top5": ", ".join(item["model_top_tokens"]), "human_rationale_count": item["human_rationale_count"]})
    for item in improved_rat_shap:
        rat_rows.append({"model": improved_bundle.display_name, "xai_method": "SHAP", "post_id": item["post_id"],
                         "overlap": item["overlap"], "matched": ", ".join(item["matched_tokens"]),
                         "model_top5": ", ".join(item["model_top_tokens"]), "human_rationale_count": item["human_rationale_count"]})
    for item in baseline_rat_lime:
        rat_rows.append({"model": baseline_bundle.display_name, "xai_method": "LIME", "post_id": item["post_id"],
                         "overlap": item["overlap"], "matched": ", ".join(item["matched_tokens"]),
                         "model_top5": ", ".join(item["model_top_tokens"]), "human_rationale_count": item["human_rationale_count"]})
    for item in improved_rat_lime:
        rat_rows.append({"model": improved_bundle.display_name, "xai_method": "LIME", "post_id": item["post_id"],
                         "overlap": item["overlap"], "matched": ", ".join(item["matched_tokens"]),
                         "model_top5": ", ".join(item["model_top_tokens"]), "human_rationale_count": item["human_rationale_count"]})
    rationale_frame = pd.DataFrame(rat_rows)
    save_dataframe(rationale_frame, XAI_DIR / "rationale_overlap.csv")

    # SHAP 기준 비교 박스플롯 저장
    _plot_rationale_comparison(
        baseline_rat_shap, improved_rat_shap,
        baseline_bundle.display_name, improved_bundle.display_name,
        XAI_DIR / "rationale_overlap_shap.png",
    )
    # LIME 기준 비교 박스플롯 저장
    _plot_rationale_comparison(
        baseline_rat_lime, improved_rat_lime,
        baseline_bundle.display_name, improved_bundle.display_name,
        XAI_DIR / "rationale_overlap_lime.png",
    )

    # ── Step 6-2: 마스킹 기반 충실도 검증 (XAI 4축 중 Faithfulness) ──
    # SHAP/LIME이 "중요하다"고 지목한 토큰이 정말로 예측에 인과적 영향을 미치는지
    # 마스킹 실험으로 직접 검증합니다. 이 단계가 Faithfulness 축의 핵심이에요.
    #
    # Comprehensiveness: Top-5를 제거하면 예측이 얼마나 떨어지나?
    # Sufficiency: Top-5만 남기면 예측이 유지되나?
    # Slur-Free: 비하어를 빼도 맥락으로 혐오를 잡나?

    # SHAP Top-5 기준 마스킹 메트릭 (기준 조건)
    baseline_masking = _compute_masking_metrics(
        baseline_bundle, baseline_shap, sample_texts,
        baseline_sample_preds, k=5,
    )
    # SHAP Top-5 기준 마스킹 메트릭 (비교 조건)
    improved_masking = _compute_masking_metrics(
        improved_bundle, improved_shap, sample_texts,
        improved_sample_preds, k=5,
    )

    # 마스킹 메트릭 시각화
    _plot_masking_comparison(
        baseline_masking, improved_masking,
        baseline_bundle.display_name, improved_bundle.display_name,
        XAI_DIR / "masking_metrics.png",
    )

    # 마스킹 상세 결과 저장
    save_json(
        {"baseline": baseline_masking, "improved": improved_masking},
        XAI_DIR / "masking_metrics.json",
    )

    # Slur-Free Prediction 실험 (테스트셋 전체 대상으로 실행)
    # 분석 샘플만이 아니라 테스트셋 전체에서 비하어가 포함된 문장을 대상으로 합니다
    baseline_slur = _compute_slur_free_prediction(
        baseline_bundle, texts, labels, baseline_preds.tolist(),
    )
    improved_slur = _compute_slur_free_prediction(
        improved_bundle, texts, labels, improved_preds.tolist(),
    )

    # Slur-Free 시각화
    _plot_slur_free_comparison(
        baseline_slur, improved_slur,
        baseline_bundle.display_name, improved_bundle.display_name,
        XAI_DIR / "slur_free_prediction.png",
    )

    # Slur-Free 상세 결과 저장 (details는 대량이므로 별도 파일)
    save_json(
        {
            "baseline": {k: v for k, v in baseline_slur.items() if k != "details"},
            "improved": {k: v for k, v in improved_slur.items() if k != "details"},
        },
        XAI_DIR / "slur_free_summary.json",
    )

    # ── Step 6-3: v2.1 자동 XAI 4축 메트릭 ─────────────────
    # CI/MSS/LOO/IS/Attention Rollout은 인간 정의 카테고리에 기대지 않는 자동 검증 축입니다.
    baseline_v2_metrics = _compute_v2_xai_metrics(
        baseline_bundle, baseline_shap, sample_texts, baseline_sample_preds, config,
    )
    improved_v2_metrics = _compute_v2_xai_metrics(
        improved_bundle, improved_shap, sample_texts, improved_sample_preds, config,
    )
    save_json(
        {
            "baseline_model": baseline_bundle.display_name,
            "improved_model": improved_bundle.display_name,
            "baseline": baseline_v2_metrics,
            "improved": improved_v2_metrics,
        },
        XAI_DIR / "xai_4axis_metrics.json",
    )

    # ── Step 7: 케이스별 비교 차트 생성 ───────────────────
    # 최대 8개 샘플에 대해 기준 조건 vs 비교 조건의 SHAP Top-5 막대 그래프를 나란히 그려요.
    # "이 문장에서 각 조건이 어떤 단어를 봤는지"
    # 직관적으로 보여주는 핵심 시각화 자료입니다!
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
            baseline_name=baseline_bundle.display_name,
            improved_name=improved_bundle.display_name,
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

    # 케이스 요약 테이블 저장
    case_frame = pd.DataFrame(case_rows)
    save_dataframe(case_frame, XAI_DIR / "case_summary.csv")

    # ── Step 8: 최종 요약 생성 및 저장 ────────────────────
    # 분석 결과를 JSON, Markdown으로 정리해서 저장합니다.
    # 이 요약 데이터가 최종 보고서에 들어가는 핵심 수치들이에요!
    summary = {
        "baseline_model": baseline_bundle.display_name,
        "improved_model": improved_bundle.display_name,
        "baseline_macro_f1": baseline_metrics["macro_f1"],        # Baseline F1 점수
        "improved_macro_f1": improved_metrics["macro_f1"],        # Improved F1 점수
        "baseline_overlap_mean": float(np.mean(baseline_overlap)) if baseline_overlap else None,
        "improved_overlap_mean": float(np.mean(improved_overlap)) if improved_overlap else None,
        "baseline_overlap_ge_60": int(sum(value >= 0.6 for value in baseline_overlap)),  # 신뢰 임계값 넘은 샘플 수
        "improved_overlap_ge_60": int(sum(value >= 0.6 for value in improved_overlap)),
        "sample_count": int(len(sample_frame)),                   # 분석한 총 샘플 수
        "fixed_error_count": int((sample_frame["category"] == "fixed_error").sum()),  # 오분류 교정 수
        # ── Human Rationale 비교 메트릭 (축 1-2: 설명 타당성) ──
        "baseline_rationale_shap_mean": round(float(np.mean([r["overlap"] for r in baseline_rat_shap])), 4) if baseline_rat_shap else None,
        "improved_rationale_shap_mean": round(float(np.mean([r["overlap"] for r in improved_rat_shap])), 4) if improved_rat_shap else None,
        "baseline_rationale_lime_mean": round(float(np.mean([r["overlap"] for r in baseline_rat_lime])), 4) if baseline_rat_lime else None,
        "improved_rationale_lime_mean": round(float(np.mean([r["overlap"] for r in improved_rat_lime])), 4) if improved_rat_lime else None,
        "baseline_rationale_ge_50": int(sum(1 for r in baseline_rat_shap if r["overlap"] >= 0.5)),
        "improved_rationale_ge_50": int(sum(1 for r in improved_rat_shap if r["overlap"] >= 0.5)),
        "rationale_sample_count": len(baseline_rat_shap),  # rationale이 있는 샘플 수
        # ── 마스킹 검증 메트릭 (축 3: 설명 충실도) ──
        "baseline_comprehensiveness": baseline_masking.get("comprehensiveness_mean"),
        "improved_comprehensiveness": improved_masking.get("comprehensiveness_mean"),
        "baseline_sufficiency": baseline_masking.get("sufficiency_mean"),
        "improved_sufficiency": improved_masking.get("sufficiency_mean"),
        "masking_sample_count": baseline_masking.get("sample_count", 0),
        # ── Slur-Free Prediction (맥락 이해 능력) ──
        "baseline_slur_free_accuracy": baseline_slur.get("slur_free_accuracy"),
        "improved_slur_free_accuracy": improved_slur.get("slur_free_accuracy"),
        "baseline_slur_prob_drop": baseline_slur.get("mean_prob_drop"),
        "improved_slur_prob_drop": improved_slur.get("mean_prob_drop"),
        "slur_free_sample_count": baseline_slur.get("sample_count", 0),
        # ── v2.1 자동 XAI 4축 메트릭 ──
        "baseline_ci": baseline_v2_metrics["ci"]["mean"],
        "improved_ci": improved_v2_metrics["ci"]["mean"],
        "baseline_mss": baseline_v2_metrics["mss"]["mean"],
        "improved_mss": improved_v2_metrics["mss"]["mean"],
        "baseline_loo": baseline_v2_metrics["loo"]["mean"],
        "improved_loo": improved_v2_metrics["loo"]["mean"],
        "baseline_interaction_strength": baseline_v2_metrics["interaction_strength"]["mean"],
        "improved_interaction_strength": improved_v2_metrics["interaction_strength"]["mean"],
        "baseline_rollout_entropy": baseline_v2_metrics["attention_rollout_entropy"]["mean"],
        "improved_rollout_entropy": improved_v2_metrics["attention_rollout_entropy"]["mean"],
    }

    # JSON 요약 저장
    save_json(summary, XAI_DIR / "xai_summary.json")

    # Markdown 요약 저장 (보고서에 바로 복붙 가능!)
    rationale_md = ""
    if baseline_rat_shap or improved_rat_shap:
        rationale_md = (
            "\n\n## Human Rationale Alignment (축 1-2: 설명 타당성)\n\n"
            f"| 지표 | {baseline_bundle.display_name} | {improved_bundle.display_name} |\n"
            f"|------|:---:|:---:|\n"
            f"| SHAP Top-5 vs Human Rationale (mean) | {summary['baseline_rationale_shap_mean']} | {summary['improved_rationale_shap_mean']} |\n"
            f"| LIME Top-5 vs Human Rationale (mean) | {summary['baseline_rationale_lime_mean']} | {summary['improved_rationale_lime_mean']} |\n"
            f"| SHAP overlap >= 0.5 (count) | {summary['baseline_rationale_ge_50']} | {summary['improved_rationale_ge_50']} |\n"
            f"| Rationale 보유 샘플 수 | {summary['rationale_sample_count']} | {summary['rationale_sample_count']} |\n"
        )

    # 마스킹 검증 Markdown
    masking_md = (
        "\n\n## Masking Verification (축 3: 설명 충실도)\n\n"
        f"| 지표 | {baseline_bundle.display_name} | {improved_bundle.display_name} |\n"
        f"|------|:---:|:---:|\n"
        f"| Comprehensiveness (mean) | {summary['baseline_comprehensiveness']} | {summary['improved_comprehensiveness']} |\n"
        f"| Sufficiency (mean) | {summary['baseline_sufficiency']} | {summary['improved_sufficiency']} |\n"
        f"| 마스킹 검증 샘플 수 | {summary['masking_sample_count']} | {summary['masking_sample_count']} |\n"
    )

    # Slur-Free Prediction Markdown
    slur_md = (
        "\n\n## Slur-Free Prediction (맥락 이해 능력)\n\n"
        f"| 지표 | {baseline_bundle.display_name} | {improved_bundle.display_name} |\n"
        f"|------|:---:|:---:|\n"
        f"| Slur-Free Accuracy | {summary['baseline_slur_free_accuracy']} | {summary['improved_slur_free_accuracy']} |\n"
        f"| Mean Prob Drop | {summary['baseline_slur_prob_drop']} | {summary['improved_slur_prob_drop']} |\n"
        f"| 대상 샘플 수 | {summary['slur_free_sample_count']} | {summary['slur_free_sample_count']} |\n"
    )

    v2_md = (
        "\n\n## XAI 4-Axis Automatic Metrics (v2.1)\n\n"
        f"| 지표 | {baseline_bundle.display_name} | {improved_bundle.display_name} |\n"
        f"|------|:---:|:---:|\n"
        f"| CI (Concentration Index) | {summary['baseline_ci']} | {summary['improved_ci']} |\n"
        f"| MSS | {summary['baseline_mss']} | {summary['improved_mss']} |\n"
        f"| LOO mean drop | {summary['baseline_loo']} | {summary['improved_loo']} |\n"
        f"| Interaction Strength | {summary['baseline_interaction_strength']} | {summary['improved_interaction_strength']} |\n"
        f"| Attention Rollout Entropy | {summary['baseline_rollout_entropy']} | {summary['improved_rollout_entropy']} |\n"
    )

    save_text(
        "# XAI Summary (v2.1 4축 검증 프레임워크)\n\n"
        + dataframe_to_markdown(pd.DataFrame([summary]))
        + "\n\n## Case Summary\n\n"
        + dataframe_to_markdown(case_frame)
        + rationale_md
        + masking_md
        + slur_md
        + v2_md,
        XAI_DIR / "xai_summary.md",
    )

    # SHAP/LIME 상세 결과 저장 (나중에 추가 분석할 때 유용해요!)
    save_json(
        {
            "baseline_shap": baseline_shap,
            "baseline_lime": baseline_lime,
            "improved_shap": improved_shap,
            "improved_lime": improved_lime,
            "baseline_rationale_shap": baseline_rat_shap,
            "improved_rationale_shap": improved_rat_shap,
            "baseline_rationale_lime": baseline_rat_lime,
            "improved_rationale_lime": improved_rat_lime,
            "baseline_v2_metrics": baseline_v2_metrics,
            "improved_v2_metrics": improved_v2_metrics,
        },
        XAI_DIR / "xai_details.json",
    )

    # 수고하셨습니다! v2.1 4축 검증 프레임워크 분석 요약을 반환합니다.
    return summary
