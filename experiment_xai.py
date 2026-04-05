# ╔══════════════════════════════════════════════════════════╗
# ║  experiment_xai.py                                      ║
# ║  XAI (설명 가능한 AI) 분석 파이프라인                       ║
# ╚══════════════════════════════════════════════════════════╝
"""
XAI (설명 가능한 AI) 분석 파이프라인.

안녕하세요! 이 파일은 우리 혐오표현 탐지 연구에서 가장 흥미로운 부분이에요.
단순히 "모델이 맞았다/틀렸다"를 넘어서, "왜 그렇게 판단했는지"를 들여다보는 거죠.
XAI(eXplainable AI)를 통해 모델의 판단 근거를 시각적으로 확인할 수 있답니다!

연구의 핵심 흐름:
  Phase 1: BERT baseline에 SHAP + LIME 적용 --> 오분류 원인 진단
           ("이 단어 때문에 틀렸구나!" 를 확인하는 단계)
  Phase 2: 개선 모델(+VADER)에 동일 분석 --> Before/After 비교
           ("VADER 감성 점수를 추가하니까 판단이 어떻게 바뀌었지?" 를 비교)

분석 산출물 (이 파이프라인이 만들어내는 결과물들):
  - Overlap@5: SHAP Top-5 와 LIME Top-5의 일치도 (>=60%이면 높은 신뢰!)
    --> 두 XAI 기법이 같은 토큰을 중요하다고 보면 신뢰할 수 있겠죠?
  - 케이스 비교: 오분류에서 정분류로 전환된 샘플의 SHAP attribution 변화
    --> "아, 개선 모델은 이 단어에 더 주목했구나!" 같은 인사이트를 줍니다
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
    SPLITS_PICKLE_PATH,            # train/val/test 분할 데이터 경로
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


# 개선 모델 후보 목록: BERT+VADER와 RoBERTa+VADER 중에서 더 좋은 걸 고를 거예요!
IMPROVED_MODEL_NAMES = ["BERT+VADER", "RoBERTa+VADER"]


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
# BERT+VADER와 RoBERTa+VADER 중에서 누가 더 잘했는지 골라주는 함수!
# 마치 기말고사 성적표를 보고 1등을 뽑는 것과 같아요.
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


# ── Baseline + Improved 모델 동시 로드 ─────────────────────
# XAI 분석의 출발점! Baseline(BERT)과 개선 모델을 한 번에 준비해줍니다.
# 이 함수 하나로 두 모델 번들이 준비되니 참 편하죠?
def load_bundles_for_xai() -> tuple[LoadedModelBundle, LoadedModelBundle]:
    """Load the baseline and improved model used in the XAI report."""
    registry = _load_best_registry()
    baseline = _instantiate_bundle("BERT-base", registry["BERT-base"])
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

    # Step 3: 각 샘플별로 Top-5 중요 토큰 추출
    results = []
    special_tokens = set(bundle.tokenizer.all_special_tokens)  # [CLS], [SEP] 등 제외용
    for index, text in enumerate(texts):
        # SHAP이 반환한 토큰 데이터 가져오기 (문자열일 수도, 리스트일 수도 있어요)
        raw_tokens = shap_values.data[index]
        tokens = [str(token) for token in raw_tokens] if not isinstance(raw_tokens, str) else bundle.tokenizer.tokenize(raw_tokens)

        # 예측 라벨에 해당하는 SHAP 점수만 추출
        token_scores = _extract_shap_scores(np.asarray(shap_values.values[index]), tokens, predicted_labels[index])

        # 특수 토큰 제거 후, 절대값 기준으로 정렬 (양수든 음수든 영향이 큰 순서!)
        token_pairs = []
        for token, score in zip(tokens, token_scores):
            normalized = _normalize_token(token)
            if not normalized or token in special_tokens:
                continue  # [CLS] 같은 특수 토큰이나 빈 토큰은 건너뛰기
            token_pairs.append({"token": token, "score": float(score), "abs_score": float(abs(score))})
        token_pairs = sorted(token_pairs, key=lambda item: item["abs_score"], reverse=True)

        # Top-5 토큰과 점수를 결과에 담기
        results.append(
            {
                "text": text,
                "top_tokens": [item["token"] for item in token_pairs[:5]],
                "top_scores": [item["score"] for item in token_pairs[:5]],
                "token_details": token_pairs,  # 전체 토큰 정보도 보관 (나중에 쓸 수 있으니까!)
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
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
        )

        # 예측 라벨에 대한 피처 가중치를 추출 (어떤 단어가 얼마나 기여했는지!)
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
        overlaps.append(len(matched_shap_tokens & lime_top) / 5.0)
    return overlaps


# ── Overlap@5 박스플롯 시각화 ──────────────────────────────
# Baseline과 Improved 모델의 Overlap@5 분포를 나란히 비교하는 차트!
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
# Baseline과 Improved 모델의 SHAP Top-5를 나란히 보여주는 가로 막대 그래프!
# "개선 전에는 어떤 단어를 봤고, 개선 후에는 어떤 단어를 보게 됐는지" 한눈에 비교 가능해요.
def _plot_case_comparison(
    case_index: int,
    text: str,
    baseline_result: dict[str, Any],
    improved_result: dict[str, Any],
    output_path: Path,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 좌: Baseline, 우: Improved
    for axis, title, result in [
        (axes[0], "BERT-base SHAP", baseline_result),
        (axes[1], "RoBERTa+VADER SHAP", improved_result),
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
# ║  7. 분석 샘플 선정 전략                                   ║
# ╚══════════════════════════════════════════════════════════╝
#
# XAI 분석은 시간이 오래 걸리기 때문에 테스트셋 전체를 분석하긴 어려워요.
# 그래서 "가장 인사이트가 풍부한" 샘플을 전략적으로 골라야 합니다!
#
# 샘플 선정 우선순위:
#   1순위: fixed_error        - Baseline은 틀렸는데 Improved는 맞힌 것 (가장 흥미로운!)
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
) -> pd.DataFrame:
    # 카테고리별로 분류할 리스트들
    fixed_rows = []          # 오분류 --> 정분류 전환 (가장 가치 있는 샘플!)
    stable_rows = []         # 둘 다 정답
    disagreement_rows = []   # 두 모델이 서로 다르게 예측
    fallback_rows = []       # 그 외 나머지
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

    # 각 샘플을 분류하기
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
    XAI 전체 파이프라인 실행: Baseline vs 개선 모델의 SHAP/LIME 비교.

    흐름 (한눈에 보기):
      Step 1. 테스트셋 예측 (baseline + improved)
      Step 2. 분석 대상 샘플 선정 (오분류->정분류 전환 우선)
      Step 3. SHAP + LIME 설명 생성 (각 모델별)
      Step 4. Overlap@5 계산 및 시각화
      Step 5. 케이스별 SHAP attribution 비교 차트 생성
      Step 6. xai_summary.json + xai_summary.md 저장

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

    # ── Step 2: 두 모델로 테스트셋 전체 예측 ──────────────
    # Baseline(BERT-base)과 Improved(BERT+VADER 또는 RoBERTa+VADER) 모델을 로드하고,
    # 테스트셋 전체에 대해 확률 예측을 수행합니다.
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

    # ── Step 5: SHAP + LIME 설명 생성 (가장 시간이 오래 걸리는 부분!) ──
    # 선정된 샘플에 대해 Baseline과 Improved 모델 각각에 SHAP, LIME을 적용합니다.
    # 총 4번의 XAI 분석이 수행돼요:
    #   1) Baseline + SHAP    2) Baseline + LIME
    #   3) Improved + SHAP    4) Improved + LIME
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

    # ── Step 7: 케이스별 비교 차트 생성 ───────────────────
    # 최대 8개 샘플에 대해 Baseline vs Improved의 SHAP Top-5 막대 그래프를 나란히 그려요.
    # "이 문장에서 Baseline은 어떤 단어를 봤고, Improved는 어떤 단어를 봤는지"
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
    }

    # JSON 요약 저장
    save_json(summary, XAI_DIR / "xai_summary.json")

    # Markdown 요약 저장 (보고서에 바로 복붙 가능!)
    save_text(
        "# XAI Summary\n\n"
        + dataframe_to_markdown(pd.DataFrame([summary]))
        + "\n\n## Case Summary\n\n"
        + dataframe_to_markdown(case_frame),
        XAI_DIR / "xai_summary.md",
    )

    # SHAP/LIME 상세 결과 저장 (나중에 추가 분석할 때 유용해요!)
    save_json(
        {
            "baseline_shap": baseline_shap,
            "baseline_lime": baseline_lime,
            "improved_shap": improved_shap,
            "improved_lime": improved_lime,
        },
        XAI_DIR / "xai_details.json",
    )

    # 수고하셨습니다! 분석 요약을 반환합니다.
    return summary
