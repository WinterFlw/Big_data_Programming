"""
설정 모듈 (src/config.py)
==========================
ExperimentConfig 데이터클래스 + config.yaml 로딩 로직.
ucam은 dict 기반(config.yaml -> config.get)이지만, 이 프로젝트는 dataclass
기반(config.attribute)이라 코드 변경을 최소화하기 위해 config.yaml을 기본값
출처로 삼아 ExperimentConfig를 구성하는 방식으로 ucam 원칙을 적용했어요.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import yaml

from src.path import CONFIG_PATH, SRC_PATH
from src.utils import save_json


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
    # HateXplain에서 offensive(28.5%)가 normal(40.6%) 대비 중간 불균형이에요.
    # threshold를 0.40으로 설정해서 balanced class weight가 적용되도록 합니다.
    imbalance_threshold: float = 0.40

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
    # v2.1 명세서: 메인 SHAP/LIME 분석은 클래스별 stratified 500 샘플 / 조건
    xai_sample_size: int = 500    # SHAP/LIME 분석할 샘플 수 (메인 비교 A_B vs D_B)
    lime_num_features: int = 5    # LIME이 보여줄 중요 단어 Top-K개
    lime_num_samples: int = 500   # LIME이 텍스트를 얼마나 변형해볼지
    shap_max_evals: int = 300     # SHAP 최대 평가 횟수
    shap_batch_size: int = 32     # SHAP 배치 크기

    # ── v2.1 ablation 설정 ─────────────────────
    # 명세서 v2.1은 BERT/RoBERTa × Attn/VADER 8조건을 기본 단위로 삼습니다.
    v2_enabled: bool = True
    alpha_grid: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    beta_grid: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3])
    attention_loss_alpha: float = 0.0
    target_loss_beta: float = 0.0
    target_labels: list[str] = field(default_factory=list)
    # 메인 비교용 자동 메트릭 샘플 수 (A_B vs D_B Context Learning 축)
    xai_context_sample_size: int = 500
    # 8조건 풀 ablation 매트릭스 전용 (시간 절약을 위해 메인보다 작게)
    # 메인은 명세서 부합 500, 매트릭스는 reduced 50으로 운영하고 한계 명시
    xai_ablation_sample_size: int = 50
    xai_interaction_pairs: int = 50
    xai_mss_threshold: float = 0.8


# ── config.yaml에서 기본값 로드 ──────────────────
def _load_yaml_defaults() -> dict:
    """src/config.yaml을 읽어 ExperimentConfig 필드에 해당하는 값만 추려요."""
    yaml_path = SRC_PATH / "config.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    valid_keys = set(ExperimentConfig.__dataclass_fields__.keys())
    return {key: value for key, value in data.items() if key in valid_keys}


# 기본 설정 인스턴스 (config.yaml 우선, 없으면 dataclass 기본값)
DEFAULT_CONFIG = ExperimentConfig(**_load_yaml_defaults())


def get_config() -> ExperimentConfig:
    """저장된 스냅샷이 있으면 불러오고, 없으면 기본값을 저장 후 반환해요."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        valid_keys = set(ExperimentConfig.__dataclass_fields__.keys())
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        config = ExperimentConfig(**filtered)
        if set(filtered.keys()) != set(data.keys()):
            save_config(config)
        return config
    save_json(asdict(DEFAULT_CONFIG), CONFIG_PATH)
    return DEFAULT_CONFIG


def save_config(config: ExperimentConfig) -> None:
    """현재 실험 설정을 JSON 스냅샷으로 저장해요. 재현성의 첫걸음!"""
    save_json(asdict(config), CONFIG_PATH)
