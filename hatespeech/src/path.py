"""
경로 정의 모듈 (src/path.py)
==============================

프로젝트의 모든 입출력 경로를 이 파일 한 곳에서 관리해요.
ucam 구조의 `src/path.py`와 동일한 역할로, 경로를 코드에 하드코딩하지 않고
여기서만 정의한 뒤 다른 모듈이 import해서 씁니다.

디렉토리 구조:
  data/raw/        원본 데이터 (dataset.json, post_id_divisions.json)
  data/processed/  전처리·분할 결과 (data_splits.pkl, vader_features.pkl)
  outputs/         실험 산출물 (reports, runs, tuning, xai)
  checkpoints/     학습된 모델 가중치 (.pt)
"""

from pathlib import Path

# ── 프로젝트 루트 ────────────────────────────────
# 이 파일은 src/ 안에 있으므로, 부모의 부모가 프로젝트 루트예요.
BASE_DIR = Path(__file__).resolve().parent.parent

# ── 코드 경로 ────────────────────────────────────
SRC_PATH = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "model"

# ── 데이터 경로 ──────────────────────────────────
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw"              # 원본 데이터
PROCESSED_PATH = DATA_DIR / "processed"  # 전처리·분할 결과

# ── 산출물 경로 ──────────────────────────────────
OUTPUT_DIR = BASE_DIR / "outputs"          # 모든 실험 결과의 최상위 폴더
CHECKPOINT_DIR = BASE_DIR / "checkpoints"  # 학습된 모델 가중치 (.pt 파일)
REPORT_DIR = OUTPUT_DIR / "reports"        # 보고서용 표/그래프
RUNS_DIR = OUTPUT_DIR / "runs"             # seed별 반복 실험 기록
TUNING_DIR = OUTPUT_DIR / "tuning"         # 하이퍼파라미터 탐색 기록
XAI_DIR = OUTPUT_DIR / "xai"               # SHAP/LIME 분석 결과

# ── 원본 데이터 파일 ─────────────────────────────
RAW_DATASET_PATH = RAW_PATH / "dataset.json"            # 원본 HateXplain 데이터
RAW_SPLIT_PATH = RAW_PATH / "post_id_divisions.json"    # 원본 split 정보

# ── 전처리 산출물 파일 ───────────────────────────
SPLITS_PICKLE_PATH = PROCESSED_PATH / "data_splits.pkl"   # 전처리된 train/val/test 분할
VADER_PICKLE_PATH = PROCESSED_PATH / "vader_features.pkl"  # VADER 감성 피처 캐시

# ── 설정/리포트 파일 ─────────────────────────────
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

# ── 디렉토리 자동 생성 ───────────────────────────
# 필요한 디렉토리가 없으면 처음 실행 시 자동으로 만들어줘요.
for _directory in [DATA_DIR, RAW_PATH, PROCESSED_PATH, OUTPUT_DIR,
                   CHECKPOINT_DIR, REPORT_DIR, RUNS_DIR, TUNING_DIR, XAI_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)
