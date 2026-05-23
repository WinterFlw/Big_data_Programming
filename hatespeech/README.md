# HateSpeech XAI Pipeline (모듈화 버전)

HateXplain 기반 혐오표현 탐지 + XAI(SHAP/LIME) 파이프라인을 **ucam 구조**로
모듈화한 버전입니다. 원본(`run_experiments.py` + `experiment_*.py` 6개 모듈)을
`main.py` + `src/` + `model/` 구조로 재편했어요.

## 디렉토리 구조

```
hatespeech/
├── main.py              # 진입점(오케스트레이션). 원본 run_experiments.py
├── requirements.txt
├── run.sh               # 셸 실행 헬퍼 (zsh/bash)
├── data/
│   ├── raw/             # 원본 데이터 (dataset.json, post_id_divisions.json)
│   └── processed/       # 전처리·분할 결과 (data_splits.pkl, vader_features.pkl)
├── src/
│   ├── config.yaml      # 모든 설정의 단일 출처 (하이퍼파라미터 등)
│   ├── config.py        # ExperimentConfig + config.yaml 로딩
│   ├── path.py          # 모든 경로 상수 (중앙 관리)
│   ├── utils.py         # 공통 유틸 (시드/IO/메트릭/시각화/통계)
│   ├── data.py          # 데이터 준비 + PyTorch Dataset
│   ├── vader.py         # VADER 감성 피처 추출기
│   ├── eda.py           # 탐색적 데이터 분석
│   ├── xai.py           # SHAP/LIME 설명
│   ├── dashboard.py     # HTML 대시보드 생성
│   └── dashboard_app.py # (독립 실행형 대시보드 뷰어, 파이프라인과 분리)
└── model/
    ├── models.py        # 분류기 정의 (Transformer/MLP/Hybrid/Condition)
    └── train.py         # 학습/평가/벤치마크/튜닝/freeze/상태점검
```

### 원본 → 모듈화 매핑
| 원본 | 새 위치 |
|---|---|
| `run_experiments.py` | `main.py` |
| `experiment_core.py` (ExperimentConfig) | `src/config.py` + `src/config.yaml` |
| `experiment_core.py` (경로 상수) | `src/path.py` |
| `experiment_core.py` (데이터/Dataset) | `src/data.py` |
| `experiment_core.py` (VADER) | `src/vader.py` |
| `experiment_core.py` (모델 클래스) | `model/models.py` |
| `experiment_core.py` (학습/벤치마크/상태) | `model/train.py` |
| `utils.py` | `src/utils.py` (경로는 `src/path.py`로 분리) |
| `experiment_eda/xai/dashboard.py` | `src/eda.py`, `src/xai.py`, `src/dashboard.py` |

## 실행 방법

프로젝트 루트(`hatespeech/`)에서 실행해야 해요 (`from src...`/`from model...` 임포트 기준).

```bash
python main.py status        # 파이프라인 상태 확인
python main.py data          # 1. 데이터 전처리 → data/processed/
python main.py vader         # 2. VADER 피처 추출
python main.py eda           # 3. 탐색적 데이터 분석
python main.py tune          # 4. 하이퍼파라미터 튜닝
python main.py benchmark     # 5. 8조건 벤치마크 (BERT/RoBERTa 학습)
python main.py freeze-study  # 6. encoder 동결 비교
python main.py xai           # 7. SHAP/LIME 설명
python main.py dashboard     # 8. HTML 대시보드
python main.py all           # 전체 순차 실행
```

### Windows 사용자 주의
- 한글 콘솔(cp949)에서 출력 인코딩 문제가 있을 수 있어 `main.py`가 시작 시
  표준출력을 UTF-8로 자동 전환합니다. 그래도 문제가 나면 콘솔에서
  `chcp 65001` 또는 환경변수 `PYTHONIOENCODING=utf-8`을 설정하세요.
- `benchmark`/`xai`는 BERT/RoBERTa를 실제 학습하므로 GPU 없이는 매우 오래 걸려요.

## 설정 변경
`src/config.yaml`에서 하이퍼파라미터(학습률, 배치, epoch, 분할 비율 등)를 바꿀 수
있어요. (단, `outputs/experiment_config.json` 스냅샷이 있으면 그게 우선 적용되니,
기본값을 바꾸려면 해당 스냅샷을 지우거나 같이 수정하세요.)
