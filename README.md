# 맥락 이해 기반 혐오표현 탐지

HateXplain 데이터셋을 기반으로 혐오표현 탐지 모델의 성능, 재현성, 설명 가능성(XAI)을 함께 검증한 빅데이터프로그래밍 산학협력 프로젝트입니다.

핵심 아이디어는 단순히 욕설이나 특정 키워드에 반응하는 모델이 아니라, 사람이 혐오 판단의 근거로 표시한 토큰 단위 근거 주석을 학습 과정에 반영해 모델의 어텐션을 더 설명 가능한 방향으로 유도하는 것입니다.

## 프로젝트 요약

| 항목 | 내용 |
|---|---|
| 상태 | 프로젝트 완료 |
| 과제 | 3분류 혐오표현 탐지 |
| 데이터셋 | HateXplain |
| 분류 라벨 | `hatespeech`, `offensive`, `normal` |
| 주요 모델 | BERT, RoBERTa |
| 제안 요소 | 근거 주석 기반 어텐션 손실 |
| 보조 피처 | VADER 감성 점수 |
| 실험 설계 | 백본 x 어텐션 손실 x VADER, 총 8조건 |
| 반복 실험 | 15개 학습 시드, 총 120회 실행 |
| 주 평가 지표 | Macro F1 |
| XAI 기법 | SHAP, LIME, 근거 정렬 분석, 마스킹 기반 검증 |

## 최종 결과

본 실험에서 가장 높은 성능을 보인 조건은 `B_B: BERT + 근거 주석 기반 어텐션 손실`입니다.

| 조건 | 백본 | 어텐션 손실 | VADER | Macro F1 평균 | Weighted F1 평균 | 해석 |
|---|---|---:|---:|---:|---:|---|
| A_B | BERT | 없음 | 없음 | 0.6798 | 0.6876 | BERT 기준 모델 |
| B_B | BERT | 있음 | 없음 | 0.6858 | 0.6935 | 본 실험 내 최고 성능 조건 |
| C_B | BERT | 없음 | 있음 | 0.6825 | 0.6903 | 감성 피처만 추가 |
| D_B | BERT | 있음 | 있음 | 0.6836 | 0.6910 | 어텐션 손실과 감성 피처 동시 적용 |
| A_R | RoBERTa | 없음 | 없음 | 0.6653 | 0.6723 | RoBERTa 기준 모델 |
| B_R | RoBERTa | 있음 | 없음 | 0.6763 | 0.6836 | RoBERTa 계열 최고 성능 조건 |
| C_R | RoBERTa | 없음 | 있음 | 0.6698 | 0.6777 | 감성 피처만 추가 |
| D_R | RoBERTa | 있음 | 있음 | 0.6743 | 0.6813 | 어텐션 손실과 감성 피처 동시 적용 |

## 통계 분석 요약

| 질문 | 결과 | 해석 |
|---|---|---|
| B_B가 A_B보다 개선되었는가? | 평균 차이 +0.0060, Holm 보정 p = 0.0027 | 작지만 통계적으로 유의한 개선 |
| 어텐션 손실은 의미가 있었는가? | 3요인 ANOVA p ~= 6.2e-06, eta squared ~= 0.0955 | 유의한 성능 요인 |
| VADER는 의미가 있었는가? | 3요인 ANOVA p ~= 0.5248, eta squared ~= 0.0017 | 독립 효과가 약함 |
| 성능 차이를 가장 크게 설명한 요인은 무엇인가? | 백본 eta squared ~= 0.3896 | 백본 선택이 가장 큰 요인 |

이 결과는 외부 전체 연구 기준의 SOTA 달성을 의미하지 않습니다. 본 저장소의 통제된 8조건 실험 안에서 `B_B`가 가장 좋은 후보였다는 의미입니다.

## 기준 모델 대비 변경점

기준 모델은 텍스트 인코더 출력만 사용해 분류합니다. 본 프로젝트의 개선 모델은 사람이 표시한 근거 주석 정보를 학습 신호로 추가합니다.

```text
텍스트
-> BERT / RoBERTa 인코더
-> 선택적 VADER 감성 피처 결합
-> MLP 분류기
-> 교차 엔트로피 손실
   + 선택적 근거 주석 기반 어텐션 손실
```

근거 주석 기반 어텐션 손실은 모델이 혐오, 공격적 표현, 정상 표현을 판단할 때 사람이 중요하다고 표시한 토큰에 더 정렬되도록 유도합니다.

## XAI 분석

XAI는 최종 성능 주장의 핵심 근거가 아니라, 모델 판단 근거를 해석하기 위한 사후 검증 계층으로 사용했습니다.

주요 관찰 내용은 다음과 같습니다.

- SHAP과 LIME으로 모델이 중요하게 본 토큰을 확인했습니다.
- top-k 토큰과 사람 근거 주석의 겹침을 통해 근거 정렬 정도를 확인했습니다.
- 중요한 토큰을 마스킹하거나 제거했을 때 예측이 어떻게 변하는지 충실성 관점에서 확인했습니다.
- 일부 `A_B`와 `D_B` 대표 체크포인트 비교에서는 설명 안정성이 개선되는 경향이 보였습니다.
- 다만 XAI는 전체 체크포인트에 대해 수행한 완전 검증이 아니므로, 정량 성능 결론과 분리해서 해석해야 합니다.

구분해야 할 점은 다음과 같습니다.

| 항목 | 조건 |
|---|---|
| 최종 성능 모델 | `B_B` |
| 대표 XAI 비교축 | `A_B` vs `D_B` |

계산 자원과 시간 제약으로 XAI는 대표 체크포인트 중심으로 수행했습니다. 따라서 XAI 결과를 `B_B` 최종 성능 향상의 직접 증거처럼 과장하지 않습니다.

## 팀원 기여 매핑

이 영역은 팀장 1명과 팀원 4명의 담당 업무를 최종 산출물과 연결하기 위한 공개용 등록 칸입니다. GitHub README에는 전화번호, 이메일, 학번 등 개인정보를 넣지 않습니다.

| 구분 | 이름 | 역할 이름 | 주요 기여 | 관련 파일 / 산출물 | 확인 근거 |
|---|---|---|---|---|---|
| 팀장 | 입력 예정 | 프로젝트 총괄 / 통합 담당 | 파이프라인 설계, 실험 통합, 최종 검수 | `v2/run.sh`, `v2/pipeline/`, `README.md` | 엔드투엔드 실행 로그, 최종 보고서, 커밋 이력 |
| 팀원 1 | 입력 예정 | 데이터 / 전처리 담당 | 데이터셋 준비, 라벨 매핑, 데이터 분할 검증 | `v2/runtime/`, `v2/configs/v2_15seed.json` | 데이터 분할 정책, 전처리 기록 |
| 팀원 2 | 입력 예정 | 모델 학습 / 실험 관리 담당 | BERT/RoBERTa 학습, 시드별 실행 관리, 체크포인트 처리 | `v2/runtime/experiment_core.py`, `v2/pipeline/training_adapter.py` | 완료 실행 기록, 벤치마크 CSV |
| 팀원 3 | 입력 예정 | 통계 / 결과 분석 담당 | Macro F1 비교, paired test, Holm 보정, ANOVA 해석 | `v2/pipeline/statistics.py`, `v2/outputs/experiments/v2_15seed/benchmark/` | 성능 요약표, 통계 결과표 |
| 팀원 4 | 입력 예정 | XAI / 보고서 담당 | SHAP/LIME 해석, 근거 정렬 분석, 보고서와 대시보드 자료 | `v2/pipeline/xai*.py`, `v2/docs/04_xai_protocol.md`, `outputs/` | XAI 시각화, 보고서 섹션 |

## 저장소 구조

```text
.
├── README.md
├── hatespeech/                  # 모듈화된 코드 스냅샷
├── v1/                          # 1차 파이프라인과 과거 산출물 보관
├── v2/                          # 최종 실험 기준 작업 공간
│   ├── configs/                 # v2_15seed 실험 설정
│   ├── docs/                    # 모델, 파이프라인, 통계, XAI, 실행 문서
│   ├── pipeline/                # 엔드투엔드 실행 제어, 실행 명세, 통계, 보고서 생성
│   ├── runtime/                 # 학습, 추론, 대시보드, XAI 실행 코드
│   ├── scripts/                 # 서버 실행, 백업, gate check 유틸리티
│   └── outputs/                 # 실험 산출물
└── outputs/                     # 선택적 로컬 분석 / 시각화 산출물
```

최종 기준 구현은 [`v2/`](v2/) 아래에 있습니다. `v1/`은 과거 실험과 프로젝트 진행 기록을 보관하기 위한 아카이브입니다.

## 실험 재현 방법

전체 실험은 CUDA GPU 환경을 권장합니다. 최종 실험은 클라우드 GPU 환경에서 8조건 x 15시드, 총 120회 실행으로 수행했습니다.

```bash
git clone https://github.com/WinterFlw/Big_data_Programming.git
cd Big_data_Programming

python -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt

cd v2
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
```

전체 120회 벤치마크를 다시 실행하려면 다음 순서를 사용합니다.

```bash
cd v2
./run.sh e2e benchmark --run-id v2_15seed --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-primary --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

대용량 체크포인트와 일부 생성 산출물은 버전 관리에서 제외했습니다. 파이프라인을 다시 실행하면 표준 CSV, JSON, 보고서, 대시보드 산출물이 `v2/outputs/experiments/v2_15seed/` 아래에 생성됩니다.

## 주요 문서

| 문서 | 용도 |
|---|---|
| [`v2/docs/01_model_definition.md`](v2/docs/01_model_definition.md) | 모델 정의와 조건명 설명 |
| [`v2/docs/02_e2e_pipeline.md`](v2/docs/02_e2e_pipeline.md) | 엔드투엔드 파이프라인 개요 |
| [`v2/docs/03_validation_and_statistics.md`](v2/docs/03_validation_and_statistics.md) | 통계 검증 계획 |
| [`v2/docs/04_xai_protocol.md`](v2/docs/04_xai_protocol.md) | XAI 프로토콜과 해석 기준 |
| [`v2/docs/06_execution_runbook.md`](v2/docs/06_execution_runbook.md) | 서버와 GPU 실행 가이드 |
| [`v2/configs/v2_15seed.json`](v2/configs/v2_15seed.json) | 최종 실험 설정 |

## 한계

- 성능 향상은 통계적으로 유의하지만 수치상으로는 작습니다.
- VADER 감성 점수는 강한 독립 개선 효과를 보이지 못했습니다.
- XAI 분석은 계산 자원 제약으로 대표 체크포인트 중심으로 수행했습니다.
- HateXplain 데이터셋 분포 안에서의 결론이므로 다른 데이터셋으로 일반화하려면 추가 검증이 필요합니다.
- 대용량 모델 체크포인트는 저장소에 포함하지 않았기 때문에 체크포인트 단위 재현은 학습 재실행이 필요합니다.

## 결론

근거 주석 기반 어텐션 학습은 본 통제 실험에서 BERT 기준 모델 대비 작지만 일관된 개선을 보였습니다. 이 프로젝트는 혐오표현 탐지에서 성능, 통계적 신뢰성, 설명 가능성을 함께 비교하는 재현 가능한 실험 프레임워크로 활용할 수 있습니다.
