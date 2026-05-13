# CLAUDE.md — HateSpeachStudy 하네스 규칙 (v2.1 기준)

> 이 파일은 AI 에이전트가 프로젝트에서 작업할 때 반드시 따라야 하는 **이밸류에이터(Evaluator)** 역할의 규칙서입니다.
> 모든 작업 시작 전 이 파일을 읽고, 작업 후 `progress.md`를 업데이트하세요.
>
> **단일 출처 명세서**: `docs/파이프라인_명세서_v2.md` (모든 코드·문서 작업의 단일 진실 출처)
> **참고문헌**: `docs/참고문헌_v2.md` (12편)
> **발표 와꾸**: `docs/발표_와꾸_v2.md` (26p)

---

## 하네스 시스템 개요

이 프로젝트는 **하네스 엔지니어링** 원칙에 따라 AI 에이전트의 작업을 제어합니다.

```
[Planner]          progress.md의 Todo를 읽고 작업 계획 수립
    │
    ▼
[Generator]        코드 작성 / 문서 수정 / 실험 실행
    │
    ▼
[Evaluator]        CLAUDE.md 규칙 준수 검증 + 문서 간 일관성 확인
    │
    ▼ (통과?)
    ├── Yes → progress.md 업데이트 → 커밋
    └── No  → Generator로 돌아가 재작업 (루프)
```

**핵심 원리:** 한 번에 완벽하지 않아도 됩니다. **생성 → 검증 → 재생성 루프**를 돌면서 품질을 높입니다.

**제어 수준:** Adaptive Harness (에러 발생 시 개입, 구조 변경은 승인 필요)

| 영역 | 제어 수준 | 기준 |
|------|:---------:|------|
| 코드 구조 / 모듈 분리 | Strict | 6개 모듈 구조 변경 금지 |
| 서술 용어 / 논문 인용 | Strict | "과학적 검증 프레임워크" 준수 |
| 구현 세부사항 | Adaptive | AI 자율 판단, 에러 시 개입 |
| 파이프라인 흐름 변경 | Strict | 반드시 사용자 승인 |

---

## 프로젝트 정체성

- **이름:** HateSpeachStudy (HateXplain 기반 혐오표현 탐지 XAI 파이프라인)
- **목적:** 한성대학교 빅데이터프로그래밍 수업 프로젝트
- **프레임워크:** "선행연구 기반 가설 → 통제된 ablation → XAI 사후 검증"의 과학적 검증 프레임워크
- **척추 메시지:** *"단어 단서뿐 아니라 맥락 단서까지 함께 학습한 모델"*이 베이스 대비 분류 성능 + 판단 투명성 모두 향상됨을 정량 입증
- **베이스 진단 (H1):** 단어 과의존 → **개선 모델 (H3):** 단어 신호 유지 + 맥락 단서 추가 학습
- **절대 아님:** "XAI 피드백 루프", "순환적 프레임워크", "진단 → 개선 → 재진단 순환", "혐오는 단어가 아닌 맥락"(이분법 과장) ← 이런 표현 금지

---

## 핵심 규칙

### 1. 코드 작성 규칙

- **Python 3.13**, PyTorch MPS (Apple M3 Max)
- 한국어 주석 사용 (다정한 톤, 이모지 사용하지 않음)
- 기존 코드 스타일을 따름 — 새 파일 만들기 전에 기존 파일의 패턴을 확인
- 모듈 구조를 유지:
  - `experiment_core.py` — 데이터, 모델, 학습, 벤치마크, 튜닝, freeze study, 듀얼 헤드(D_B 부가)
  - `experiment_eda.py` — 탐색적 데이터 분석
  - `experiment_xai.py` — SHAP + LIME + LOO + **CI + IS + MSS + Attention Rollout** + Plausibility (4축)
  - `experiment_dashboard.py` — HTML 대시보드
  - `utils.py` — 공통 유틸리티 (경로, 시드, 평가, 시각화, 통계 검정 — Two-way/3-way ANOVA 포함)
  - `run_experiments.py` — CLI 엔트리포인트
- 새 모듈을 만들지 말 것 — 기존 6개 파일에 통합
- `run.sh` → `run_experiments.py` → 각 모듈 함수 호출 체계를 깨지 말 것

### 2. 모델 관련 규칙 (v2.1: 8조건 풀 ablation)

- **8조건 ablation 매트릭스** (2×2 × 2 사전학습 모델)
  - BERT 패밀리: A_B (베이스), B_B (+Attn), C_B (+VADER), D_B (둘 다)
  - RoBERTa 패밀리: A_R, B_R, C_R, D_R (동일 구조)
- 추가 모델: TF-IDF+LR/SVM (전통 ML 베이스라인), Freeze Study (BERT 동결 통제)
- 분류 헤드 통일: 모든 transformer 조건 MLP(768→256→3 또는 772→256→3)
- VADER 유무는 입력 차원 차이만 (768d ↔ 772d)
- L_target multi-task 부가: D_B 부가 실험에만 듀얼 헤드 (Main + Aux)
- 모델 추가/제거 시 반드시: benchmark 함수 + tuning 함수 + README + 명세서 v2 동기화

### 3. 학습 규칙 (v2.1)

- **모델 입력 단일 소스 원칙**: 텍스트(post_tokens)만 입력. rationale·target·source·agreement는 학습 supervision 또는 분석으로만 사용 (입력 X)
- **VADER는 텍스트 파생 피처** (추론 시 자동 계산) — 본 원칙 부합
- **학습에 사용 안 함**: Slur-Masking augmentation 제거, source 입력 X, agreement 미사용
- 학습 손실: `L_total = L_cls + α·L_attn (+ β·L_target)`
- α 그리드: {0.0, 0.1, 0.3, 0.5, 0.7, 1.0} — **B_B 조건에서 결정** 후 D_B/B_R/D_R 동일 적용
- 변수 통제: 시드/분할/하이퍼파라미터 동일, 단일 변수만 변경

### 4. 서술/문서 규칙

- **과학적 검증 프레임워크** (O) / 순환적 프레임워크 (X) / 피드백 루프 (X)
- **단어 vs 맥락 이분법 금지** — "혐오는 단어가 아닌 맥락" 표현 X / "단어 단서뿐 아니라 맥락 단서까지 함께 학습" (O)
- 베이스의 단어 *과의존*이 문제 — 단어 자체가 신호가 아닌 건 아님
- VADER 선택은 "Cheng (2022) 선행연구 기반 사전 가설"이지 "XAI 진단 결과"가 아님
- XAI는 "사후 검증 도구"로 서술
- Cheng(2022) Virginia Tech 논문은 반드시 인용하고 차별점 명시
- 한계점을 숨기지 말 것 — 투명하게 서술
- **단일 출처 명세서**: `docs/파이프라인_명세서_v2.md` 참조

### 5. Git 규칙

- 커밋 메시지: 한국어, `type: 제목` 형식 (feat, fix, docs, refactor)
- `cat` 명령어가 `bat`으로 alias 되어 있음 — HEREDOC 커밋 메시지 사용 금지, 직접 -m "..." 사용
- `.gitignore`에 있는 것: outputs/, data/, checkpoints/, *.pt, __pycache__/, .cache/, .mplconfig/
- 대용량 파일(체크포인트, 데이터) 절대 git에 올리지 말 것
- `빅데프 참고문헌pdf/` 는 저작권 때문에 git 제외

### 6. 실행 환경

- `run.sh`가 유일한 실행 진입점 — `python experiment_core.py` 직접 호출 금지
- SHAP은 CPU only (MPS 비호환) — SHAP 관련 코드에서 device 강제 지정
- `python` 명령어가 없을 수 있음 — `python3` 또는 conda 환경 사용

---

## 에이전트 작업 프로토콜

### Phase 1: Planner (작업 전)
1. `progress.md` 읽기 → 현재 위치와 Todo 파악
2. `architecture.md` 확인 → 수정 대상의 의존성/영향 범위 파악
3. 작업 계획을 세우고, 구조 변경이 필요하면 **멈추고 승인 요청**

### Phase 2: Generator (작업 중)
4. 코드 작성 / 문서 수정 / 실험 실행
5. 한 번에 완벽하지 않아도 됨 — 생성 후 검증으로 넘어감

### Phase 3: Evaluator (작업 후)
6. [ ] 변경한 파일 목록 확인
7. [ ] CLAUDE.md 규칙 위반 없는지 자체 검증
8. [ ] 문서 간 일관성 확인 (모델 수, 파이프라인 흐름, 용어)
9. [ ] 깨진 유니코드 없는지 확인
10. [ ] 검증 실패 시 → Phase 2로 돌아가 수정 (루프)

### Phase 4: Orchestrator (마무리)
11. [ ] `progress.md` 업데이트 (완료 항목 체크, 새 이슈 추가)
12. [ ] 커밋 준비 (변경 파일 목록 + 커밋 메시지 초안)

### 논리적 충돌 발생 시
- **즉시 Generator 루프를 중단**하고 사용자에게 승인 요청
- 예: 모델 아키텍처 변경, 파이프라인 흐름 변경, 평가 메트릭 변경, 새 의존성 추가

---

## 금지 사항

- 새 Python 모듈 생성 (기존 6개에 통합)
- `compat/` 디렉토리 부활 (삭제 완료)
- 호환 alias 복원 (run.sh에서 제거 완료)
- outputs/, checkpoints/ 를 git에 추가
- 실험 결과를 조작하거나 선택적으로 보고
