# CLAUDE.md — HateSpeachStudy 하네스 규칙

> 이 파일은 AI 에이전트가 프로젝트에서 작업할 때 반드시 따라야 하는 규칙입니다.
> 모든 작업 시작 전 이 파일을 읽고, 작업 후 `progress.md`를 업데이트하세요.

---

## 프로젝트 정체성

- **이름:** HateSpeachStudy (HateXplain 기반 혐오표현 탐지 XAI 파이프라인)
- **목적:** 한성대학교 빅데이터프로그래밍 수업 프로젝트
- **프레임워크:** "가설 → 실험 → XAI Before/After 검증"의 과학적 프레임워크
- **절대 아님:** "XAI 피드백 루프", "순환적 프레임워크", "진단 → 개선 → 재진단 순환" ← 이런 표현 금지

---

## 핵심 규칙

### 1. 코드 작성 규칙

- **Python 3.13**, PyTorch MPS (Apple M3 Max)
- 한국어 주석 사용 (다정한 톤, 이모지 사용하지 않음)
- 기존 코드 스타일을 따름 — 새 파일 만들기 전에 기존 파일의 패턴을 확인
- 모듈 구조를 유지:
  - `experiment_core.py` — 데이터, 모델, 학습, 벤치마크, 튜닝, freeze study
  - `experiment_eda.py` — 탐색적 데이터 분석
  - `experiment_xai.py` — SHAP + LIME + Overlap@5
  - `experiment_dashboard.py` — HTML 대시보드
  - `utils.py` — 공통 유틸리티 (경로, 시드, 평가, 시각화, 통계 검정)
  - `run_experiments.py` — CLI 엔트리포인트
- 새 모듈을 만들지 말 것 — 기존 6개 파일에 통합
- `run.sh` → `run_experiments.py` → 각 모듈 함수 호출 체계를 깨지 말 것

### 2. 모델 관련 규칙

- 모델은 6개 (TF-IDF+LR, TF-IDF+SVM, BERT-base, BERT+MLP, BERT+VADER, RoBERTa+VADER)
- Freeze Study 포함 시 7개 구성
- `TransformerMLPClassifier`는 ablation 통제 모델 — VADER 없이 동일 MLP 구조
- 모델 추가/제거 시 반드시: benchmark 함수 + tuning 함수 + README + docs 전부 동기화

### 3. 서술/문서 규칙

- **과학적 검증 프레임워크** (O) / 순환적 프레임워크 (X) / 피드백 루프 (X)
- VADER 선택은 "선행 연구 기반 사전 가설"이지 "XAI 진단 결과"가 아님
- XAI는 "사후 검증 도구"로 서술
- Cheng(2022) Virginia Tech 논문은 반드시 인용하고 차별점 명시
- 한계점을 숨기지 말 것 — 투명하게 서술

### 4. Git 규칙

- 커밋 메시지: 한국어, `type: 제목` 형식 (feat, fix, docs, refactor)
- `cat` 명령어가 `bat`으로 alias 되어 있음 — HEREDOC 커밋 메시지 사용 금지, 직접 -m "..." 사용
- `.gitignore`에 있는 것: outputs/, data/, checkpoints/, *.pt, __pycache__/, .cache/, .mplconfig/
- 대용량 파일(체크포인트, 데이터) 절대 git에 올리지 말 것
- `빅데프 참고문헌pdf/` 는 저작권 때문에 git 제외

### 5. 실행 환경

- `run.sh`가 유일한 실행 진입점 — `python experiment_core.py` 직접 호출 금지
- SHAP은 CPU only (MPS 비호환) — SHAP 관련 코드에서 device 강제 지정
- `python` 명령어가 없을 수 있음 — `python3` 또는 conda 환경 사용

---

## 작업 체크리스트

모든 작업 후:
1. [ ] 변경한 파일 목록 확인
2. [ ] `progress.md` 업데이트 (완료 항목 체크, 새 이슈 추가)
3. [ ] 문서 간 일관성 확인 (모델 수, 파이프라인 흐름, 용어)
4. [ ] 깨진 유니코드 없는지 확인

논리적 충돌 발생 시:
- **작업을 멈추고 사용자에게 승인 요청**
- 예: 모델 아키텍처 변경, 파이프라인 흐름 변경, 평가 메트릭 변경

---

## 금지 사항

- 새 Python 모듈 생성 (기존 6개에 통합)
- `compat/` 디렉토리 부활 (삭제 완료)
- 호환 alias 복원 (run.sh에서 제거 완료)
- outputs/, checkpoints/ 를 git에 추가
- 실험 결과를 조작하거나 선택적으로 보고
