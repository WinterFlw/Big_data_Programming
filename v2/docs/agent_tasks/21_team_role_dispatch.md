# 21. 5명 분배 — 실명·페이지·명령·마감 매핑 (확정안)

> 마지막 업데이트: 2026-05-17
> 본 문서는 작업 #1~#14 완료(2026-05-17) 직후 5인 전 분담을 한 곳에 모은 **확정안**이다.
> 5가지 차원(Stage / 발표 페이지 / 작업 #N 인계 / XAI 12지표 / Gate 6조건 / D0~D10 마일스톤)을 모두 실명에 매핑.
>
> 단일 진실 출처:
> - 카탈로그 (인원 미배정 깊이 버전): [`19_team_role_tracks.md`](../19_team_role_tracks.md)
> - 브리프 (작업 #1~#14): [`20_claude_code_completion_brief.md`](20_claude_code_completion_brief.md)
> - 회의용 카드 (인원 미배정 압축): [`22_stage_briefs.md`](22_stage_briefs.md)
> - 발표 페이지 분담: [`발표_와꾸_v2.md`](../../../docs/발표_와꾸_v2.md) (학교 산학협력 26p)
>
> 스왑이 필요하면 §0 표만 바꾸고 나머지는 자동으로 따라감.

---

## 0. 한 눈에 보기 (확정안)

### 0.1 Stage × 발표 페이지 × 첫 마감

| # | Stage / 역할 | 담당 | 발표 페이지 | 첫 마감 | 핵심 명령 |
|---|---|---|---|---|---|
| 1 | **Pilot** (Benchmark 실행) | **정수현** (팀장) | p2, p3, p12~p16, p20, p21 (9p) | **D3** | `./run.sh e2e benchmark --execute` |
| 2 | **Stat Auditor** (집계·검정) | **박종화** | p9, p10, p11, p17, p18 (5p) | **D5** | `./run.sh e2e aggregate` |
| 3 | **XAI Curator** (SHAP/LIME) | **차종민** | p1, p4, p5, p6 (4p) | **D7** | `./run.sh e2e xai-primary --resume` |
| 4 | **Author** (Bundle + Report + 오분류 분석 + Q&A 카드 + 시각화) | **조은** | p19, p22~p26 (6p) + 부록 슬라이드 + Q&A 카드 | **D9** | `./run.sh e2e xai-bundle && report && dashboard` + `scripts/figures.py` |
| 5 | **QA Conductor** (preflight + Gate) | **김정훈** | p7, p8 (2p) | **D0 매일** | `./v2/scripts/daily.sh` |

### 0.2 작업 #1~#14 결과 인계 (5인 1:n 매핑)

§2에 상세. 한 줄 요약:

```
정수현 (Pilot)        ← 작업 #1 (cudnn)   + #6 (statsmodels)  + #12 (AMP)
박종화 (Stat Auditor) ← 작업 #2 (ANOVA)   + #7 (bootstrap)    + #14 effect size
차종민 (XAI Curator)  ← 작업 #4 (adapter) + #8 ablation 4축   + #9 jsonl + #11 primary 4축
조은   (Author)       ← 작업 #5 (bundle)  + #10 subgroup×context + #14 subgroup
김정훈 (QA Conductor) ← 작업 #3 (failed/completed + daily.sh) + #13 (gate_check.py)
```

### 0.3 XAI 12지표 학습 책임자 (확정)

각자 본인 지표만 5분 발표 가능 수준까지 마스터. 발표 Q&A 대비.

| 축 | # | 지표 | 담당 | 본업 매칭 |
|---|---|---|---|---|
| 1 Attribution | 1 | SHAP | **박종화** | p18 평가지표 본문 |
| 1 Attribution | 2 | LIME | **김정훈** | 페이지 가벼우니 1축 보조 흡수 |
| 1 Attribution | 3 | LOO | **김정훈** | 단순한 sanity check |
| 2 Faithfulness | 4 | Comprehensiveness | **박종화** | p18 본문 |
| 2 Faithfulness | 5 | Sufficiency | **박종화** | p18 본문 |
| 2 Faithfulness | 6 | MSS | **박종화** | p18 본문 |
| **3 Context Learning ⭐** | 7 | CI (Concentration) | **차종민** | p4 학술 인용 (Gini 1912) |
| **3 Context Learning ⭐** | 8 | IS (Interaction Strength) | **차종민** | p5 Lundberg 2020 Nature MI |
| **3 Context Learning ⭐** | 9 | Attention Rollout | **차종민** | p5 Abnar 2020 ACL |
| 4 Plausibility | 10 | Token F1 | **조은** | p19 메인 결과 표 |
| 4 Plausibility | 11 | IOU | **조은** | p19 메인 결과 표 |
| 보조 | 12 | Overlap@5 | **김정훈** | 1축 안정성 보조 |

3축(차종민 3개)은 본 연구 결정 카드, 가장 깊이 마스터해야. p4·p5 학술 인용과 직결.

### 0.4 Full Run Gate 6조건 점검 책임자

QA Conductor가 매일 자동 점검(`gate_check.py`)하지만, FAIL 시 fix 책임자는 분리.

| # | 조건 | 자동 점검 | FAIL 시 fix 책임자 |
|---|---|---|---|
| 1 | cudnn 결정성 | gate_check 자동 | **정수현 (Pilot)** — A_B seed 42 두 번 재학습 |
| 2 | 8조건 metadata 정합성 | gate_check 자동 | **정수현 (Pilot)** — schema vs runtime spec 정렬 |
| 3 | aggregate CSV 7개 존재 | gate_check 자동 | **박종화 (Stat Auditor)** |
| 4 | XAI sample 결정성 (md5 일치) | gate_check 자동 (--skip-sample-check 옵션) | **차종민 (XAI Curator)** |
| 5 | failed_runs.csv 0건 | gate_check 자동 | **정수현 (Pilot)** — stderr.log 진단 + 재실행 |
| 6 | 산출물 contract 8개 존재 | gate_check 자동 | **조은 (Author)** — bundle/report 재실행 |

GO/STOP **최종 결정**은 **김정훈 (QA)**. STOP이면 위 fix 책임자에게 단톡 멘션.

### 0.5 D0~D10 마일스톤 책임자

| Day | 마일스톤 | 주 책임자 | 보조 |
|---|---|---|---|
| **D0** (5/17) | 본 문서 정독 + 단톡 카드 확정 + 환경 준비 | 전원 | 김정훈 단톡 사회 |
| D1 | A_B seed 42 smoke 1차 | 정수현 | 김정훈 daily.sh |
| D2 | A_B + D_B 2조건 smoke | 정수현 | 박종화 aggregate 점검 |
| **D3** | **Full Run Gate 6/6 통과 → GO 결정** | 김정훈 | 정수현 fix 대응 |
| D4 | full 120 unit 학습 시작 | 정수현 | 김정훈 매시간 status 모니터 |
| **D5** | 8조건 × 3시드 partial → ANOVA + p18 본문 박기 | 박종화 | 차종민 |
| D6 | full 120 진행 + p22~p26 골격 작성 | 정수현 + 조은 | — |
| **D7** | A_B + D_B × 15시드 XAI 완료 → 토큰 sanity + p4/p5 인용 매핑 | 차종민 | 박종화 |
| D8 | xai-bundle + report 자동 채움 검수 + p19 메인 결과 표 | 조은 | 차종민 |
| **D9** | 발표 자료 26p 1차 완성 | 조은 | 김정훈 페어 |
| D10 | 발표 리허설 + 척추 메시지·금지 표현 검수 + 최종 GO | 정수현 (팀장) | 전원 |

**굵게**: stage gate 마일스톤. 늦으면 critical path 영향.

---

## 1. 매핑 근거 (왜 이 사람이 이 stage)

발표 페이지 분담이 이미 확정된 상태라, 그 분담과 가장 자연스럽게 이어지는 stage를 묶었다. 인원·분량·역량 균형이 맞도록 조정.

### 정수현 → Pilot

- 팀장이자 파이프라인팀 본인. 명세서(p2, p3)와 모델/실험설계(p12~p16)를 직접 쓰는 사람이라, 학습 실행을 트리거하고 smoke 결과를 가장 빨리 읽을 수 있어야 한다.
- 발표 분량 9p로 가장 무거우므로 stage 자체는 D0~D3에 끝나고 D3 이후 페이지 작성에 시간을 더 쓰는 흐름이 안전하다.
- 작업 #1(cudnn 시드 결정성)이 Pilot의 첫 의존성 — 시드 재현성이 본 연구의 메인 메시지라 Pilot이 직접 이걸 확인하면 깔끔하다.

### 박종화 → Stat Auditor

- 평가지표 페이지(p18)와 EDA·라벨 신뢰성 페이지(p9~p11)를 담당하는 사람. 15 seed 평균/표준편차, 핵심 A_B vs D_B paired test, effect size 해석이 페이지 본문과 직접 연결된다. ANOVA·Holm은 보조/부록 분석으로만 다룬다.
- benchmark_summary.csv / paired_tests_holm.csv / anova_*.csv를 본인이 검수해야 p17(베이스라인 정의)·p18(평가지표) 본문이 수치로 들어찬다.
- 작업 #2(ANOVA 2-way/3-way)가 Stat Auditor의 첫 의존성.

### 차종민 → XAI Curator

- 선행연구팀. p1(배경), p4·p5(관련 연구), p6(차별성)에서 Mathew(2021), Cheng(2022), Lundberg(2017), Ribeiro(2016), Kennedy(2020)를 직접 인용한다. SHAP/LIME·rationale 학술 근거가 본인 본업.
- XAI 토큰 sanity check(BERT WordPiece `##` / RoBERTa BPE `Ġ` 서브워드 집계)는 학술 인용과 직접 맞닿아 있어 자연스럽다.
- 작업 #4(XAI adapter)가 XAI Curator의 첫 의존성 — pipeline/xai.py가 runtime의 SHAP·LIME 함수를 호출하는 흐름을 검수.

### 조은 → Author

- 문서·통합팀. p19(메인 결과), p22~p26(일정·분담·문제점·향후·통합)를 책임지는 본업이 정확히 final_report.md + 발표 자료 통합.
- 작업 #5(Bundle/Report 자동 채움)가 Author의 첫 의존성 — Stat/XAI 결과가 들어오면 markdown/dashboard가 자동으로 표·claim을 채우므로, Author는 placeholder 메시지 보고 결과 들어오기까지 골격만 다듬으면 된다.
- 한계 서술(p25)과 향후 연구(p26)도 본인 페이지라, xai_risk_flags.csv 기반의 limitations 섹션과 직접 연결.
- **추가 업무 A+B+C** (자동 채움 덕에 본업이 가벼우니 부담 균형 차원에서 추가):
  - A: 오분류 분석 — D_B가 틀린 sample을 직접 열어 hate↔offensive 혼동·implicit hate 미탐·target별 오분류율 정성 분류. 발표 부록 슬라이드 1장 + `final_report.md` "Error Analysis" 섹션.
  - B: Q&A 답변 카드 — v2 결과 기반 예상 질문 30~50개 + 답변. `docs/Q&A_v2_답변카드.md` 신규.
  - C: 추가 시각화 — matplotlib 그림 4장 (boxplot / target heatmap / SHAP 비교 / ANOVA η²). `scripts/figures.py` 신규 작성.
- 추가 업무로 주당 시간 7~10h → **10~15h**로 재조정. 다른 4명과 부담 균형.

### 김정훈 → QA Conductor

- 데이터팀(전처리). 페이지 분량 2p로 가장 가벼우므로 매일 daily.sh를 돌리고 5인 전체 결과를 모니터링할 여유가 있다.
- 데이터/전처리 본업이므로 split hash·VADER feature·prepare_data 회귀를 가장 먼저 잡아낼 수 있는 위치.
- 후반부(D7~D10)에는 Author 페어로 글쓰기 보조까지 들어간다 — 19 문서 §0.2 의존 그래프와 일치.
- 작업 #3(failed/completed CSV + daily.sh)이 QA Conductor의 첫 의존성.

---

## 2. 작업 #1~#14 결과 인계 (이미 끝난 부분 ↔ 사람)

본 분배 직전에 끝난 작업 #1~#14는 5 stage에 다대일로 매핑된다. 본인 stage의 산출물이 어디까지 와 있는지만 알면 첫 PR을 바로 시작 가능.

### 1차 라운드 (작업 #1~#5, 코드 골격 완성)

| 작업 # | 핵심 파일 | 산출물 위치 | 인계 받는 사람 | "받았을 때 해야 할 것" |
|---|---|---|---|---|
| #1 cudnn 결정성 | `v2/runtime/utils.py` | (런타임 동작) | **정수현 (Pilot)** | A_B seed 42 두 번 돌려서 macro_f1 ±0.0001 이내 확인 |
| #2 ANOVA | `v2/pipeline/statistics.py` | `outputs/.../benchmark/anova_*.csv` (3개) | **박종화 (Stat Auditor)** | aggregate 한 번 돌려 헤더 보이는지 확인, full seed 들어오면 실제 row 검수 |
| #3 failed/completed + daily.sh | `v2/pipeline/{artifacts,runner}.py`, `v2/scripts/daily.sh` | `outputs/.../{failed,completed}_runs.csv`, `daily.sh` | **김정훈 (QA Conductor)** | 오늘부터 매일 1회 `./v2/scripts/daily.sh` 실행 |
| #4 XAI adapter | `v2/pipeline/xai.py`, `xai_sampling.py` | `outputs/.../xai/{samples,primary,deep,ablation}/...` | **차종민 (XAI Curator)** | primary_samples.csv가 200row + seed 무관 md5 일치인지 확인, checkpoint 들어오면 첫 sample SHAP top-5 sanity check |
| #5 Bundle + Report | `v2/pipeline/xai_bundle.py`, `reporting.py` | `outputs/.../xai/evidence_bundle/...` (15개), `reports/final_report.{md,docx}`, `dashboard/index.html` | **조은 (Author)** | placeholder 메시지 확인 + p19 골격 작성 시작 |

### 2차 라운드 (작업 #6~#14, 완성도 채움)

| 작업 # | 핵심 파일 | 산출물 / 동작 | 인계 받는 사람 | "받았을 때 해야 할 것" |
|---|---|---|---|---|
| #6 statsmodels 의존 | `v2/runtime/requirements.txt` | NVIDIA에서 `pip install` 시 ANOVA 의존 자동 설치 | **정수현 (Pilot)** | `pip install -r runtime/requirements.txt` 한 번 |
| #7 Bootstrap CI | `pipeline/statistics.py` | `manifest.statistics.bootstrap_iterations > 0`이면 percentile bootstrap, 아니면 t-분포 | **박종화 (Stat Auditor)** | `benchmark_summary.csv` ci_low/ci_high가 bootstrap 값인지 확인 |
| #8 XAI 4축 메트릭 (ablation) | `pipeline/xai.py` | `xai_ablation_metrics.csv` 11컬럼 (attention_entropy/mss/IS/CI 추가) | **차종민 (XAI Curator)** | ablation CSV 보고 BERT/RoBERTa 4축 트렌드 해석 |
| #9 token_attributions.jsonl | `pipeline/xai_bundle.py` | `.cache/`→jsonl 평탄화, stale 자동 reset | **차종민 (XAI Curator)** | sample 1~2개 jsonl 열고 SHAP/LIME 토큰 sanity check |
| #10 subgroup × context | `pipeline/xai_bundle.py` | `subgroup_xai_metrics.csv` source × target 두 차원, `context_metrics.csv` window/sensitivity | **조은 (Author)** | p19 메인 결과에서 subgroup 차이 언급 |
| **#11 primary 4축** | `pipeline/schema.py`, `pipeline/xai.py` | `seed_level_metrics.csv` 18컬럼 (primary도 CI/MSS/IS/AttnEntropy) + 신규 `sample_level_metrics.csv` | **차종민 (XAI Curator)** + **조은 (Author)** | XAI Curator는 4축 트렌드 해석, Author는 subgroup 표 본문 박기 |
| **#12 AMP autocast** | `runtime/experiment_core.py` | NVIDIA CUDA에서 fp16 학습 시간 30~50% 단축 | **정수현 (Pilot)** | 첫 smoke 학습 후 epoch당 시간 비교 |
| **#13 Gate 자동 판정** | `scripts/gate_check.py`, `scripts/daily.sh` | `[Gate: GO/STOP]`로 daily.sh 종료. exit 0/1 | **김정훈 (QA Conductor)** | D0부터 매일 결과 단톡 한 줄 보고 |
| **#14 ANOVA effect size + sample-level subgroup** | `pipeline/statistics.py`, `pipeline/xai_bundle.py` | `anova_*.csv`에 eta²/partial η², subgroup이 sample × source/target 진짜 분해 | **박종화 (Stat Auditor)** + **조은 (Author)** | Stat은 효과 크기 해석(Cohen 기준), Author는 subgroup 본문 통합 |

**중요**: 작업 #1~#14는 "코드 완성"이지 "데이터 채움"이 아니다. CSV/JSON은 대부분 헤더만 있고 GPU 학습이 끝나야 row가 채워진다. 각 사람의 D0~D3은 "골격·placeholder 검수 + 본인 첫 명령 1회 실행"이 끝이다.

### 코드 완성도 (1차+2차 라운드 종합)

| Stage | 1차 완료 후 | **2차 완료 후 (현재)** |
|---|:---:|:---:|
| 1. Benchmark | 95% | **100%** (cudnn + AMP) |
| 2. Statistics | 85% | **100%** (ANOVA + bootstrap + effect size) |
| 3. XAI Core | 40% | **100%** (4축 primary + sample-level + cache jsonl) |
| 4. XAI Bundle + Report | 30% | **100%** (진짜 subgroup + claim 자동) |
| 5. QA + Server | 90% | **100%** (Gate 자동 판정) |

---

## 3. D0 ~ D10 일정 (5인 통합)

```
D0 (오늘, 5/17)
  ├─ 모두: 본 문서 + 19/20 문서 정독 (60분)
  ├─ 모두: 본인 카드 stage의 카탈로그 섹션(19 문서) 정독 (30분)
  ├─ 김정훈(QA): ./v2/scripts/daily.sh 첫 실행 → 결과 단톡 공유
  └─ 정수현(Pilot): NVIDIA 서버 접근 권한·CUDA 버전·디스크 확인

D1
  ├─ 정수현(Pilot): A_B seed 42 smoke (1 시드만) → 산출물 6개 ls 확인
  ├─ 박종화(Stat Auditor): aggregate 1회 → benchmark_runs.csv 헤더 점검
  ├─ 차종민(XAI Curator): xai-primary 1회 (dry-run) → primary_samples.csv 200row
  ├─ 조은(Author): report 1회 → final_report.md 본문 placeholder 메시지 확인
  └─ 김정훈(QA): daily.sh 2회차 → 회귀 0건 확인

D2
  ├─ 정수현(Pilot): A_B + D_B 2조건 × seed 42 smoke (≈30분)
  ├─ Stat/XAI/Author: 본인 발표 페이지 골격 markdown 작성 시작
  └─ 김정훈(QA): Full Run Gate 6조건 중 1~2 통과 점검

D3 — Full Run Gate
  ├─ 정수현(Pilot): A_B 모든 seed 1회 시드 결정성 확인 (±0.0001)
  ├─ 김정훈(QA): 6조건 모두 통과 판단 → GO/STOP 단톡 보고
  └─ GO이면: 120 unit 풀 벤치마크 시작 (3~5일 예상)

D4 ~ D7 (벤치마크 진행 중)
  ├─ 정수현(Pilot): 매일 status 확인, failed unit 즉시 재실행
  ├─ 박종화(Stat Auditor): partial aggregate 가능, ANOVA row 채워지는지 모니터
  ├─ 차종민(XAI Curator): A_B seed 42 checkpoint 들어오면 xai-primary --resume 시동
  ├─ 조은(Author): p22~p26 (일정·분담·문제점·향후·통합) markdown 완성
  └─ 김정훈(QA): daily.sh 매일, Author 페어 글쓰기 보조 시작

D5 — Stat Mid-checkpoint
  └─ 박종화: 8조건 × 3 seed (최소) 결과 들어왔을 때 paired test + ANOVA 검수,
            p17/p18 본문에 수치 박기

D7 — XAI Mid-checkpoint
  └─ 차종민: A_B + D_B × 15 seed XAI 완료 → seed_level_metrics.csv 검수,
            SHAP top-5 1~2 sample 직접 눈으로 sanity check

D8 ~ D9
  ├─ 조은(Author): xai-bundle + report 자동 채움 결과 검수,
  │                  p19 메인 결과 표 + p24 결과 요약 완성
  ├─ 박종화: ANOVA 3-way 결과 페이지 통합 (p18)
  └─ 차종민: p6 차별성 표를 실제 결과 수치로 갱신

D10 — 통합 리허설
  ├─ 모두: 26p 발표 자료 통합 1회독
  ├─ 김정훈(QA) + 조은(Author): 척추 메시지 일관성 검수,
  │                                  금지 표현("순환적", "이분법") 0건 확인
  └─ 정수현(팀장): 최종 GO 결정
```

---

## 4. 5명 카드 (실명별 상세)

### 4.1 정수현 — Pilot (Benchmark Stage Owner)

**한 문장 요약**: "8조건 × 15시드 = 120개 학습을 NVIDIA 서버에서 안정적으로 굴려서 완료시키는 사람."

**발표 페이지 (9p)**:
- p2 프로젝트 목표
- p3 연구 질문 + 가설 (H1~H4)
- p12 구현 모델 6+1 라인업
- p13 모델 선정 이유 + 학습 방법 (★ 평가 핵심)
- p14 하이퍼파라미터
- p15 모델 구현 + Freeze Study
- p16 실험 설계 — 8조건 풀 ablation
- p20 중간 성능 분석 + 해석
- p21 일정 대비 진행 현황

**받은 작업 산출물**:
- 작업 #1: cudnn.deterministic=True / cudnn.benchmark=False (CUDA 시드 결정성)

**D0~D3 To-Do** (브리프 §부록 C 1번 + 19 문서 §1 카드):
- [ ] NVIDIA 서버 접근 권한 + `nvidia-smi`로 GPU 사양 / 메모리 확인
- [ ] `pip install -r v2/runtime/requirements.txt` 환경 구축
- [ ] `data/dataset.json` / `data/post_id_divisions.json` 서버 업로드
- [ ] **시드 결정성 smoke**: A_B seed 42 두 번 학습 → macro_f1 ±0.0001 이내 일치 확인
- [ ] **smoke 산출물 6개 ls 확인**:
  - `outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json`
  - `outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv`
  - `outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/run_config.json`
  - `outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/predictions.csv`
  - `outputs/experiments/v2_15seed/benchmark/checkpoints/a_b_seed_42.pt`
  - `outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stdout.log`
- [ ] Full Run Gate 6조건 중 1~3번 통과 시점에 단톡 공유

**핵심 명령**:
```bash
cd v2
# smoke
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute

# 시드 재현성 확인 (두 번째 실행)
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --force

# 두 macro_f1 차이가 0.0001 이내인지 비교
python3 -c "
import json
m1 = json.load(open('outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json'))
print(f'macro_f1 = {m1[\"macro_f1\"]}')"

# Full Run Gate 통과 후 120 unit
./run.sh e2e benchmark --run-id v2_15seed --execute --resume
```

**"내가 한 일" 1문장**:
> "NVIDIA 서버에서 8조건 × 15시드 학습을 직접 실행하고, cudnn 결정성 + 시드 재현성 smoke를 통해 Full Run Gate를 통과시켰습니다."

---

### 4.2 박종화 — Stat Auditor (Statistics Stage Owner)

**한 문장 요약**: "학습이 끝난 metrics.json 120개를 받아서 15 seed summary와 핵심 A_B vs D_B paired test, effect size를 손계산으로 한 번 더 검증하는 사람."

**발표 페이지 (5p)**:
- p9 EDA ① — 라벨·소스 분포
- p10 EDA ② — Target 편향
- p11 EDA ③ — XAI 정당성·라벨 신뢰성
- p17 베이스라인 모델 정의
- p18 성능 평가 지표 — 분류 + XAI 4축

**받은 작업 산출물**:
- 작업 #2: ANOVA 2-way (BERT/RoBERTa) + 3-way (cross-family) 자동 생성

**D0~D5 To-Do**:
- [ ] `outputs/experiments/v2_15seed/benchmark/` 의 4개 CSV 헤더 점검
  - benchmark_runs.csv / benchmark_summary.csv / paired_tests_holm.csv / anova_3way.csv
- [ ] 정수현이 첫 smoke seed 1개 완료시키면 aggregate 돌려 row 1줄 채워지는지 확인
- [ ] paired test 손계산: A_B vs D_B macro_f1 시드별 차이의 mean/std/effect size가 CSV와 일치하는지 1행 검증
- [ ] Cohen dz 손계산: mean_diff / std_diff가 effect_size 컬럼과 일치하는지 1행 검증
- [ ] D5 시점: 8조건 × 3 seed가 들어오더라도 본문은 A_B vs D_B 핵심 비교 중심으로 쓰고, ANOVA는 부록/보조 분석으로만 짧게 확인

**핵심 명령**:
```bash
cd v2
./run.sh e2e aggregate --run-id v2_15seed

# benchmark_runs.csv 행 수 점검
wc -l outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv

# 손계산 검증용
column -t -s, outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv | head -20
column -t -s, outputs/experiments/v2_15seed/benchmark/anova_2way_bert.csv
```

**손계산 검증 sheet** (Excel/Numbers):
- A_B vs D_B paired t-test: 두 condition의 macro_f1 시드별 차이를 직접 평균/표준편차 계산 → t-statistic → scipy 결과와 비교
- BERT 2-way ANOVA: SS_attention, SS_vader, SS_interaction, SS_residual 분해를 손으로 한 번 따라가기

**"내가 한 일" 1문장**:
> "15시드 평균/표준편차와 핵심 paired t-test, Cohen's dz/effect size 결과를 CSV로 정리하고 손계산으로 교차 검증했습니다."

---

### 4.3 차종민 — XAI Curator (XAI Core Stage Owner)

**한 문장 요약**: "A_B vs D_B 모델 + 8조건 ablation에서 SHAP/LIME/Comp/Suff/CI/MSS/IS/Rollout 같은 XAI 지표를 정확히 산출하고 토큰 sanity check까지 책임지는 사람."

**발표 페이지 (4p)**:
- p1 프로젝트 수행 배경 + 필요성
- p4 관련 연구 ① — 혐오표현 탐지 흐름
- p5 관련 연구 ② — XAI + Rationale + Slur 의존
- p6 차별성 — 4가지 (Mathew/Cheng/본 프로젝트 비교표)

**받은 작업 산출물**:
- 작업 #4: pipeline/xai.py + xai_sampling.py — runtime SHAP/LIME 호출 어댑터, 결정적 sample 선택, 캐싱

**D0~D7 To-Do**:
- [ ] **sample 결정성 확인**: primary_samples.csv 두 번 재생성 후 md5 일치 검증
  ```bash
  md5sum outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
  ./run.sh e2e xai-primary --run-id v2_15seed
  md5sum outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
  ```
- [ ] primary_samples.csv (200) / deep_samples.csv (500) / ablation_samples.csv (50) 행 수 점검
- [ ] checkpoint 들어오면 (정수현이 A_B seed 42 smoke 완료 후) xai-primary 1회 실제 실행:
  ```bash
  ./run.sh e2e xai-primary --run-id v2_15seed
  ```
- [ ] **토큰 sanity check 2회**: 1~2 sample을 골라 SHAP top-5 토큰이 텍스트와 의미적으로 맞는지 직접 눈으로 확인
  ```bash
  python3 -c "
  import json, csv
  from pathlib import Path
  cache = json.load(open('outputs/experiments/v2_15seed/xai/.cache/a_b_seed_42.json'))
  for s in cache['shap'][:3]:
      print(f'text: {s[\"text\"][:80]}')
      print(f'top_tokens: {s[\"top_tokens\"]}')
      print('---')
  "
  ```
- [ ] BERT WordPiece (`##`) / RoBERTa BPE (`Ġ`) 서브워드 집계가 어색하지 않은지 — runtime의 `_aggregate_subword_scores` 호출 결과가 단어 단위로 정확한지 확인
- [ ] D7 시점: A_B + D_B × 15 seed 완료 후 seed_level_metrics.csv 검수, seed_stability.csv의 top-k Jaccard / rank_corr 수치 해석

**핵심 명령**:
```bash
cd v2
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed

# 산출물 7개 ls
ls outputs/experiments/v2_15seed/xai/samples/*.csv
ls outputs/experiments/v2_15seed/xai/primary/*.csv
ls outputs/experiments/v2_15seed/xai/deep/case_summary.csv
ls outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv
ls outputs/experiments/v2_15seed/xai/xai_summary.json
```

**캐시 활용**: `outputs/experiments/v2_15seed/xai/.cache/<cond>_seed_<seed>.json` — 한 sample SHAP 계산이 수 초~수십 초이므로 재실행 시 자동 hit. sample_size를 바꾸면 size 불일치로 무효화됨.

**"내가 한 일" 1문장**:
> "SHAP/LIME 출력을 토큰 단위로 sanity check하고 BERT/RoBERTa 서브워드 집계, seed 무관 sample 선택, attribution 캐싱이 정확히 동작하는지 검증했습니다."

---

### 4.4 조은 — Author (XAI Bundle + Report Stage Owner)

**한 문장 요약**: "Stat/XAI 결과를 받아 evidence bundle 15개 + final_report.md + 발표 자료 26p를 통합하는 사람."

**발표 페이지 (6p)**:
- p19 베이스라인 vs 제안 모델 비교 (메인 결과 표)
- p22 향후 남은 일정
- p23 팀원별 역할
- p24 주요 실험 결과 요약 (1페이지 요약)
- p25 진행 중 발생한 문제점 (한계점)
- p26 모델 개선 방향 + 향후 연구

**받은 작업 산출물**:
- 작업 #5: pipeline/xai_bundle.py + reporting.py — Stat/XAI 입력이 들어오면 markdown 표·dashboard 카드·xai_claims.json이 자동 채워짐

**D0~D9 To-Do**:
- [ ] **placeholder 검수**: 빈 입력 상태에서 final_report.md 본문이 5개 새 섹션 헤더(Benchmark Summary / Paired Tests / ANOVA × 3 / XAI Evidence Summary / Seed Stability / Limitations / Reproducibility)와 "_no ... yet — populate by ..._" 메시지를 정확히 표시하는지 확인
- [ ] **자동 채움 검증**: 가짜 데이터 한 줄 주입 → xai-bundle + report 재실행 → strong/moderate claim이 자동 생성되는지 확인 (작업 #5 검증 명령 그대로)
- [ ] **p22~p26 markdown 골격** 작성 시작 (외부 결과 의존 없음):
  - p22 일정: 발표_와꾸_v2.md §22 표 그대로
  - p23 역할: 본 문서 §0 표 그대로
  - p25 한계점: 19 문서 §6 + xai_risk_flags.csv 예상 항목
  - p26 향후 연구: 발표_와꾸_v2.md §26 그대로
- [ ] D8~D9 시점: 박종화/차종민 결과 들어오면 p19 메인 결과 표 + p24 결과 요약 1페이지로 통합
- [ ] **금지 표현 검사**: "순환적", "피드백 루프", "혐오는 단어가 아닌 맥락", "XAI 진단 결과 기반" 0건 확인
- [ ] **일관 표현 검사**: "과학적 검증 프레임워크", "전이학습 기반 full fine-tuning", "VADER는 Cheng (2022) 선행연구 기반 사전 가설", "단어 단서뿐 아니라 맥락 단서까지 함께 학습" 모든 슬라이드 통일

**핵심 명령**:
```bash
cd v2
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

# 본문 확인
head -80 outputs/experiments/v2_15seed/reports/final_report.md
open outputs/experiments/v2_15seed/dashboard/index.html  # macOS

# JSON 유효성
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json > /dev/null
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json > /dev/null
```

**페어 작업** (D7~D10): 김정훈(QA)과 함께 final_report.md 본문 + 발표 자료 통합 검수.

**"내가 한 일" 1문장**:
> "evidence bundle 15개 + final_report.md/docx + dashboard HTML + 발표 자료 26p를 작성하고, 통계적으로 확증된 claim만 자동 채워지도록 source_artifacts 검증을 책임졌습니다."

---

### 4.5 김정훈 — QA Conductor (QA + Server Stage Owner)

**한 문장 요약**: "매일 daily.sh를 돌려서 5인 전체 산출물이 깨지지 않게 관제하고, Full Run Gate 6조건으로 풀 학습 GO/STOP을 판단하며, 후반에는 Author와 글쓰기 페어로 들어가는 사람."

**발표 페이지 (2p)**:
- p7 실제 데이터 설명
- p8 전처리 파이프라인

**받은 작업 산출물**:
- 작업 #3: failed/completed CSV 분리 + v2/scripts/daily.sh

**D0~D10 To-Do** (D0부터 상시):
- [ ] **D0 첫 실행**: `./v2/scripts/daily.sh` 한 번 돌려서 [daily preflight ok] 떨어지는지 확인
- [ ] **매일 1회 daily.sh 실행** → 단톡에 결과 한 줄 보고
  - failed_runs.csv 행 수
  - completed_runs.csv 행 수
  - 회귀 발생 시 stderr.log 즉시 공유
- [ ] **Full Run Gate 6조건 점검** (D3 시점):

  | # | 조건 | 확인 방법 |
  |---|---|---|
  | 1 | cudnn 결정성 확인 (정수현 smoke) | A_B seed 42 두 번 macro_f1 ±0.0001 |
  | 2 | 8조건 metadata 정합성 | CONDITION_METADATA / V2_CONDITION_SPECS 일치 |
  | 3 | aggregate 빈 입력에서도 무회귀 | daily.sh 통과 |
  | 4 | XAI sample 결정성 | primary_samples.csv md5 일치 |
  | 5 | failed_runs.csv 0건 | smoke seed들에서 fatal marker 없음 |
  | 6 | docs/02_e2e_pipeline 산출물 contract 일치 | daily.sh end-to-end ls 확인 |

- [ ] **GO/STOP 판단**: 6조건 모두 통과 시 단톡에 "Full Run Gate: GO. 120 unit 학습 시작 권장." 보고
- [ ] **서버 시간 관리**: GPU 사용 시간대 표 작성, conflict 시 정수현과 조정
- [ ] **D7~D10 Author 페어**: 조은과 함께 final_report.md 본문 + 발표 자료 통합 검수, 척추 메시지·금지 표현 교차 검수

**핵심 명령**:
```bash
# 매일 1회 (alias로 등록 권장)
./v2/scripts/daily.sh

# 회귀 발생 시
ls -la v2/outputs/experiments/v2_15seed/benchmark/runs/*/seed_*/stderr.log
grep -E "RuntimeError|Traceback|CUDA out of memory|nan loss" v2/outputs/experiments/v2_15seed/benchmark/runs/*/seed_*/stderr.log

# Author 페어 검수
diff v2/outputs/experiments/v2_15seed/reports/final_report.md <(./run.sh e2e report --run-id v2_15seed --dry-run 2>/dev/null)
```

**"내가 한 일" 1문장**:
> "매일 daily.sh로 전체 stage를 관제하고, Full Run Gate 6조건으로 풀 120 unit 학습 GO/STOP을 판단했으며, Author와 글쓰기 페어로 발표 자료의 척추 메시지·금지 표현을 검수했습니다."

---

## 5. 의존 그래프 (시간순)

```
   D0 ───── D1 ───── D2 ───── D3 ───── D5 ───── D7 ───── D9 ───── D10
   │        │        │        │        │        │        │        │
[모두]   [정수현] [정수현] [정수현] [박종화] [차종민] [조은] [정수현]
정독     smoke   2조건  Gate6   ANOVA   XAI    p19/24  최종
                  smoke 통과    검수    검수    통합    GO
                            │
                            ▼
                       120 unit
                       full run
                            │
                            ▼
                        [정수현]
                        매일 status
                                
[김정훈]  daily.sh 매일 (D0~D10 상시)
                                                  
                                                [김정훈]
                                                + Author
                                                페어 보조
                                                (D7~D10)
```

**Critical Path**: 정수현(D3 Gate) → 박종화(D5 ANOVA) → 차종민(D7 XAI) → 조은(D9 Report) → 정수현(D10 최종 GO).

**Side Channel**: 김정훈 매일 daily.sh + D7~D10 Author 페어.

---

## 6. Full Run Gate 6조건 (D3 결정 기준)

QA Conductor가 점검·보고. Pilot은 결과 제공. 모두 통과해야 120 unit 풀 학습 시작.

| # | 조건 | 통과 기준 | 점검 방법 |
|---|---|---|---|
| 1 | **cudnn 결정성** | A_B seed 42 두 실행의 macro_f1 ±0.0001 | Pilot 두 번 학습 후 metrics.json 비교 |
| 2 | **8조건 metadata 정합성** | CONDITION_METADATA 8개 ↔ V2_CONDITION_SPECS 8개 동일 | schema.py 정독 |
| 3 | **aggregate 빈 입력 무회귀** | `./run.sh e2e aggregate` 무오류 + 7개 CSV 생성 | daily.sh 자동 |
| 4 | **XAI sample 결정성** | primary_samples.csv 두 번 생성 후 md5 일치 | XAI Curator 명령 |
| 5 | **failed_runs.csv = 0건** | smoke 시드들에서 stderr fatal marker 없음 | daily.sh 자동 |
| 6 | **산출물 contract** | docs/02_e2e_pipeline §6에 명시된 모든 산출물 경로가 daily.sh ls에 등장 | daily.sh 자동 |

5/6 통과 시 STOP — 깨진 조건 stage owner가 해결 후 재점검.

---

## 7. 회의 주기

- **D0 (5/17)**: 30분, 본 문서 + 19/20 문서 정독 + 본인 카드 확정
- **D1, D2**: 단톡 보고만, 회의 없음
- **D3 (Full Run Gate)**: 30분, 6조건 점검 결과 공유 + GO/STOP 결정
- **D5 (Stat Mid)**: 30분, ANOVA 결과 미리보기 + p17/p18 본문 박을 수치 확정
- **D7 (XAI Mid)**: 30분, SHAP top-5 sanity check + seed_stability 해석
- **D9 (Final draft)**: 60분, final_report.md + 발표 자료 26p 1회독
- **D10 (리허설)**: 90분, 발표 리허설 + 최종 GO

---

## 8. AI 에이전트에 던지는 첫 명령 (각 stage의 첫 PR)

본인 카드를 AI에게 통째로 던질 때 쓰는 첫 문장 예시. 19 문서의 §"에이전트에 던질 첫 문장" 섹션을 본인 stage용으로 정리한 것.

### 정수현 (Pilot)
> "v2/docs/agent_tasks/20_claude_code_completion_brief.md 부록 C 1번 Pilot 카드 + v2/docs/19_team_role_tracks.md §1 Benchmark 카드를 읽고, NVIDIA 서버에서 A_B seed 42 smoke를 시작하기 위한 환경 점검 + 첫 명령을 step-by-step으로 알려줘. 명령은 실제 NVIDIA 환경 가정. cudnn 결정성은 작업 #1에서 이미 적용되어 있음."

### 박종화 (Stat Auditor)
> "v2/docs/agent_tasks/20 부록 C 2번 + 19 §2 Statistics 카드를 읽고, aggregate가 만든 paired_tests_holm.csv 한 행을 손계산으로 검증하는 절차를 알려줘. ANOVA 2-way는 작업 #2에서 이미 들어가 있음."

### 차종민 (XAI Curator)
> "v2/docs/agent_tasks/20 부록 C 3번 + 19 §3 XAI Core 카드를 읽고, primary_samples.csv가 seed 무관 결정적이며 SHAP top-5 토큰이 의미적으로 맞는지 sanity check하는 절차를 알려줘. pipeline/xai.py + xai_sampling.py는 작업 #4에서 이미 구현됨."

### 조은 (Author)
> "v2/docs/agent_tasks/20 부록 C 4번 + 19 §4 Author 카드를 읽고, 현재 placeholder 상태의 final_report.md를 발표 26p의 p19/p22~p26 와꾸로 골격을 채우는 markdown 작성 가이드를 줘. xai_bundle + report 자동 채움은 작업 #5에서 이미 구현됨."

### 김정훈 (QA Conductor)
> "v2/docs/agent_tasks/20 부록 C 5번 + 19 §5 QA 카드를 읽고, 오늘부터 매일 ./v2/scripts/daily.sh 실행 후 결과를 단톡에 어떻게 보고할지 한 줄 양식을 만들어줘. Full Run Gate 6조건 점검 자동화도 검토. failed/completed + daily.sh는 작업 #3에서 이미 구현됨."

---

## 9. 산출물 위치 한 눈에 정리

```
v2/
├── outputs/experiments/v2_15seed/
│   ├── execution_status.csv            ← QA Conductor 매일 확인
│   ├── failed_runs.csv                 ← QA Conductor 매일 확인 (작업 #3)
│   ├── completed_runs.csv              ← QA Conductor 매일 확인 (작업 #3)
│   ├── benchmark/
│   │   ├── benchmark_runs.csv          ← Pilot 채움 / Stat 검수
│   │   ├── benchmark_summary.csv       ← Stat Auditor 검수 (bootstrap CI, 작업 #7)
│   │   ├── paired_tests.csv            ← Stat Auditor 검수
│   │   ├── paired_tests_holm.csv       ← Stat Auditor 검수
│   │   ├── anova_2way_bert.csv         ← eta²/partial η² 포함 (작업 #2 + #14)
│   │   ├── anova_2way_roberta.csv      ← eta²/partial η² 포함
│   │   ├── anova_3way.csv              ← eta²/partial η² 포함
│   │   ├── runs/<cond>/seed_<n>/       ← Pilot 실행 결과 (AMP fp16, 작업 #12)
│   │   └── checkpoints/<cond>_seed_<n>.pt ← XAI Curator가 읽음
│   ├── xai/
│   │   ├── samples/                    ← XAI Curator 검수 (md5 일치)
│   │   ├── primary/
│   │   │   ├── seed_level_metrics.csv  ← 4축 18컬럼 (작업 #4 + #11)
│   │   │   ├── sample_level_metrics.csv ← subgroup 분해 입력 (작업 #14)
│   │   │   ├── paired_xai_tests.csv
│   │   │   └── seed_stability.csv
│   │   ├── deep/                       ← case_summary, xai_details
│   │   ├── ablation/                   ← 11컬럼 (작업 #4 + #8)
│   │   ├── xai_summary.json
│   │   ├── .cache/                     ← attribution 캐시 → token_attributions.jsonl 입력
│   │   └── evidence_bundle/            ← Author 채움 (작업 #5/#9/#10/#14)
│   ├── reports/
│   │   ├── final_report.md             ← Author 검수 (작업 #5, 자동 표 채움)
│   │   └── final_report.docx           ← Author 검수
│   └── dashboard/
│       └── index.html                  ← Author 검수 (benchmark/XAI summary cards)
└── scripts/
    ├── daily.sh                        ← QA Conductor 매일 실행 (작업 #3 + #13)
    └── gate_check.py                   ← QA Conductor: Full Run Gate 6조건 자동 (작업 #13)
```

---

## 10. 막힐 때

- **본인 stage 안에서**: 19 문서 §"막힐 때" + §"에이전트에 던질 첫 문장" 그대로 활용
- **stage 간 인계**: 본 문서 §3 "D0~D10 일정" + §5 "의존 그래프" 의 마일스톤 확인
- **GO/STOP 판단**: 김정훈(QA) → 정수현(팀장) 순으로 에스컬레이션
- **글쓰기 충돌**: 조은(Author) → 김정훈(QA 페어) 검수 → 정수현 최종 결재

---

문서 끝.
