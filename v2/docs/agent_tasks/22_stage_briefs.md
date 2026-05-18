# 22. Stage 선택 카드 — 회의용

> 마지막 업데이트: 2026-05-17
> 5개 stage의 직무 · 비유 · 난이도★ · 하루 모양 · "내가 잘할 수 있겠다" 신호를 한 페이지에.
>
> **인원 매핑은 [`21_team_role_dispatch.md`](21_team_role_dispatch.md) §0에서 확정됨.**
>
> 단일 진실 출처:
> - 카탈로그 (깊이 있는 카드): [`19_team_role_tracks.md`](../19_team_role_tracks.md)
> - 본 문서 (회의용 압축 카드): 22번
> - **실명 매핑 (확정안)**: [`21_team_role_dispatch.md`](21_team_role_dispatch.md)

---

## 확정 매핑 한 줄 (21 문서 §0.1 발췌)

```
Stage 1  Pilot           — 정수현 (팀장)         p2, p3, p12~p16, p20, p21 (9p)
Stage 2  Stat Auditor    — 박종화                p9~p11, p17, p18 (5p)
Stage 3  XAI Curator     — 차종민                p1, p4, p5, p6 (4p)
Stage 4  Author          — 조은                  p19, p22~p26 (6p)
Stage 5  QA Conductor    — 김정훈                p7, p8 (2p)
```

XAI 12지표·Gate 6조건 fix 책임자·D0~D10 마일스톤 책임자는 21 §0.3~§0.5 참조.

---

## 사용법 (회의 5분)

1. 5장 카드 각자 1분씩 훑기.
2. 본인이 "내가 잘할 수 있겠다 신호" 3개 중 2개 이상 해당하는 카드 1순위·2순위 메모.
3. 마지막 §치트시트 표 보고 본인 강점 축이 어디인지 확인.
4. **Author 먼저 → Pilot → 나머지 3명** 순서로 정함. Author가 글쓰기 부담이 가장 크기 때문에 starting point.
5. 결정되면 21 문서 §0 표만 갱신.

---

## Stage 1 — Pilot (Benchmark 실행자)

**한 줄**
> 8조건 × 15시드 = 120개 학습을 NVIDIA 서버에서 **안정적으로 굴려서 끝까지 완료시키는** 사람.

**비유**
> 비행기 조종사. smoke로 활주로 점검 → Gate 통과 신호 받음 → 본 비행 8시간. 비행 중에는 매일 계기판(`status`) 한 번씩 확인.

**난이도 ★1~5**

| 축 | 점수 | 이유 |
|---|---|---|
| 명령 실행 | ★★☆☆☆ | `./run.sh e2e benchmark --execute` 한 줄. 옵션 몇 개. |
| 코드 이해 | ★★★☆☆ | `runtime/experiment_core.py` 학습 루프 흐름은 알아야 OOM·NaN loss 트러블슈팅 가능. |
| 통계 | ★☆☆☆☆ | 시드 결정성 ±0.0001 확인만. 어려운 통계 없음. |
| 글쓰기 | ★★☆☆☆ | smoke 결과 단톡에 한 줄. 발표 페이지(설계·일정)는 별개 본업. |
| 서버 운영 | ★★★★☆ | CUDA 버전, 디스크 공간, 시간대 관리, failed unit 재시동. |

**하루가 어떤 모양**
> *D2 오전*: `./run.sh e2e benchmark --conditions A_B --seeds 42 --execute --force`로 두 번째 학습 → 1시간 뒤 `metrics.json` 두 개 `macro_f1` 비교 → ±0.0001 일치 확인 → 단톡: "*A_B seed 42 결정성 OK. Gate 6조건 중 1번 통과.*"
>
> *D4~D7 매일*: `./run.sh e2e status` 1회 + failed unit 있으면 stderr.log 읽고 재실행.

**"내가 잘할 수 있겠다" 신호**
- NVIDIA 서버 접근 권한·시간이 확보됨
- bash·python 명령 실행이 손에 익음
- 모델 한 번이라도 직접 학습 돌려본 경험
- 새벽에 학습 끝났는지 확인하러 들어가는 게 부담 없음

**시간 부하 곡선**: D0~D3 ★★★★☆ → 이후 D10까지 매일 status 5분 ★★☆☆☆

---

## Stage 2 — Stat Auditor (수치 검수자)

**한 줄**
> 학습이 끝나면 15 seed mean/std, 핵심 paired t-test, Cohen dz/effect size가 **자동 계산된 표를 받아**, 그 수치를 발표 본문(p17 · p18)에 **어떻게 박을지 정하는** 사람. Holm/ANOVA는 보조·부록 분석으로만 다룬다.

**비유**
> 신문 기자. 통계청이 자동으로 보내준 보도자료(CSV)를 읽고 핵심 한두 문장을 뽑아내서 기사 본문에 박는다. 컴퓨터가 한 계산을 손으로 다시 하지는 않는다. *읽고 해석하고 본문에 꽂는 게 본업.*

**난이도**

| 축 | 점수 | 이유 |
|---|---|---|
| 명령 실행 | ★★☆☆☆ | `./run.sh e2e aggregate` 한 줄. |
| 코드 이해 | ★★☆☆☆ | `pipeline/statistics.py`가 어떤 컬럼을 만드는지만 알면 됨. 내부 수식 안 봐도 OK. |
| 통계 | ★★★☆☆ | t-test · ANOVA가 무엇을 측정하는지는 알아야 본문에 풀어쓸 수 있음. 직접 계산은 X. |
| 글쓰기 | ★★★☆☆ | p17(베이스라인 정의) · p18(평가지표) 본문에 수치 박기. |
| 서버 운영 | ★☆☆☆☆ | 거의 없음. |

**하루가 어떤 모양**
> *D5 오후*: `./run.sh e2e aggregate` → `benchmark_summary.csv` + `paired_tests_holm.csv` + `anova_2way_bert.csv` 열어보기 → **수치 자체는 그대로 신뢰** (코드 검증은 이미 작업 #2에서 끝남) → spot check 한 번: "A_B vs D_B n_pairs가 시드 수랑 일치하나? 평균 차이와 effect size가 어느 정도인가?" → 결과를 p18 한 문장으로 풀어쓰기.
>
> *예시 본문*:
> > *"D_B vs A_B Macro F1 차이는 +0.0XX였고, 같은 seed 기반 paired t-test와 effect size를 함께 볼 때 개선 경향을 확인했다. 여러 조건 비교와 ANOVA는 보조 분석으로 확인했다."*

**"내가 잘할 수 있겠다" 신호**
- 통계학을 한 번이라도 들어봐서 t-test·ANOVA가 무엇을 측정하는지 그림이 잡힘
- Excel/Numbers/CSV 보면서 표 읽는 게 부담 없음
- 수치를 한국어 문장으로 자연스럽게 풀어쓰는 게 가능
- "이 숫자가 본문에서 무슨 뜻을 갖는지" 설명하는 걸 좋아함

**시간 부하 곡선**: D3~D5 ★★★☆☆ (결과 들어올 때 집중) → 그 외 ★★☆☆☆

---

## Stage 3 — XAI Curator (SHAP/LIME 큐레이터)

**한 줄**
> A_B vs D_B 모델 + 8조건 ablation에서 SHAP/LIME이 지목한 토큰이 **텍스트 안에서 의미적으로 맞는지 직접 눈으로 sanity check**하고, 서브워드 집계(BERT `##` / RoBERTa `Ġ`)가 깨지지 않았는지 검수하는 사람.

**비유**
> 박물관 큐레이터. 모델이 "이 단어가 hate 판단의 근거였다"고 지목한 작품(토큰)을 텍스트와 대조하면서 진짜 그런지 한 작품씩 감정. 작품 검증이 본업이지 신작 제작이 아님.

**난이도**

| 축 | 점수 | 이유 |
|---|---|---|
| 명령 실행 | ★★☆☆☆ | `./run.sh e2e xai-primary --resume` 1줄. |
| 코드 이해 | ★★★★☆ | `pipeline/xai.py` + `runtime/experiment_xai.py`의 SHAP·LIME 호출, 서브워드 집계 흐름. 5개 stage 중 코드 부담 가장 큼. |
| 통계 | ★★☆☆☆ | Jaccard / Spearman / paired t 정도. |
| 글쓰기 | ★★★☆☆ | p4·p5 학술 인용(Mathew/Cheng/Lundberg/Ribeiro) + p18 XAI 4축 해설. |
| 서버 운영 | ★☆☆☆☆ | SHAP은 CPU 강제이므로 GPU 거의 안 씀. |

**하루가 어떤 모양**
> *D5 오후*: `outputs/.../xai/.cache/a_b_seed_42.json` 열어서 첫 3개 sample의 SHAP `top_tokens`가 `text` 안에 실제로 등장하는지, 클래스(hate/offensive/normal)와 의미적으로 맞는지 눈으로 확인. BERT WordPiece `##ger`이 단어 단위로 잘 집계됐는지 1~2 sample 점검. → 단톡: "*A_B seed 42 SHAP top-5 sanity OK. 서브워드 집계 정상.*"

**"내가 잘할 수 있겠다" 신호**
- SHAP·LIME 논문(Lundberg 2017, Ribeiro 2016) 한 번이라도 훑어봤거나 호기심 있음
- BERT 토크나이저 구조(`##`, `Ġ`)가 무엇인지 알고 있거나 배우는 게 부담 없음
- 텍스트를 토큰 단위로 뜯어보는 게 재미있음
- 선행연구 인용(p4·p5)이 본업과 직접 연결되는 게 좋음

**시간 부하 곡선**: D3~D7 ★★★☆☆ → 그 외 ★★☆☆☆

---

## Stage 4 — Author (보고서·발표 통합)

**한 줄**
> Stat/XAI 결과가 들어오면 `final_report.md` + 발표 자료 26p를 통합하는 사람. **표 · dashboard 카드 · claim은 자동으로 채워지므로 검수 · 문장 다듬기 · 발표 톤 통일이 본업.**

**비유**
> 잡지 편집장. 기자들(Pilot · Stat · XAI)이 보낸 원고(CSV/JSON)를 받아서 한 호의 흐름을 짜고 표지 · 헤드라인 · 교정을 한다. 본인이 직접 쓰는 분량은 의외로 적고, **톤 통일 · 금지 표현 검수 · 발표 분량 조정**이 핵심.

**난이도**

| 축 | 점수 | 이유 |
|---|---|---|
| 명령 실행 | ★★☆☆☆ | `xai-bundle && report && dashboard` 3줄. |
| 코드 이해 | ★★☆☆☆ | `reporting.py`가 어떤 입력을 어디서 읽는지만 알면 됨. SHAP 내부는 안 봐도 됨. |
| 통계 | ★★☆☆☆ | Stat 결과를 본문에 풀어쓸 수 있을 정도. 직접 계산 안 함. |
| 글쓰기 | ★★★★★ | 5개 stage 중 글쓰기 비중 가장 큼. 26p 분량 + 톤 통일 + 금지 표현 검수. |
| 서버 운영 | ★☆☆☆☆ | 없음. |

**하루가 어떤 모양**
> *D8 오후*: `./run.sh e2e report --run-id v2_15seed` → `final_report.md` 본문에 자동 채워진 Benchmark Summary 표 + paired test 표 + XAI claim 보면서 발표 슬라이드 p19/p24를 평가 문서 와꾸(헤드라인 + 8~15 bullet + 근거 수치 + 시사점)로 풀어쓰기. 금지 표현 "*순환적*", "*피드백 루프*", "*혐오는 단어가 아닌 맥락*" 0건 grep 검수.

**"내가 잘할 수 있겠다" 신호**
- 한국어 보고서 · 논문 글쓰기가 손에 익음
- 다른 사람 원고 합치고 톤 통일하는 데 거부감 없음
- 발표 자료(PPT/Notion/Markdown) 만들어본 경험
- 26p 분량을 두려워하지 않음
- 본인 본업 페이지가 p19 + p22~p26 통합/한계/향후 쪽이면 자연스러움

**시간 부하 곡선**: D0~D5 ★★☆☆☆ (placeholder 검수 + p22~p26 골격) → D7~D10 ★★★★★ (메인 작업)

---

## Stage 5 — QA Conductor (콘서트 엔지니어)

**한 줄**
> 매일 `daily.sh` 한 번 돌려서 **5인 전체 산출물이 깨지지 않게 관제**하고, D3 시점에 Full Run Gate 6조건으로 120 unit 풀 학습 **GO/STOP을 판단**하는 사람. 후반(D7~D10)에는 Author 글쓰기 페어.

**비유**
> 콘서트 음향 콘솔 엔지니어. 본 공연 전 매일 사운드체크 한 번씩 돌리고, 본 공연 시작 전 6개 채널이 다 OK인지 한 번에 판단. 공연 중에는 콘솔 앞에서 페이더 미세 조정.

**난이도**

| 축 | 점수 | 이유 |
|---|---|---|
| 명령 실행 | ★☆☆☆☆ | `./v2/scripts/daily.sh` 1줄. |
| 코드 이해 | ★★☆☆☆ | 어디 깨지면 어느 파일 보는지 정도. 내부 구현은 안 봐도 됨. |
| 통계 | ★☆☆☆☆ | 거의 없음. |
| 글쓰기 | ★★★☆☆ | 단톡 보고 한 줄 매일 + 후반 Author 페어 검수. |
| 서버 운영 | ★★★☆☆ | 실패 unit 진단 (stderr.log 읽기). |

**하루가 어떤 모양**
> *D0~D10 매일 09시 (5~10분)*: 1) `./v2/scripts/daily.sh` 실행, 2) `failed_runs.csv` 행 수 + `completed_runs.csv` 행 수 확인, 3) 단톡 한 줄: "*[D3] preflight OK, failed 0건, completed 8/120, Gate 6/6 통과 → 풀 학습 GO 의견.*"
>
> *D7~D10*: Author와 같이 `final_report.md` 본문 한 번 더 읽고 금지 표현 · 일관 표현 교차 검수.

**"내가 잘할 수 있겠다" 신호**
- 매일 정해진 시간에 짧은 루틴 돌리는 게 가능 (학교 가는 길에 한 번)
- 다른 사람 결과 보고 문제 찾는 거 좋아함 (페어 리뷰 성향)
- 본인 발표 페이지가 가벼워서 보조 시간 여유 있음
- "내가 결정한다"보다 "내가 안전망이다" 역할이 편함

**시간 부하 곡선**: D0~D10 매일 ★★☆☆☆ 상시 + D7~D10 Author 페어 ★★★☆☆

---

## Cheat Sheet — 5장 한 표 비교

| Stage | 명령 | 코드이해 | 통계 | 글쓰기 | 운영 | 시간부하 곡선 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Pilot** | ★★ | ★★★ | ★ | ★★ | **★★★★** | D0~D3 ★★★★ → 이후 ★★ |
| **Stat Auditor** | ★★ | ★★ | ★★★ | ★★★ | ★ | D3~D5 ★★★ |
| **XAI Curator** | ★★ | **★★★★** | ★★ | ★★★ | ★ | D3~D7 ★★★ |
| **Author** | ★★ | ★★ | ★★ | **★★★★★** | ★ | D0~D5 ★★ → D7~D10 ★★★★★ |
| **QA Conductor** | ★ | ★★ | ★ | ★★★ | ★★★ | D0~D10 ★★ 상시 |

---

## 한 줄 선택 가이드

```
서버·실행이 익숙하다       →  Pilot
표·수치 읽고 본문에 박기   →  Stat Auditor
텍스트·논문이 좋다          →  XAI Curator
글쓰기·통합이 좋다          →  Author
꾸준한 루틴·관찰이 좋다     →  QA Conductor
```

---

## 회의 진행 순서 (확정됨 — 본 섹션은 향후 스왑 회의용 보존)

```
[확정 상태] 21 문서 §0 표 그대로 시작. 스왑 의견 없으면 D0 첫 명령 인증으로 직행.

스왑이 필요한 경우 (5분):
[0~1분]   본 문서 §사용법 + 치트시트 같이 화면에 띄우기.
[1~2분]   각자 1순위·2순위 카드 메모 (말 안 하고).
[2~3분]   Author 정함 (글쓰기 부담 큼 → 누가 26p 통합 가능?)
[3~4분]   Pilot 정함 (NVIDIA 서버 접근 가능 인원 중 D0~D3 가용?)
[4~5분]   남은 3명이 Stat/XAI/QA 자기 본업·관심대로 가져감.
[종료]    21 문서 §0 표 5인 이름으로 갱신.
```

---

## 결정된 다음에 할 것 (D0 첫 명령)

확정된 5명은 각자 다음을 D0 안에 한 번 실행해서 단톡에 인증.

```bash
# Pilot — 환경 + AMP 활성 확인
cd v2 && ./run.sh e2e status --run-id v2_15seed
pip install -r runtime/requirements.txt   # statsmodels 포함
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"   # AMP 활성 확인

# Stat Auditor — ANOVA + bootstrap CI
cd v2 && ./run.sh e2e aggregate --run-id v2_15seed
ls outputs/experiments/v2_15seed/benchmark/*.csv
# benchmark_summary.csv의 ci_low/ci_high는 manifest.statistics.bootstrap_iterations>0 이면 bootstrap.
# anova_*.csv는 eta_squared / partial_eta_squared 컬럼 있어야 함.

# XAI Curator — sample 결정성 + 4축
cd v2 && ./run.sh e2e xai-primary --run-id v2_15seed --dry-run
wc -l outputs/experiments/v2_15seed/xai/samples/primary_samples.csv   # 200
head -1 outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv   # 18컬럼 (CI/MSS/IS/AttnEntropy 포함)

# Author — bundle + report 자동 채움 흐름
cd v2 && ./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
head -60 outputs/experiments/v2_15seed/reports/final_report.md   # 5개 새 섹션 placeholder 메시지 확인

# QA Conductor — Gate 자동 점검
cd v2 && ./scripts/daily.sh   # 마지막 줄 "[daily preflight ok — Gate: GO]"
python3 scripts/gate_check.py --run-id v2_15seed --skip-sample-check   # 6/6 PASS
```

D0 인증이 다 들어오면 21 문서 §3 "D0~D10 일정" 그대로 진행 시작.

> ⚠ Stat Auditor의 "수치 검증"은 손계산이 아니라 **자동 계산된 표를 본문에 박는** 역할로 재정의됨.
> ANOVA 효과 크기 해석은 Cohen 기준 (η² < 0.01 negligible / 0.01~0.06 small / 0.06~0.14 medium / ≥ 0.14 large).

---

문서 끝.
