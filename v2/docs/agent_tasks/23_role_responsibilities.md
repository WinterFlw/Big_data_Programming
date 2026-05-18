# 23. 역할별 책임 풀이 (인원 미배정)

> 마지막 업데이트: 2026-05-17
> 5 stage 각각이 **정확히 무엇을 하는지**, **왜 필요한지**, **어떤 코드·산출물에 닿는지**, **다른 stage와 어떻게 인터페이스 하는지**, **어떤 위험이 있고 어떤 안전장치가 있는지**를 한 페이지씩 풀어 쓴 문서.
>
> **담당자 명시 없음.** 22 회의용 카드와 21 실명 매핑 사이의 중간 깊이.
>
> 단일 진실 출처:
> - 카탈로그 (1045줄 깊이): [`19_team_role_tracks.md`](../19_team_role_tracks.md)
> - 회의용 압축 (252줄): [`22_stage_briefs.md`](22_stage_briefs.md)
> - **본 문서 (역할 책임 풀이, 인원 미배정)**: 23번
> - 실명 매핑 (확정안): [`21_team_role_dispatch.md`](21_team_role_dispatch.md)

---

## 사용법

회의 전에 한 번 읽기. 각자 어느 stage가 본인 손에 맞는지 책임의 무게로 가늠한다. 회의에서 22 카드로 카드 고를 때 본 문서를 옆에 두고 "내가 이 책임을 정말 질 수 있나" 한 번 더 확인.

각 stage는 다음 7개 섹션으로 구성:

```
1. 한 줄 정의
2. 무엇을 하는가 (시간순으로 풀어쓰기)
3. 왜 이 역할이 필요한가 (이게 빠지면 무엇이 깨지나)
4. 닿는 코드 / 산출물
5. 다른 stage와의 인터페이스
6. 위험과 안전장치
7. "이 역할이 실패하면 발표가 어디서 막히나"
```

---

## Stage 1 — Pilot (Benchmark 실행자)

### 1. 한 줄 정의

8조건 × 15시드 = 120개 GPU 학습을 NVIDIA 서버에서 **안정적으로 굴려서 끝까지 완료시키는** 역할.

### 2. 무엇을 하는가 (시간순)

**D0 (환경 준비)**
- NVIDIA 서버 접근 권한 확인. `nvidia-smi`로 GPU 사양·메모리·CUDA 버전 점검.
- `pip install -r v2/runtime/requirements.txt` 환경 구축 (statsmodels 포함).
- `data/dataset.json` / `data/post_id_divisions.json` 서버 업로드 확인.

**D1 (시드 결정성 smoke)**
- A_B seed 42 한 번 학습 → `metrics.json`의 `macro_f1` 기록.
- 같은 명령 한 번 더 실행(`--force`) → 두 번째 `macro_f1`.
- 두 값이 ±0.0001 이내 일치 확인. 안 맞으면 `set_seed()` cudnn 적용 점검.

**D2 (paired smoke)**
- A_B + D_B seed 42 2조건 학습 (≈30분 × 2 = 1시간).
- 산출물 6개(metrics/history/run_config/predictions/stdout/checkpoint) 직접 `ls` 확인.
- `predictions.csv` 행 수가 test split 크기와 일치하는지 확인.

**D3 (Full Run Gate 통과 전 대기)**
- QA가 `gate_check.py`로 6조건 자동 점검. 본인은 1, 2, 5번 FAIL 시 fix 책임.
- Gate 통과 → "GO" 단톡 보고 받으면 full 120 unit `--execute --resume` 시작.

**D4~D6 (full 학습 진행 모니터링)**
- 매일 1회 `./run.sh e2e status --run-id v2_15seed`로 unit별 진행 확인.
- 실패한 unit 있으면 `stderr.log` 읽어 fatal 원인 식별 (OOM / NaN loss / 디스크 등).
- 회수 가능하면 `--resume`으로 재시도. 회수 불가능하면 QA에게 보고.

**D7~D10 (학습 종료 후 정리)**
- 모든 unit `completed` 확인. failed unit 0건 보장.
- checkpoint 디렉토리 디스크 사용량 점검 (15시드 × 8조건 ≈ 50GB).
- 학습 시간 epoch별 통계를 정리해 발표 자료 p14 하이퍼파라미터 페이지에 박을 수 있게 준비.

### 3. 왜 이 역할이 필요한가

본 연구의 척추는 "**8조건 × 15시드 paired test로 효과를 측정한다**"이다. 그러려면 120 unit이 같은 sample·같은 split·같은 하이퍼파라미터로 끝까지 학습돼야 한다. 단 한 unit이라도 빠지거나 다른 설정으로 학습되면 paired test의 짝이 무너지고 통계 power가 떨어진다.

또 시드 재현성이 보장돼야 cudnn 결정성 / AMP autocast 설계가 실제로 의미를 갖는다. Pilot이 D1 smoke를 신중히 안 하면 D3 Gate에서 잡히지 않은 비결정성이 D10 발표 직전에 터질 수 있다.

### 4. 닿는 코드 / 산출물

**코드**
- `v2/runtime/experiment_core.py` — 학습 루프, AMP autocast, `train_neural_model()`
- `v2/runtime/utils.py` — `set_seed()` cudnn 결정성
- `v2/pipeline/training_adapter.py` — runtime → v2 산출물 정규화
- `v2/pipeline/runner.py` `benchmark()` — `--execute` / `--resume` / `--force`
- `v2/run.sh e2e benchmark` — CLI 진입점

**산출물 (Pilot이 직접 책임)**
```
outputs/experiments/v2_15seed/
├── benchmark/runs/<cond>/seed_<n>/
│   ├── metrics.json          # 완료 신호
│   ├── history.csv           # epoch별 학습곡선
│   ├── run_config.json       # 사용한 하이퍼파라미터
│   ├── predictions.csv       # 정규화된 sample × class 확률
│   ├── stdout.log / stderr.log
└── benchmark/checkpoints/<cond>_seed_<n>.pt
```

### 5. 다른 stage와의 인터페이스

- **Stat Auditor** ← Pilot의 `metrics.json`이 모든 통계 검정의 입력. paired test 짝이 안 맞으면 Stat이 막힘.
- **XAI Curator** ← Pilot의 `checkpoint` `.pt` 파일을 SHAP/LIME에 로드. 손상 시 XAI 전체 막힘.
- **QA Conductor** ← Pilot의 `failed_runs.csv` 행 수가 Gate 5번 조건. 0건 유지가 GO 조건.
- **Author** ← Pilot이 정리한 학습 시간·하이퍼파라미터가 발표 p14 본문 입력.

### 6. 위험과 안전장치

| 위험 | 안전장치 |
|---|---|
| 한 unit이 OOM으로 학습 중 죽음 | `--resume`로 마지막 epoch부터 재시작 가능. checkpoint가 매 epoch 저장됨. |
| 시드 재현성 깨짐 | 작업 #1 cudnn.deterministic / 작업 #12 AMP가 enabled=False면 fp32 강제. 그래도 NaN 손실 한 번 발생하면 시드 효과 무너짐 → `stderr.log` 검수. |
| 디스크 부족 | 시작 전 `df -h`. 60GB 이상 여유 확보 권장. |
| 학습 시간 너무 김 | AMP가 자동으로 fp16 → ~50% 단축. 그래도 부족하면 batch size 조정 (단 manifest hash 바뀌면 결과 비교 무너짐). |
| 학습 진행 모니터링 안 함 | QA가 매일 daily.sh로 status 자동 점검. Pilot은 본인이 D2에 한 번 더 직접 status. |

### 7. 이 역할이 실패하면

- D3 Gate에 못 들어가면 full 120 시작 X → 모든 후속 stage 정지.
- 일부 unit이 빠지면 paired test n_pairs 감소 → 통계 power 손실, p-value 못 잡음.
- 시드 재현성 실패 → "본 연구의 메인 메시지가 15 seed 결정성"인데 그게 깨지면 발표 핵심 무너짐.

---

## Stage 2 — Stat Auditor (수치 검수자)

### 1. 한 줄 정의

학습이 끝나면 15 seed mean/std, 핵심 A_B vs D_B paired t-test, Cohen dz/effect size, 95% CI가 **자동 계산된 표를 받아**, 그 수치를 발표 본문(p17 · p18)에 **어떻게 박을지 정하는** 역할. Holm/ANOVA/bootstrap은 계산은 하되 보조·부록 분석으로 낮춰서 다룬다.

### 2. 무엇을 하는가 (시간순)

**D0~D2 (빈 입력 검수)**
- `./run.sh e2e aggregate --run-id v2_15seed` 한 번 실행.
- 7개 CSV(benchmark_runs/summary/paired/holm + anova_2way_bert/roberta/3way) 헤더 직접 열어 확인.
- 컬럼명이 [docs/07_output_and_report_contract.md](../07_output_and_report_contract.md)와 일치하는지 점검.

**D3~D4 (smoke 결과 부분 검증)**
- Pilot의 A_B + D_B seed 42 2조건 smoke 들어오면 aggregate 다시.
- `benchmark_runs.csv`에 두 행 들어가는지 확인. `status=completed`인지.
- `paired_tests_holm.csv`는 n_pairs=1이라 통계 의미 없지만 컬럼이 깨지지 않는지만 확인.

**D5 (8조건 × 일부 시드 ANOVA 1차 read)**
- Pilot의 full 120 학습이 부분적으로 들어오면 ANOVA 가능 (4조건 × 1시드 이상).
- `anova_2way_bert.csv`의 `attention_loss` / `vader` / `interaction` 행 → F·p·η² 확인.
- Cohen 기준으로 효과 크기 해석 ("attention_loss η² = 0.6 large").

**D6~D7 (15시드 완료 후 메인 통계)**
- 15시드 paired test 7쌍 (A_B:D_B, A_B:B_B, A_B:C_B, B_B:D_B, C_B:D_B, A_R:D_R, D_B:D_R).
- 각 쌍의 `mean_diff` / `p_value_holm` / `effect_size` / `ci_low/high` 직접 표로 정리.
- bootstrap CI vs t-분포 CI 차이가 의미 있게 다른지 spot check (`manifest.statistics.bootstrap_iterations=1000`).

**D8 (발표 본문 박기)**
- p17 베이스라인 정의: "TF-IDF + LR / SVM은 macro F1 X.XX 한계선, BERT 베이스 A_B는 Y.YY"
- p18 평가지표 + 결과 한 문장:
  > "D_B vs A_B Macro F1 차이는 +0.0XX였고, 같은 seed 기반 paired t-test와 effect size, 95% CI를 함께 고려할 때 개선 경향을 확인했다. 여러 조건 비교와 ANOVA는 보조 분석으로 확인했다."

### 3. 왜 이 역할이 필요한가

코드가 자동으로 계산한 표는 자체로는 발표 본문에 들어갈 수 없다. "**의미를 한 줄로 풀어쓰는**" 사람이 필요하다. 본 연구는 효과가 클 거라는 가설(H2/H3)을 세웠지만, 1차 파이프라인에서 이미 ANOVA p > 0.05가 나온 적이 있다. 15시드에서도 비슷한 결과면 한계점으로 솔직히 보고해야 한다 — 그 판단을 Stat이 직접 본문에 쓴다.

또 ANOVA effect size(작업 #14)와 bootstrap CI(작업 #7)는 단순 p값보다 강한 증거다. "p < 0.05" 한 줄로 끝낼 게 아니라 "η² = 0.5 large + 95% CI [0.04, 0.08]" 처럼 본문에 풀어쓰면 평가위원이 더 설득된다.

### 4. 닿는 코드 / 산출물

**코드**
- `v2/pipeline/statistics.py`
  - `summarize_benchmark()` — mean ± std + bootstrap CI
  - `compute_paired_tests()` — 7쌍 paired t
  - `apply_holm_correction()` — multiple comparison
  - `compute_two_way_anova()` / `compute_three_way_anova()` — η² 포함
- `v2/pipeline/runner.py` `aggregate_stage()` — CLI 진입점

**산출물**
```
outputs/experiments/v2_15seed/benchmark/
├── benchmark_runs.csv         # raw row
├── benchmark_summary.csv      # mean ± std + 95% CI (bootstrap)
├── paired_tests.csv           # 7쌍 × 3 metric raw
├── paired_tests_holm.csv      # adjusted p-value 포함, 보조 확인용
├── anova_2way_bert.csv        # eta_squared + partial_eta_squared
├── anova_2way_roberta.csv
└── anova_3way.csv             # backbone × attention × vader
```

### 5. 다른 stage와의 인터페이스

- **Pilot** → Stat: `metrics.json` 120개가 입력. 빠진 unit 있으면 paired n_pairs 감소.
- **XAI Curator** ↔ Stat: XAI도 `paired_xai_tests.csv`로 paired test. statistics.py의 헬퍼를 재사용.
- **Author** ← Stat: 자동 채워진 markdown 표를 본문에 옮기는 게 본업이지만, Stat의 한 줄 해석이 없으면 본문이 "이 숫자가 의미하는 게 뭔지" 빠진 채로 들어감.
- **QA Conductor** ← Stat: Gate 3번 조건 (aggregate CSV 7개 존재).

### 6. 위험과 안전장치

| 위험 | 안전장치 |
|---|---|
| statsmodels 미설치 환경 | 작업 #6에서 `runtime/requirements.txt`에 명시. 없어도 ANOVA가 빈 CSV로 graceful skip. |
| n_pairs 부족 (시드 일부만 완료) | aggregate가 n_pairs를 컬럼에 박음. < 5면 "n_pairs={n}" 표시 + 한계점에 명시. |
| p-value 해석 오류 | p-value 하나만 보고 결론을 쓰지 않는다. 핵심 A_B vs D_B paired test, 평균 차이, CI, effect size를 함께 보고, Holm은 여러 비교표의 보조 확인값으로 둔다. |
| ANOVA cell 부족 | `_filter_anova_rows()`가 cell unique < 4(2-way) / < 8(3-way) 시 빈 리스트로 fallback. |
| 효과 크기 무시 | 본 문서 §1의 Cohen 기준 표를 발표 본문에 함께 박음. |

### 7. 이 역할이 실패하면

- p18 평가지표 페이지가 "macro_f1 = 0.7" 한 줄로 끝남 → 발표 평가 핵심 빠짐.
- p17 베이스라인 정의에서 TF-IDF 한계 정량화 못 함 → "왜 transformer가 필요한가" 근거 약함.
- Q&A "이 효과 크기가 얼마나 큰가요?"에 답 못 함.

---

## Stage 3 — XAI Curator (SHAP/LIME 큐레이터)

### 1. 한 줄 정의

A_B vs D_B 모델 + 8조건 ablation에서 SHAP/LIME이 지목한 토큰이 **텍스트 안에서 의미적으로 맞는지 직접 눈으로 sanity check**하고, 자동 XAI 4축(CI/MSS/IS/AttnEntropy)의 트렌드를 학술 인용과 함께 해석하는 역할.

### 2. 무엇을 하는가 (시간순)

**D0~D2 (sample 결정성 검증)**
- `./run.sh e2e xai-primary --run-id v2_15seed --dry-run` → status 확인.
- 실제 실행 → `primary_samples.csv` 200행 생성. `deep_samples.csv` 500행, `ablation_samples.csv` 50행.
- `md5sum primary_samples.csv` → 다시 실행 → md5 일치 확인 (seed 무관 결정성).

**D3~D5 (sample selection 학술 근거)**
- 발표 p4·p5에서 인용할 논문 매핑:
  - SHAP: Lundberg & Lee 2017 NeurIPS
  - LIME: Ribeiro 2016 KDD
  - LOO: Li 2016
  - CI: Gini 1912 (분포 불평등 측정)
  - IS: Lundberg 2020 Nature Machine Intelligence
  - Attention Rollout: Abnar & Zuidema 2020 ACL
  - Comp/Suff/MSS: DeYoung 2020 ERASER
- Cheng 2022 Virginia Tech vs 본 연구의 차별점(rationale → 학습 손실 반영) 한 줄 정리.

**D5~D7 (체크포인트 들어오면 메인 XAI)**
- Pilot의 A_B seed 42 checkpoint 들어오면 `xai-primary --resume` 1차 실행.
- `outputs/.../xai/.cache/a_b_seed_42.json` 열어서 첫 3 sample의 SHAP `top_tokens` 확인:
  - 텍스트 안에 실제 등장하는가
  - hate/offensive/normal과 의미적으로 맞는가
  - BERT `##` / RoBERTa `Ġ` 서브워드가 단어 단위로 잘 집계됐는가 (`nig` + `##ger` → `nigger`)
- 한두 sample에서 SHAP top-5가 어색하면 `_aggregate_subword_scores` 호출부 점검.

**D7~D8 (전체 XAI 결과 해석)**
- 15시드 × A_B / D_B 완료 후 `seed_level_metrics.csv` 18컬럼 보고:
  - 1축 Attribution: shap_lime_overlap@5/@10 — 두 방법이 같은 토큰을 가리키나
  - 2축 Faithfulness: Comp ↑ / Suff ↓ — D_B에서 D_B의 top-k가 정말 중요한가
  - 3축 Context Learning: **CI ↓ / IS ↑ / AttnEntropy ↑** — D_B가 단어 의존에서 맥락 의존으로 이동했나
  - 4축 Plausibility: rationale_f1_at_5 — 인간 근거 토큰과 일치하나
- `seed_stability.csv`의 top-k Jaccard / Spearman으로 "같은 sample에 대한 다른 seed의 설명이 얼마나 일관적인가" 확인.

**D8~D10 (발표 본문 작성)**
- p4 관련 연구: SHAP/LIME/Cheng/Mathew 인용 한 줄씩.
- p5 차별성: Mathew(평가만) → Cheng(분석만 + VADER) → 본 연구(학습 손실 반영 + 4축 자동).
- p6 차별성 표를 실제 결과 수치로 갱신.
- p18 평가지표 본문에 4축 12지표 한 줄씩 풀어쓰기.

### 3. 왜 이 역할이 필요한가

XAI 코드가 자동으로 메트릭을 뽑긴 한다. 하지만 **"이 메트릭이 의미적으로 맞는가"는 사람이 토큰을 직접 봐야** 알 수 있다. SHAP이 `top_tokens = ["the", "and", "is"]`를 뽑으면 메트릭 숫자가 좋아 보여도 의미 없다 — 그런 false positive를 잡는 게 본 역할.

또 본 연구의 가장 강력한 학술 기여는 "**자동 XAI 4축으로 인간 라벨 의존을 최소화**"이다. 이를 평가위원에게 설득하려면 12지표가 정확히 어떤 학술 기반을 갖는지 한 줄씩 답할 수 있어야 한다 — XAI Curator의 본업.

### 4. 닿는 코드 / 산출물

**코드**
- `v2/pipeline/xai.py` — adapter, `_compute_unit_metrics()` 4축 계산
- `v2/pipeline/xai_sampling.py` — seed 무관 sample 선택
- `v2/runtime/experiment_xai.py` — SHAP/LIME 실제 호출 (수정 금지)

**산출물**
```
outputs/experiments/v2_15seed/xai/
├── samples/{primary,deep,ablation}_samples.csv      # seed 무관 결정적
├── primary/
│   ├── seed_level_metrics.csv          # 18컬럼 (4축 모두)
│   ├── sample_level_metrics.csv        # subgroup 분해 입력
│   ├── paired_xai_tests.csv            # A_B vs D_B 메트릭별
│   └── seed_stability.csv              # top-k Jaccard / Spearman
├── deep/case_summary.csv               # median seed × 500 sample
├── ablation/xai_ablation_metrics.csv   # 11컬럼 (4축 ablation)
├── .cache/                              # SHAP/LIME attribution 캐시
└── xai_summary.json
```

### 5. 다른 stage와의 인터페이스

- **Pilot** → XAI: `.pt` checkpoint 입력. 손상 시 XAI 전체 정지.
- **Stat Auditor** ↔ XAI: `paired_xai_tests.csv`가 statistics.py 패턴 재사용. ANOVA effect size 해석법 공유.
- **Author** ← XAI: token sanity check 결과 + 12지표 학술 인용이 p4/p5/p18 본문 입력.
- **QA Conductor** ← XAI: Gate 4번 조건 (sample 결정성 md5 일치).

### 6. 위험과 안전장치

| 위험 | 안전장치 |
|---|---|
| SHAP CUDA 메모리 누수 | runtime이 CPU 강제. `bundle.model.to("cpu")` 패턴 유지. |
| sample이 seed에 의존 | xai_sampling.py에 numpy seed=0 고정. md5 일치 검사. |
| cache 형식 변경 후 stale | 작업 #11에서 `.cache/` schema 발전 이력 + 정리 명령 명시. |
| 서브워드 집계 실패 | runtime의 `_aggregate_subword_scores`가 BERT `##` / RoBERTa `Ġ` 자동 분기. 첫 sample sanity로 검증. |
| IS 계산 시간 과다 | `xai_interaction_pairs = 50` cap. 200 sample × 50 pair = 1만 forward (수십 분 내). |
| Attention Rollout 실패 | try/except로 빈 리스트 fallback. 결과적으로 entropy 컬럼 빈 값. |

### 7. 이 역할이 실패하면

- p4·p5 학술 인용 약함 → 평가위원이 "이 SHAP이 학술적으로 어떻게 다른가" 질문에 못 답함.
- p18 4축 메트릭 해설 빠짐 → 본 연구의 결정 카드(3축 Context Learning) 실종.
- "단어 의존 → 맥락 의존"이라는 척추 메시지 검증 불가.

---

## Stage 4 — Author (보고서·발표 통합자)

### 1. 한 줄 정의

Stat/XAI/Pilot 결과가 들어오면 `final_report.md` + 발표 자료 26p를 통합하는 사람. **표·dashboard 카드·claim은 자동으로 채워지므로 검수·문장 다듬기·발표 톤 통일이 본업.**

### 2. 무엇을 하는가 (시간순)

**D0~D2 (placeholder 검수)**
- `./run.sh e2e xai-bundle && report && dashboard` 한 번 실행.
- `final_report.md` 본문에서 5개 새 섹션 placeholder 메시지 확인:
  - "_no benchmark results yet — populate by running ..._"
  - "_no paired tests yet for metric `macro_f1`_"
  - "_no XAI evidence yet — populate by running ..._"
  - "_no seed stability rows yet_"
  - "_no automatic limitations recorded yet_"
- 메시지가 정확히 표시되면 자동 채움 로직이 살아있는 것.

**D3~D6 (외부 결과 의존 없는 페이지 골격)**
- 발표 26p 중 본인 페이지(p19, p22~p26) 골격을 markdown으로 미리 작성:
  - p22 향후 일정: 발표_와꾸_v2.md §22 그대로
  - p23 팀원별 역할: 21 문서 §0 표 그대로
  - p25 진행 중 문제점: 19 문서 §6 + `xai_risk_flags.csv` 예상 항목
  - p26 향후 연구: 발표_와꾸_v2.md §26 그대로

**D7~D8 (결과 통합)**
- Stat의 ANOVA 결과 + XAI Curator의 4축 트렌드 들어오면 자동 채워진 표 검수.
- p19 메인 결과 표:
  - benchmark_summary.csv → "조건별 Macro F1 mean ± std + 95% CI"
  - paired_tests_holm.csv → "A_B vs D_B ΔF1 + paired p + effect size, adjusted p는 보조"
  - xai_dashboard_bundle.json → "claim 1: D_B is more aligned with human rationale..."
- p24 결과 요약 한 페이지: 발표용 한 페이지 와꾸로 풀어쓰기.

**D8~D9 (발표 자료 통합)**
- markdown → pptx 변환 (또는 직접 작성). 평가 문서 와꾸 (헤드라인 + 8~15 bullet + 근거 수치 + 시사점).
- 금지 표현 grep 검수: `grep -E "순환적|피드백 루프|혐오는 단어가 아닌|XAI 진단 결과" *.md`. 0건이어야.
- 일관 표현 검수: "과학적 검증 프레임워크", "전이학습 기반 full fine-tuning", "VADER는 Cheng (2022) 선행연구 기반 사전 가설", "단어 단서뿐 아니라 맥락 단서까지 함께 학습".

**D10 (리허설)**
- 26p 통합 1회독. 척추 메시지 모든 슬라이드에서 통일.
- QA Conductor와 함께 본문 한 번 더 읽고 교차 검수.

### 3. 왜 이 역할이 필요한가

코드는 표를 자동으로 채우지만 "**26p 분량 한국어 보고서를 통일된 톤으로 묶는**" 것은 자동화 불가능. 척추 메시지("단어 + 맥락 함께 학습") 가 모든 슬라이드에서 같은 표현으로 반복돼야 발표가 일관적이다.

또 본 연구는 1차 파이프라인에서 ANOVA p > 0.05를 경험한 적이 있다. 한계점을 솔직히 보고하는 톤(p25)을 Author가 잡지 않으면 발표가 과장된 느낌으로 흐를 수 있다.

### 4. 닿는 코드 / 산출물

**코드**
- `v2/pipeline/xai_bundle.py` — 15개 bundle 파일, claim 자동 생성
- `v2/pipeline/reporting.py` — markdown/docx + dashboard HTML 자동 렌더

**산출물**
```
outputs/experiments/v2_15seed/
├── xai/evidence_bundle/ (15 파일)
│   ├── xai_claims.json                 # statistically confirmed only
│   ├── xai_dashboard_bundle.json
│   ├── xai_interpretation_cards.json
│   ├── subgroup_xai_metrics.csv        # source × target 진짜 분해
│   ├── context_metrics.csv             # window/sensitivity
│   ├── token_attributions.jsonl        # .cache 평탄화
│   └── ...
├── reports/
│   ├── final_report.md                 # 자동 표 채움
│   └── final_report.docx
└── dashboard/index.html                # benchmark + XAI summary cards
```

### 5. 다른 stage와의 인터페이스

- **Pilot** → Author: 학습 시간·하이퍼파라미터가 p14 본문 입력.
- **Stat Auditor** → Author: ANOVA 한 줄 해석이 p17·p18 본문 입력.
- **XAI Curator** → Author: 12지표 학술 인용이 p4·p5 본문 입력. 토큰 sanity 결과가 p19 본문 입력.
- **QA Conductor** ↔ Author: D7~D10 글쓰기 페어. 척추 메시지·금지 표현 교차 검수.

### 6. 위험과 안전장치

| 위험 | 안전장치 |
|---|---|
| Stat/XAI 입력이 D7까지 없음 | placeholder 메시지로 자동 fallback. D5~D6 손이 비면 p22~p26 골격 미리 작성. |
| 금지 표현 누락 | `grep -E "순환적|이분법|XAI 진단 결과"` 자동 검수. |
| docx 깨짐 | reporting.py가 minimal Word 양식 (python-docx 없이). 한글 폰트 없으면 영문 fallback. |
| claim 출처 누락 | xai_claims.json 스키마에 `source_artifacts` 필드 필수. 코드가 강제. |
| dashboard 정적 HTML 한계 | benchmark + XAI summary cards 두 표만 자동 렌더. 검색/필터는 미구현. |

### 7. 이 역할이 실패하면

- 26p가 5인 5톤으로 분리됨 → 발표가 일관적이지 않음.
- p25 한계점 약함 → "ANOVA p > 0.05인데 왜 효과 있다고 주장?"에 답 못 함.
- 척추 메시지("단어 + 맥락 함께 학습")가 슬라이드마다 다르게 표현됨.

---

## Stage 5 — QA Conductor (콘서트 엔지니어 + Author 페어)

### 1. 한 줄 정의

매일 `daily.sh` 한 번 돌려서 **5인 전체 산출물이 깨지지 않게 관제**하고, D3 시점에 `gate_check.py`로 Full Run Gate 6조건 **GO/STOP 최종 결정**하는 역할. 후반(D7~D10)에는 Author 글쓰기 페어로 들어감.

### 2. 무엇을 하는가 (시간순)

**D0~D2 (매일 preflight 정착)**
- 아침 09시(혹은 학교 가는 길)에 `./v2/scripts/daily.sh` 1회 실행 (5~10분).
- 단톡에 한 줄 보고: "*[D0] preflight OK, failed 0건, completed 0/120, Gate 5/6 통과 (sample-check skip)*"
- 회귀 발생 시(컴파일 오류 / CSV 헤더 변경 등) 즉시 stderr.log 공유 + 해당 stage owner 멘션.

**D3 (Full Run Gate 최종 결정 ★)**
- Pilot의 A_B + D_B smoke 완료 직후 `gate_check.py --skip-sample-check` 단독 실행.
- 6조건 모두 PASS 확인:
  1. cudnn 결정성 (정수현 fix 책임)
  2. 8조건 metadata 정합성 (정수현)
  3. aggregate CSV 7개 (박종화)
  4. XAI sample 결정성 (차종민)
  5. failed_runs 0건 (정수현)
  6. 산출물 contract 8개 (조은)
- **GO**: 단톡에 "*Gate 6/6 PASS → full 120 unit 학습 GO*" + Pilot 명령 실행 권한.
- **STOP**: FAIL 조건 + fix 책임자 멘션. 통과할 때까지 Pilot 대기.

**D4~D6 (full 학습 모니터링)**
- 매일 daily.sh + 매시간 `./run.sh e2e status` 정도로 unit 진행률 확인.
- failed_runs.csv 행 수 변화 추적. 1건이라도 늘면 Pilot에게 즉시 단톡.
- 서버 사용 시간대 조정 (다른 사람 작업과 conflict 회피).

**D7~D10 (Author 페어 글쓰기 보조 ★)**
- Author와 함께 final_report.md 본문 한 번 더 읽고 교차 검수.
- 척추 메시지 일관성: "단어 + 맥락 함께 학습" 표현 통일 grep.
- 금지 표현 0건 검수: "순환적", "피드백 루프", "혐오는 단어가 아닌 맥락", "XAI 진단 결과".
- 일관 표현 통일: "과학적 검증 프레임워크", "전이학습 기반 full fine-tuning".
- 데이터 split hash 검증: split_profile.json이 변경 없는지 확인 (15시드 전체에 같은 split 보장).

### 3. 왜 이 역할이 필요한가

다른 4명은 각자 본인 stage에 깊이 들어가 있어서 "**전체가 깨지지 않는지 매일 한 번 보는 사람**"이 따로 있어야 한다. CSV 헤더 변경 / 컴파일 실패 / 의존성 누락 같은 문제는 누군가 매일 한 번 ls + 실행해야 잡힌다.

또 D3 Gate는 본 연구의 critical path 결정점이다. "**감정적으로 GO 하고 싶은 압박**"에서 객관적 6조건 자동 점검으로 결정하려면 Pilot이 아닌 제3자가 판단해야 한다.

### 4. 닿는 코드 / 산출물

**코드**
- `v2/scripts/daily.sh` — 매일 preflight 묶음
- `v2/scripts/gate_check.py` — Full Run Gate 6조건 자동 (작업 #13)
- `v2/pipeline/artifacts.py` — failed/completed CSV 분리 (작업 #3)

**산출물**
```
outputs/experiments/v2_15seed/
├── execution_status.csv         # 매일 갱신
├── failed_runs.csv              # QA 매일 모니터
└── completed_runs.csv           # QA 매일 모니터

stdout (단톡 보고용):
[daily preflight ok — Gate: GO]
```

### 5. 다른 stage와의 인터페이스

- **Pilot** ↔ QA: failed_runs.csv 0건 보장. Gate 5번 조건. failed 발생 시 stderr.log 즉시 공유.
- **Stat / XAI / Author** ← QA: 매일 daily.sh가 모든 stage의 빈 입력 무회귀 보장. 한 stage라도 깨지면 즉시 멘션.
- **Author** ↔ QA: D7~D10 글쓰기 페어. 척추 메시지·금지 표현 교차 검수.

### 6. 위험과 안전장치

| 위험 | 안전장치 |
|---|---|
| daily.sh를 매일 안 돌림 | alias로 등록 + 단톡 보고를 매일 의무화. 한 번 빠지면 회귀가 며칠 후 발견됨. |
| Gate에서 객관성 잃음 | gate_check.py가 exit 0/1로 자동 결정. 사람 판단 X. |
| failed unit이 누적 | Pilot에게 즉시 단톡. resume으로 회수 가능한지 stderr.log 진단. |
| Author 페어 시간 부족 | 본업(p7/p8) 분량이 2p로 가장 가벼우니 D7~D10에 시간 확보 가능. |
| Gate 4번 (sample 결정성) skip | daily.sh에서는 비용 회피로 skip. Pilot이 D3 별도로 직접 검사. |

### 7. 이 역할이 실패하면

- daily.sh 안 돌리면 회귀가 며칠 후 발견 → 모든 stage가 영향받은 채로 D7 도달.
- Gate 결정을 Pilot이 직접 함 → "본인 결과가 GO 하길 원함" 편향 들어감.
- p25 한계점 + 척추 메시지 교차 검수 없음 → 발표 톤 흔들림.

---

## 부록 A — 5 Stage 페어/의존 그래프

```
       D0~D3        D3       D4~D7       D7~D10
       ─────       ───       ─────       ──────
       Pilot ─────┬─→ full ─→ Pilot ────→ Pilot
                  │ 학습      모니터       정리
                  │
       QA ────────┴─→ Gate ─→ QA 매일 ──→ QA + Author
       매일                   모니터       글쓰기 페어

       Stat ──────────────→ Stat 검수 ─→ Stat 본문
                            (aggregate)   (p17/p18)

       XAI ──────────────→ XAI 검수 ──→ XAI 본문
                            (xai-primary) (p4/p5/p18)

       Author ─────────────────────────→ Author 통합
                                          (26p 발표)
```

핵심 critical path: **Pilot D3 Gate → 다른 4명이 동시 시동**.

---

## 부록 B — "내가 이 역할이라면 회의 끝나고 30분 안에 할 것"

### Pilot
```bash
ssh nvidia-server
nvidia-smi   # GPU 사양 메모
df -h        # 디스크 여유 60GB+ 확인
cd /path/to/Big_data_Programming
pip install -r v2/runtime/requirements.txt
cd v2 && ./run.sh e2e status --run-id v2_15seed
```

### Stat Auditor
```bash
cd v2 && ./run.sh e2e aggregate --run-id v2_15seed
ls outputs/experiments/v2_15seed/benchmark/*.csv
# 모든 CSV 헤더 직접 열어보고 컬럼 매핑 메모
column -t -s, outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv | head
```

### XAI Curator
```bash
cd v2 && ./run.sh e2e xai-primary --run-id v2_15seed --dry-run
./run.sh e2e xai-primary --run-id v2_15seed
md5sum outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
wc -l outputs/experiments/v2_15seed/xai/samples/*.csv
# 200 / 500 / 50 row 확인
```

### Author
```bash
cd v2 && ./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
head -80 outputs/experiments/v2_15seed/reports/final_report.md
# placeholder 메시지 5개 확인
```

### QA Conductor
```bash
cd v2 && ./scripts/daily.sh
# "[daily preflight ok — Gate: GO]" 확인
python3 scripts/gate_check.py --run-id v2_15seed --skip-sample-check
# 6/6 PASS, exit 0 확인
```

---

문서 끝.
