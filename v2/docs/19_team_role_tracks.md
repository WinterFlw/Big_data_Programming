# 19. Team Role Tracks (현행 — 사람 작업 중심)

> 코드베이스는 거의 갖춰져 있다. 사람은 코드 신규 작성이 아니라 **검증·실행·판단·글쓰기·운영**을 한다.
> 5명을 5개 **stage 오너**로 묶고, 각자 자기 stage의 AI 에이전트를 매니저처럼 굴린다.
>
> 인원 미배정. 회의에서 카드 보고 본인이 가져갈 것 선택.
> 각 카드는 self-contained. 자기 카드만 통째로 AI 에이전트에 던지면 즉시 작업 시작 가능.

---

## 0. 분배 원칙

```
AI 에이전트가 한다  : 코드 신규 작성 / 디버깅 / 패치 / 리팩토링 / 단위 테스트 작성
사람이 한다         : 명령 실행 / 결과 검수 / 숫자 검증 / 판단 / 글쓰기 / 단톡 보고
```

5명 모두 자기 stage의 \"AI 매니저\" 역할을 가진다. 코드 직접 작성 시간은 거의 없다.

---

## 0.1 5 stage 한눈에 비교

| Stage | 한 줄 정의 | 주당 시간 | AI 지시 강도 |
|---|---|---|---|
| **1. Benchmark** | 8조건 × 15시드 = 120개 학습을 서버에서 안정 실행 | 4~6h | 중 |
| **2. Statistics** | 학습 결과 metrics.json 120개 → paired test + Holm + Cohen's d CSV | 3~5h | 중 |
| **3. XAI Core** | A_B vs D_B 모델 + 8조건 ablation → SHAP/LIME/메트릭 CSV | 4~6h | 중-높 |
| **4. XAI Bundle + Report** | XAI/Stat 결과 → evidence bundle + 최종 보고서 + 발표 자료 | 7~10h | 중 (글쓰기 ↑) |
| **5. QA + Server + Author 페어** | 매일 preflight + Gate 판단 + 서버 운영 + Author 글쓰기 보조 | 5~7h | 낮 (사람 판단 ↑) |

총합: 약 25~35h × 4주 = 한 명당 평균 25~35h 학기 작업량.

---

## 0.2 의존 흐름

```
[1. Benchmark] ──┬──→ [2. Statistics] ─┐
                 │                       ├──→ [4. XAI Bundle + Report]
                 └──→ [3. XAI Core] ────┘
                                          
[5. QA + Server] ──→ 전 stage 옆에서 운영 + Gate
                  └─→ [4]의 Author 페어 (글쓰기 보조)
```

- 1번이 D3까지 smoke 통과해야 2,3번이 본격 작업 시작.
- 4번은 2,3번 결과를 받음. 그 전까지는 placeholder + 글쓰기 골격.
- 5번은 D0부터 D10까지 상시 — 매일 preflight + 후반에 Author 페어.

---

## 0.3 학교 평가용 \"내가 한 일\" 1문장

각자 평가서나 발표 Q&A에서 즉답 가능:

```
1. Benchmark: "서버에서 8조건 × 15시드 학습을 직접 실행하고 smoke 검증으로
              full run gate를 통과시켰습니다."

2. Statistics: "15시드 paired t-test / Holm 보정 / Cohen's d 결과를
                CSV로 정리하고 손계산으로 교차 검증했습니다."

3. XAI Core: "SHAP/LIME 출력을 토큰 단위로 점검하고 BERT/RoBERTa 서브워드
              집계가 정확히 동작하는지 검증했습니다."

4. XAI Bundle + Report: "evidence bundle 15개 산출물 + 최종 보고서 + 발표 자료를
                         작성하고 TF-IDF baseline 대비 강점 프레이밍을 책임졌습니다."

5. QA + Server: "전체 일정 + Full Run Gate 판단 + 서버 시간 관리를 책임지고
                 Author와 글쓰기 페어로 검수했습니다."
```

---

## 0.4 정하는 방법 (회의 5분)

순서:
1. 5개 카드 같이 훑기.
2. 가장 무거운 **카드 4 (XAI Bundle + Report)** 먼저 정하기. 글쓰기 부담 큼.
3. 카드 4의 페어인 **카드 5 (QA + Server)** 정하기. 4번과 호흡 잘 맞는 사람.
4. **카드 1 (Benchmark)** 정하기. 서버 운영/CLI 익숙한 사람.
5. 남은 **2, 3** 둘은 흥미·배경 기준으로 자유 선택.

추천 매칭:
- 카드 4 ↔ 카드 5: 페어 — 호흡 잘 맞을 두 명.
- 카드 1: 한 사람 단독 — D3까지 가장 시급.
- 카드 2, 3: 평행 — 서로 영향 적음.

---

## 0.5 공통 원칙 (5명 다 지킬 것)

```
산출물은 v2/outputs/experiments/v2_15seed/ 안에만.
v1/ 폴더는 read-only archive. 절대 import 금지.
자기 stage 밖 파일 수정은 AI 에이전트에 시키지 말 것 (다른 사람 영역 침범).
full benchmark / full XAI는 카드 5의 승인 전 절대 실행 금지.
모든 stage는 resume 가능해야 함 (실패 unit만 재실행).
매일 단톡에 1줄 진행 보고.
```

---
---

# 카드 1 — Benchmark Stage

## 이 stage가 뭔지

8조건 × 15시드 = 120개 학습을 서버에서 안정적으로 돌리는 stage.
명령 진입점: `./run.sh e2e benchmark --execute`

**산출물**: 120개 run 디렉토리 + 120개 checkpoint.

## 사람이 하는 일 vs AI에 시키는 일

| 사람이 직접 (너) | AI 에이전트에 시킬 일 |
|---|---|
| 서버 SSH 접속 + `nvidia-smi` 확인 | `training_adapter.py` 패치 작성 |
| `./run.sh e2e benchmark --dry-run` 실행 | runtime 코드 NVIDIA 호환 수정 (device/AMP/cudnn) |
| `--execute` 직접 실행, GPU 사용량 모니터링 | resume 로직 / failed unit 재실행 로직 구현 |
| 산출물 6개 파일 `ls`로 직접 확인 | 디버깅 / 에러 메시지 해석 |
| 실패 run 발견 시 \"재실행 vs 보고\" 판단 | checkpoint 경로 정규화 코드 |
| 단톡에 \"[smoke ok] A_B seed 42\" 1줄 보고 | unit test 작성 |

너의 비중: AI가 코드 짜고 → 너는 \"진짜 돌아가는지\" 확인하는 사람.

## 첫 주 D0~D3 체크리스트 (행동 중심)

- [ ] **D0**: AI 에이전트에 이 카드의 \"에이전트 첫 문장\" 던지기. 받은 PR 검수.
- [ ] **D0**: 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/01_benchmark_agent.md`
  - `v2/docs/02_e2e_pipeline.md`
- [ ] **D1**: `./run.sh e2e --help` / `status` 직접 실행. 깨지면 에이전트에 로그 던지고 패치 요청.
- [ ] **D2**: dry-run 실행 → AI가 생성한 패치로 깨짐 잡기.
- [ ] **D2**: 서버 접속 + `nvidia-smi` + `python3 -c \"import torch; print(torch.cuda.is_available())\"` 확인.
- [ ] **D3**: A_B seed 42 single smoke `--execute`. 6개 산출물 파일 `ls`로 손 확인.
- [ ] **D3**: 단톡에 `[smoke ok] A_B seed 42 / metrics.json / history.csv / ...` 보고.

## 완료 기준 = 산출물 검수

A_B seed 42 smoke 후 아래 6개 파일이 직접 `ls`로 보여야 함:
```
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/run_config.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stdout.log
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stderr.log
outputs/experiments/v2_15seed/benchmark/checkpoints/a_b_seed_42.pt
```

`metrics.json`을 직접 열어서 macro_f1 값이 0.5~0.8 사이에 있는지 sanity check.

## 너가 직접 돌릴 명령 (복붙)

```bash
cd v2

# D2 dry-run
./run.sh e2e --help
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run

# D2 NVIDIA 환경 확인
nvidia-smi
python3 -c \"import torch; print('cuda:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())\"

# D3 smoke (서버에서, venv 활성화 후)
PYTHON_BIN=/path/to/venv/bin/python \\
  ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume

# 산출물 확인
ls -la outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/
cat outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json | python3 -m json.tool
```

## 막힐 때

- **페어**: 카드 5 (QA/Server) — manifest/CLI 의존. 카드 5가 매일 preflight 돌리니까 너가 깨면 카드 5가 먼저 잡아냄.
- **금지**: full 120 benchmark 임의 실행. 카드 5의 Gate 통과 후에만.
- **금지**: condition별 hyperparameter 임의 변경. config 값 그대로.

## 에이전트에 던질 첫 문장 (그대로 복사)

```
당신은 HateSpeachStudy v2_15seed 파이프라인의 Benchmark Stage 담당 에이전트입니다.

상황:
나는 사람이고, 코드는 거의 다 작성되어 있다. 내 역할은 명령을 직접 실행하고
결과를 검수하는 것. 당신은 내가 명령을 깨끗하게 돌릴 수 있도록 코드 패치 /
디버깅 / 환경 호환 수정을 책임진다.

목표:
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
명령이 NVIDIA CUDA 환경에서 깨지지 않고 학습 1개를 돌려, run_id 내부에
metrics.json/history.csv/run_config.json/stdout.log/stderr.log/checkpoint를
정확히 생성하게 만드는 것.

원칙:
- 실행 기준 코드는 v2/runtime과 v2/pipeline. v1은 archive — 절대 import 금지.
- 모든 산출물은 outputs/experiments/v2_15seed/ 아래.
- 자기 담당 파일 밖을 수정해야 하면 먼저 이유와 변경 범위를 명시.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/01_benchmark_agent.md
v2/docs/02_e2e_pipeline.md
v2/docs/15_runtime_code_validation_matrix.md (1번, 2번 섹션)

수정 가능 파일:
v2/pipeline/runner.py
v2/pipeline/training_adapter.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py
v2/runtime/utils.py (device/seed 관련만)

수정 금지:
v2/pipeline/statistics.py / xai.py / xai_bundle.py / reporting.py
v1/

작업 순서:
1. training_adapter가 v1 import하지 않는지 확인.
2. NVIDIA 환경: get_device()가 cuda 우선, MPS fallback. set_seed에
   cudnn.deterministic=True, benchmark=False. AMP는 cuda일 때만 활성화.
3. dry-run 명령이 깨지지 않게 패치.
4. A_B seed 42 단일 smoke가 6개 산출물을 생성하도록 검증.

완료 시 변경 파일 목록 / 실행한 명령 / 생성된 산출물 경로 모두 보고.
```

## 금지 / 권장 표현

- 금지: \"full 돌려봤는데 결과가 안 나옴\" (smoke 없이 full 절대 X)
- 권장: \"smoke 통과 → Gate 6조건 확인 → full 시작\"

## 완료 보고 양식

```
[v2 작업 완료 - Benchmark]
수정한 파일: (AI가 만든 PR 목록)
직접 실행한 명령:
생성/확인한 산출물:
통과한 검증: smoke ok / 6개 파일 생성 / NVIDIA cuda 동작
남은 위험: (예: D_B 조건에서 attention loss 적용 검증 아직)
다음 사람이 이어받을 부분: (카드 2가 metrics.json 읽고 aggregate 시작)
```

---
---

# 카드 2 — Statistics Stage

## 이 stage가 뭔지

카드 1이 생성한 metrics.json 120개를 읽어 통계 결과 CSV 4개를 만드는 stage.
명령 진입점: `./run.sh e2e aggregate`

**산출물**: paired test + Holm 보정 + 95% CI + Cohen's d 컬럼이 들어간 CSV.

## 사람이 하는 일 vs AI에 시키는 일

| 사람이 직접 (너) | AI 에이전트에 시킬 일 |
|---|---|
| `./run.sh e2e aggregate` 실행 | `statistics.py` 함수 구현 |
| `benchmark_summary.csv` 열어서 숫자 손 점검 | paired t-test / Wilcoxon fallback 알고리즘 |
| 한두 쌍의 paired test를 손계산으로 교차 검증 | Holm-Bonferroni 보정 코드 |
| `paired_tests_holm.csv` 7쌍 비교 모두 들어 있는지 확인 | Cohen's dz 계산 |
| 이상치 발견 시 카드 1에 \"이 run 좀 봐\" 보고 | 빈 결과 / 부분 결과 처리 |
| 결과 해석 1문단 작성 → 카드 4에 전달 | unit test 작성 (fake metrics) |

너의 비중: 통계 결과가 \"말이 되는지\" 직접 눈으로 확인하는 사람.

## 첫 주 D0~D3 체크리스트

- [ ] **D0**: AI에 카드 2 첫 문장 던지기. PR 검수.
- [ ] **D0**: 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/02_statistics_agent.md`
  - `v2/docs/03_validation_and_statistics.md`
  - `v2/docs/07_output_and_report_contract.md` (스키마 컬럼)
- [ ] **D1**: AI가 fake metrics로 paired test smoke 함수 구현하도록 지시.
- [ ] **D1**: 아래 PY smoke 직접 돌려서 `paired_statistics_smoke: ok` 확인.
- [ ] **D2**: `./run.sh e2e aggregate` 빈 입력에서 깨지지 않는지 확인.
- [ ] **D3**: 카드 1의 smoke 결과 들어오면 실제 paired row 1개 생성 확인.

## 완료 기준 = 산출물 검수

7개 CSV 파일 직접 열어서 컬럼 + 값 확인:

```
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv     # bootstrap CI 컬럼 (작업 #7)
outputs/experiments/v2_15seed/benchmark/paired_tests.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
outputs/experiments/v2_15seed/benchmark/anova_2way_bert.csv       # eta²/partial η² (작업 #2 + #14)
outputs/experiments/v2_15seed/benchmark/anova_2way_roberta.csv
outputs/experiments/v2_15seed/benchmark/anova_3way.csv
```

`paired_tests_holm.csv` 직접 열어서 7쌍 비교 모두 row 있는지 확인:
```
A_B:D_B, A_B:B_B, A_B:C_B, B_B:D_B, C_B:D_B, A_R:D_R, D_B:D_R
```

수치 검증: 자동 계산을 그대로 신뢰. spot check 한 줄 — "n_pairs가 시드 수랑 일치하나" 정도.
효과 크기 해석: ANOVA η² Cohen 기준 (< 0.01 negligible / 0.01~0.06 small / 0.06~0.14 medium / ≥ 0.14 large).

## 너가 직접 돌릴 명령 (복붙)

```bash
cd v2

# D1 fake metrics 통계 smoke
python3 - <<'PY'
from pipeline.statistics import compute_paired_tests, apply_holm_correction
manifest = {'statistics': {'paired_tests': ['A_B:D_B']}}
rows = []
for seed, a, d in [(42, .60, .65), (52, .61, .66), (62, .62, .67)]:
    rows.append({'condition':'A_B','seed':seed,'status':'completed',
                 'macro_f1':a,'accuracy':a,'weighted_f1':a})
    rows.append({'condition':'D_B','seed':seed,'status':'completed',
                 'macro_f1':d,'accuracy':d,'weighted_f1':d})
paired = compute_paired_tests(manifest, rows)
corrected = apply_holm_correction(paired)
assert paired[0]['n_pairs'] == 3
assert round(float(paired[0]['mean_diff']), 4) == 0.05
assert corrected[0]['p_value_holm'] != ''
print('paired_statistics_smoke: ok')
PY

# D2 빈 입력 aggregate
./run.sh e2e aggregate --run-id v2_15seed

# D3 실제 결과 확인
column -t -s, outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv | head -20
```

## 막힐 때

- **페어**: 카드 1 (metrics.json 입력). 카드 1이 산출물 안 떨어뜨리면 너 작업 못 시작.
- **페어**: 카드 4 (출력 컬럼 의존). 카드 4가 너의 CSV를 읽고 보고서 row 만듦. 컬럼 빠지면 카드 4 깨짐.

## 에이전트에 던질 첫 문장

```
당신은 HateSpeachStudy v2_15seed 파이프라인의 Statistics Stage 담당 에이전트입니다.

상황:
나는 사람이고 통계 결과 CSV를 직접 열어서 숫자가 말이 되는지 검증하는 역할이다.
당신은 statistics.py / schema.py 구현과 unit test 작성을 책임진다.

목표:
outputs/experiments/v2_15seed/benchmark/runs/ 아래 metrics.json을 읽어
benchmark_summary.csv / paired_tests.csv / paired_tests_holm.csv 를 생성.

원칙:
- same-seed paired comparison만 인정. 다른 seed끼리 비교 금지.
- p-value만 보고하지 말 것 — mean_diff, 95% CI, Cohen's dz, Holm 보정 p
  모두 함께 출력.
- 빈 결과/부분 결과에서도 aggregate 안 깨지게.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/02_statistics_agent.md
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md

수정 가능 파일:
v2/pipeline/statistics.py
v2/pipeline/schema.py

수정 금지:
v2/runtime/*  /  v2/pipeline/xai*.py  /  v2/pipeline/reporting.py

필수 컬럼:
comparison, metric, n_pairs, mean_diff, std_diff, ci_low, ci_high,
test_name, p_value, p_value_holm, effect_size, significant_0_05

필수 paired 비교 7쌍:
A_B:D_B, A_B:B_B, A_B:C_B, B_B:D_B, C_B:D_B, A_R:D_R, D_B:D_R

작업 순서:
1. fake metrics로 paired_statistics_smoke unit test 통과 (시드 3개 가정).
2. 빈 입력 aggregate 안 깨지는지 확인.
3. 카드 1의 smoke 결과가 들어오면 paired row 생성 확인.

해석 문장은 작성하지 말 것 — 카드 4의 영역.
```

## 금지 / 권장 표현

- 금지: \"p-value가 작으니 모델이 좋다\" (해석은 카드 4 영역)
- 금지: 단일 시드 결과로 결론
- 권장: \"7쌍 paired test row 생성. Holm 보정 후 유의 비교: A_B vs D_B (p<0.05)\"

## 완료 보고 양식

```
[v2 작업 완료 - Statistics]
수정한 파일:
직접 실행한 명령: aggregate / 손계산 검증 / CSV 열람
생성/확인한 산출물: paired_tests_holm.csv (7쌍 row)
통과한 검증: paired_statistics_smoke / Holm 보정 / CI 컬럼 채워짐
남은 위험:
다음 사람이 이어받을 부분: 카드 4가 paired_tests_holm.csv 읽고 보고서 row 생성
```

---
---

# 카드 3 — XAI Core Stage

## 이 stage가 뭔지

학습된 모델이 \"어디를 보고 그렇게 판단했는지\"를 SHAP/LIME으로 뜯어 CSV로 떨어뜨리는 stage. 3개 sub-stage:
- `xai-primary`: A_B vs D_B × 15시드 × 200샘플 → seed-level metrics
- `xai-deep`: median 시드 × 500샘플 → 정성 case
- `xai-ablation`: 8조건 × median 시드 × 50샘플 → ablation 매트릭스

명령: `./run.sh e2e xai-primary / xai-deep / xai-ablation`

## 사람이 하는 일 vs AI에 시키는 일

| 사람이 직접 (너) | AI 에이전트에 시킬 일 |
|---|---|
| `./run.sh e2e xai-*` 명령 직접 실행 | `xai.py` SHAP/LIME 함수 구현 |
| 200 sample이 seed마다 동일한지 확인 (sample manifest 비교) | sample stratification 알고리즘 |
| SHAP 결과 토큰 단위로 sanity check (말이 되는 토큰 짚는지) | 토크나이저 접두 분기 (`##`, `Ġ`, `▁`) |
| 워드 단위 집계 BERT/RoBERTa 분기 결과 직접 확인 | 메트릭 10개 계산 |
| `xai_summary.json` 열어서 메트릭 10개 다 있는지 확인 | SHAP CPU 강제 패턴 유지 |
| 결과 해석 1문단 작성 → 카드 4에 전달 | seed 간 sample 고정 코드 |

너의 비중: SHAP/LIME 결과가 \"이상한 토큰 짚고 있지 않은지\" 눈으로 확인하는 사람.

## 첫 주 D0~D3 체크리스트

- [ ] **D0**: AI에 카드 3 첫 문장 던지기. PR 검수.
- [ ] **D0**: 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/03_xai_agent.md`
  - `v2/docs/04_xai_protocol.md`
  - `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md` ← 카드 4의 입력 계약
- [ ] **D1**: AI가 SHAP CPU 강제 패턴 유지하는지 코드 검수.
- [ ] **D2**: dry-run으로 stratified sample 200개 추출이 seed 간 동일한지 확인.
- [ ] **D3**: 카드 1 smoke 끝나면 primary 1 seed × 50 sample smoke 직접 실행.

## 완료 기준 = 산출물 검수

9개 산출물 파일 + 메트릭 14개 확인:
```
outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv      # 18컬럼 (작업 #11)
outputs/experiments/v2_15seed/xai/primary/sample_level_metrics.csv    # 작업 #14
outputs/experiments/v2_15seed/xai/primary/paired_xai_tests.csv
outputs/experiments/v2_15seed/xai/primary/seed_stability.csv
outputs/experiments/v2_15seed/xai/deep/case_summary.csv
outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv   # 11컬럼 (작업 #8)
outputs/experiments/v2_15seed/xai/.cache/                              # SHAP/LIME 캐시 (작업 #4)
outputs/experiments/v2_15seed/xai/xai_summary.json
```

`seed_level_metrics.csv` 직접 열어서 4축 메트릭 14개 다 있는지 확인:
```
1축 Attribution:    SHAP-LIME Overlap@5 / @10
2축 Faithfulness:   Comprehensiveness / Sufficiency / LOO Drop
3축 Context Learn:  CI / MSS / Interaction Strength / Attention Entropy (작업 #11)
4축 Plausibility:   Rationale Precision / Recall / F1 @5
보조 (seed 간):     Top-k Jaccard / Rank correlation
```

토큰 단위 sanity check: cache 또는 deep case 1~2개 직접 열어서 SHAP top-5 토큰이 "이 문장의 혐오 근거"로 말이 되는지 눈으로 확인.
BERT/RoBERTa 서브워드 (`##` / `Ġ`) 집계가 단어 단위로 정확한지 1~2 sample 점검.

## 너가 직접 돌릴 명령

```bash
cd v2

# D2 dry-run
./run.sh e2e xai-primary --run-id v2_15seed --dry-run

# D3 primary smoke (카드 1 smoke 후)
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed

# 산출물 확인
cat outputs/experiments/v2_15seed/xai/xai_summary.json | python3 -m json.tool
column -t -s, outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv | head -10

# sample manifest 동일성 (seed 간) 검증
python3 - <<'PY'
import pandas as pd
df = pd.read_csv('outputs/experiments/v2_15seed/xai/samples/primary_samples.csv')
print('total samples:', len(df))
print('unique sample_ids:', df['sample_id'].nunique())
print('seeds covered:', sorted(df['seed'].unique()) if 'seed' in df else 'fixed sample set')
PY
```

## 막힐 때

- **페어**: 카드 1 (checkpoint 의존). 카드 1의 checkpoint 없으면 XAI 못 돌림.
- **페어**: 카드 4 (출력 계약 의존). 카드 4가 너의 CSV/JSON을 evidence bundle로 묶음.
- **금지**: full XAI 임의 실행. smoke 통과 후 카드 5 승인 받고 진행.

## 에이전트에 던질 첫 문장

```
당신은 HateSpeachStudy v2_15seed 파이프라인의 XAI Core 담당 에이전트입니다.

상황:
나는 사람이고 SHAP/LIME 출력을 토큰 단위로 직접 눈으로 검수하는 역할이다.
당신은 xai.py 구현과 토크나이저 접두 분기 코드 작성을 책임진다.

목표:
xai-primary / xai-deep / xai-ablation 산출물 7개 + xai_summary.json 생성.

원칙:
- XAI는 모델 설계 근거가 아니라 사후 검증 도구.
- 같은 sample set을 seed별 checkpoint에 동일하게 적용 (seed 간 변동 ≠ sample 변동).
- SHAP은 CPU 강제 (PartitionExplainer + transformer는 GPU에서 메모리 누수 잦음).
- 워드 단위 집계 시 BERT(##) / RoBERTa(Ġ) / SentencePiece(▁) 접두 분기 필수.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/03_xai_agent.md
v2/docs/04_xai_protocol.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md (카드 4의 입력 계약)

수정 가능 파일:
v2/pipeline/xai.py
v2/pipeline/schema.py

수정 금지:
v2/pipeline/xai_bundle.py / statistics.py / reporting.py
v2/runtime/experiment_core.py

필수 메트릭 10개:
SHAP-LIME Overlap@5/@10, Rationale Precision/Recall/F1@5,
Comprehensiveness, Sufficiency, Leave-one-out Drop,
Top-k Jaccard across seeds, Rank correlation across seeds

작업 순서:
1. xai.py 현황 점검 — SHAP CPU 강제 패턴 유지.
2. stratified sample 200 추출 함수 — seed 간 동일성 검증.
3. 카드 1 담당의 smoke checkpoint 들어오면 primary 1 seed × 50 sample smoke.

권장 톤:
"v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향을 보였다는
사후 근거로 해석한다."
```

## 금지 / 권장 표현

- 금지: \"XAI가 VADER 추가의 효과를 증명했다\"
- 금지: \"모델이 맥락을 완전히 이해한다\"
- 권장: \"v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향\"

## 완료 보고 양식

```
[v2 작업 완료 - XAI Core]
수정한 파일:
직접 실행한 명령:
생성/확인한 산출물: (7개 파일)
통과한 검증: sample seed 고정 / 워드 집계 정확 / 메트릭 10개 채움
남은 위험:
다음 사람이 이어받을 부분: 카드 4가 evidence bundle로 통합
```

---
---

# 카드 4 — XAI Bundle + Report Stage

## 이 stage가 뭔지

카드 2의 통계 CSV + 카드 3의 XAI CSV를 받아:
1. **xai-bundle**: 15개 evidence bundle 파일로 통합 (재계산 X).
2. **report**: `final_report.md` + `final_report.docx` 생성.
3. **dashboard**: `dashboard/index.html` 생성.
4. **발표 자료**: 발표 ppt + 교수 질문 대응 시나리오.

발표/제출의 절반이 너 손을 거친다. 가장 무거운 자리.

## 사람이 하는 일 vs AI에 시키는 일

| 사람이 직접 (너) | AI 에이전트에 시킬 일 |
|---|---|
| `./run.sh e2e xai-bundle / report / dashboard` 실행 | `xai_bundle.py` / `reporting.py` 구현 |
| **보고서 본문 글쓰기** (가장 무거움) | 15개 evidence bundle 파일 자동 생성 코드 |
| **발표 자료(ppt) 작성** | claim → source artifact 자동 연결 코드 |
| 카드 2/3의 \"해석 1문단\"을 받아 통합·재작성 | placeholder 패턴 구현 |
| 표/그림 정돈 + 캡션 작성 | HTML dashboard 렌더링 |
| 교수 질문 대응 시나리오 준비 (TF-IDF 강점) | JSON 유효성 검증 코드 |
| 카드 5와 글쓰기 페어 (검수·표 정리·발표 보조) | unit test |

너의 비중: 사람 손이 가장 많이 가는 자리. AI는 \"틀\"을 만들고 사람이 \"내용\"을 채움.

## 첫 주 D0~D4 체크리스트

- [ ] **D0**: AI에 카드 4 첫 문장 던지기. **09 문서 정독 (696줄, 가장 중요)**.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md` ★★★
  - `v2/docs/agent_tasks/04_report_dashboard_agent.md`
  - `v2/docs/07_output_and_report_contract.md`
  - `v2/docs/08_xai_report_template.md`
- [ ] **D1**: AI에 placeholder bundle 생성 요청. 빈 입력에서 안 깨지는지 확인.
- [ ] **D2**: `xai-bundle` 실행 → 15개 파일 `ls`로 확인.
- [ ] **D3**: `report` / `dashboard` 실행 → placeholder report 생성 확인.
- [ ] **D3**: **보고서 골격 markdown으로 미리 짜기** (full run 결과 없어도 됨).
- [ ] **D4**: 발표 자료 ppt 골격 작성. 카드 5와 페어 체크인.

## D5~D10 체크리스트 (결과 들어온 후)

- [ ] **D5**: 카드 2 통계 CSV row를 보고서에 채움. \"D_B는 A_B 대비 macro_f1 [X] 차이...\" 같은 문장 완성.
- [ ] **D6**: 카드 3 XAI 결과로 \"근거 토큰 정렬도\" 문장 완성.
- [ ] **D7**: TF-IDF baseline 대비 강점 섹션 마무리.
- [ ] **D8**: 발표 자료 1차 완성. 카드 5에게 검수 요청.
- [ ] **D9**: 카드 5 피드백 반영. dashboard HTML 최종 점검.
- [ ] **D10**: 발표 리허설. 교수 Q&A 시나리오 5개 준비.

## 완료 기준 = 산출물 검수

**xai-bundle (15개)**:
```
outputs/experiments/v2_15seed/xai/evidence_bundle/
  evidence_inventory.csv, xai_run_metadata.json, xai_sample_manifest.csv,
  xai_predictions.csv, method_agreement.csv, faithfulness_metrics.csv,
  context_metrics.csv, plausibility_metrics.csv, subgroup_xai_metrics.csv,
  xai_risk_flags.csv, xai_claims.json, xai_interpretation_cards.json,
  xai_dashboard_bundle.json, token_attributions.jsonl, README.md
```

**report (3개)**:
```
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

**발표 자료** (이건 너가 직접):
```
v2/outputs/experiments/v2_15seed/reports/presentation.pptx  (또는 발표 자료 폴더)
```

JSON 유효성 검증:
```bash
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
```

## 09 문서의 핵심 원칙 (절대 어기지 마)

**저장은 full, 노출은 요약**
- bundle엔 모든 raw 증거 (full).
- dashboard/report엔 핵심 주장과 대표 사례만 (요약).

**XAI는 사후 검증**
- xai-bundle은 SHAP/LIME 재계산 X — 통합만.
- claim 하나하나 source artifact 파일 경로 연결 필수.
- 통계 미확증 내용을 확정 claim으로 쓰지 마.

**TF-IDF 대비 강점 프레이밍**
- \"정확도 압도\"가 아님.
- \"근거 토큰 / 설명 충실도 / human rationale 정렬도 / 단어 의존성 / subgroup 취약성까지 함께 산출\".

## 가장 위험한 함정

15시드 중 일부 실패 가능성 항상. \"부분 결과로 report 안 깨지게\" 만드는 게 너의 가장 중요한 일.
모든 row에 \"결과 없으면 자동 placeholder\" 패턴 필수.

## 막힐 때

- **페어**: 카드 5 (글쓰기 보조). 카드 5가 검수·표 정리·발표 보조.
- **페어**: 카드 2, 3 (입력 의존). 둘에게 \"해석 1문단\" 요청.

## 에이전트에 던질 첫 문장

```
당신은 HateSpeachStudy v2_15seed 파이프라인의 XAI Bundle + Report 담당 에이전트입니다.

상황:
나는 사람이고 최종 보고서 / 발표 자료 / dashboard 글쓰기를 직접 책임지는 역할이다.
당신은 xai_bundle.py / reporting.py 구현, 15개 evidence bundle 파일 자동 생성,
placeholder 패턴, HTML 렌더링을 책임진다.

목표:
1. xai-primary/deep/ablation 결과를 통합해 full XAI evidence bundle 생성 (재계산 X).
2. benchmark/statistics 결과 + bundle을 읽어 final_report.md / final_report.docx /
   dashboard/index.html 생성.

핵심 원칙 (09 문서):
- 저장은 full, 노출은 요약.
- XAI는 사후 검증. 인과 증명이 아님.
- TF-IDF 대비 강점 = "정확도 압도"가 아니라 "근거 추적/검증/투명성".
- 부분 결과(일부 시드 실패)에서도 report가 안 깨져야 함.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md (가장 중요, 696줄)
v2/docs/agent_tasks/04_report_dashboard_agent.md
v2/docs/07_output_and_report_contract.md
v2/docs/08_xai_report_template.md

수정 가능 파일:
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py

수정 금지:
v2/pipeline/xai.py / statistics.py
v2/runtime/

산출물:
xai-bundle: 15개 파일 (09 문서의 contract 그대로)
report: final_report.md / final_report.docx / dashboard/index.html

작업 순서:
1. 빈 입력에서도 placeholder bundle/report가 깨지지 않게.
2. 모든 claim에 source artifact 파일 경로 연결.
3. 통계 미확증 내용은 확정 claim으로 쓰지 말 것.

금지 표현:
"XAI가 모델 개선 원인을 증명했다"
"XAI 결과를 보고 VADER를 설계했다"

권장 표현:
"통제된 ablation 조건 간 설명 패턴 차이를 사후 검증한다"
"여러 독립 지표가 같은 방향을 보인다"
```

## 금지 / 권장 표현

- 금지: \"XAI가 모델 개선 원인을 증명했다\"
- 금지: best seed 수치를 대표 성능처럼
- 권장: \"통제된 ablation 조건 간 설명 패턴 차이를 사후 검증\"

## 완료 보고 양식

```
[v2 작업 완료 - XAI Bundle + Report]
수정한 파일: (AI가 만든 PR)
직접 실행한 명령: xai-bundle / report / dashboard / 발표 자료 작성
생성/확인한 산출물: (bundle 15개 + report 3개 + ppt)
통과한 검증: JSON 유효성 / claim source 연결 / placeholder 동작 / 발표 자료 1차 완성
남은 위험:
다음 사람이 이어받을 부분: 카드 5의 글쓰기 페어 검수
```

---
---

# 카드 5 — QA + Server + Author 페어

## 이 stage가 뭔지

매일 preflight 명령으로 stage 깨짐 조기 감지 + Full Run Gate 판단 + 서버 시간 관리 + 4명 변경사항 통합 + **카드 4의 글쓰기 페어 보조**.

가장 폭넓은 역할. 모든 stage가 깨지지 않게 보는 자리.

## 사람이 하는 일 vs AI에 시키는 일

| 사람이 직접 (너) | AI 에이전트에 시킬 일 |
|---|---|
| 매일 preflight 명령 묶음 직접 실행 (10개) | `cli.py` / `manifest.py` 패치 |
| `status` 결과 확인 + `total=120` 검증 | `failed_runs.csv` / `completed_runs.csv` 생성 코드 |
| **Full Run Gate 6조건 통과 여부 판단** (사람의 결정) | `run.sh` 개선 |
| 서버 접속 + `nvidia-smi` | preflight 자동화 스크립트 |
| full 120 run 진행 중 모니터링 | resume 로직 강화 |
| 실패 run 발견 시 \"재실행 vs 중단\" 결정 | manifest hash 검증 |
| 단톡 진행 보고 (매일 1줄) | unit test |
| **카드 4의 글쓰기 페어**: 보고서 검수 / 표 정리 / 발표 보조 | — |

너의 비중: 매일 짧게 자주. 판단/결정/소통/페어 보조.

## D0~D10 체크리스트 (상시)

- [ ] **D0**: AI에 카드 5 첫 문장 던지기. 모든 팀원의 D0 요약 모아서 단톡에 분담 정리.
  - 정독: `00_common_agent_rules.md` / `05_qa_server_agent.md` / `06_integration_lead_agent.md` / `07_review_agent.md` / `11_team_tasking_and_server_run_plan.md` (657줄, 전체)
- [ ] **D1**: ~~매일 preflight 명령 묶음을 `daily.sh`로 자동화~~ → 작업 #3에서 `v2/scripts/daily.sh` 완성. 작업 #13에서 `gate_check.py` 통합으로 GO/STOP까지 자동.
- [ ] **D2**: 카드 1 smoke 준비 도와주기 (CLI/manifest 점검).
- [ ] **D3**: 카드 1 smoke 통과 확인 + `execution_status.csv` 신뢰성 검증.
- [ ] **D4**: **Full Run Gate 6조건 모두 통과 점검**. paired smoke 확인. 통과 시 \"GO\" 결정.
- [ ] **D5-D6**: full 120 benchmark 서버 실행. 매시간 status 모니터링.
- [ ] **D7+**: 실패 unit 회수 결정. 카드 2/3 진행 도움.
- [ ] **D8-D10**: 카드 4의 글쓰기 페어로 본격 합류. 보고서 검수, 발표 자료 표 정리, ppt 슬라이드 점검.

## 매일 돌릴 preflight (작업 #3 + #13에서 한 줄로 통합 완료)

```bash
cd v2
./scripts/daily.sh
```

내부적으로 다음을 자동 실행:
- python3 -m compileall pipeline + config 유효성
- ./run.sh e2e --help / status / plan / dry-run benchmark
- 빈 입력에서 aggregate / xai-bundle / report / dashboard 무회귀
- failed_runs.csv / completed_runs.csv 행 수 출력
- **python3 scripts/gate_check.py — Full Run Gate 6조건 자동 점검 (작업 #13)**

마지막 줄로 GO/STOP을 자동 표시:
```
[daily preflight ok — Gate: GO]
```

매일 단톡 보고는 그 한 줄 캡처만.

## Full Run Gate (작업 #13에서 자동화)

```
1. cudnn 결정성: utils.set_seed가 cudnn.deterministic=True 설정 확인 (CUDA 환경)
2. 8조건 metadata 정합성: CONDITION_METADATA ↔ V2_CONDITION_SPECS 일치
3. aggregate CSV 7개 모두 존재 (benchmark_*, paired_*, anova_*)
4. XAI sample 결정성: primary_samples.csv md5 일치 (재실행 시)
5. failed_runs.csv 0행
6. 산출물 contract 8개 모두 존재
```

`python3 scripts/gate_check.py --run-id v2_15seed` 단독 실행 가능. exit 0 = GO, 1 = STOP.
daily.sh 안에서는 `--skip-sample-check`로 호출 (xai-primary 재실행 비용 회피).

너의 결정: gate_check가 6/6 PASS이면 full 120 GO 단톡 보고. 하나라도 FAIL이면 STOP + 해당 카드에 fix 요청.

## 즉시 중단 기준 (있으면 stage 멈춰)

```
split hash 로컬/서버 다름
같은 seed에서 condition마다 sample order 다름
VADER feature가 A/B 조건에 들어감
attention loss가 rationale 없는 샘플에 잘못 적용
metrics schema가 condition마다 다름
NaN loss 반복 발생
checkpoint가 run_id 외부에만 저장됨
```

## failed unit만 재실행 (전체 중단 X)

```
일시적 CUDA OOM
서버 세션 끊김
단일 seed run 실패
일부 XAI sample 실패
```

## 카드 4 Author 페어 보조 패턴 (D8~D10)

D8부터 카드 4와 매일 30분 페어 체크인:

| 작업 | 카드 4가 한 것 | 카드 5의 보조 |
|---|---|---|
| 보고서 본문 | 초안 작성 | 문장 검수 / 숫자 일치 확인 / 컬럼 명 검수 |
| 발표 자료 표 | 데이터 추출 | 표 정렬 / 캡션 작성 / 시각화 점검 |
| dashboard HTML | 자동 생성 | 직접 열어서 깨진 부분 확인 |
| 교수 Q&A 시나리오 | 5개 질문 작성 | 답변 1문단씩 같이 다듬기 |

## 너만 할 수 있는 한 가지

다른 사람이 자기 stage 밖 파일을 건드리면 \"왜?\" 물어보고 정리해줘.
모든 통합 충돌의 마지막 결정자. **Full Run Gate \"GO\" 권한은 너에게**.

## 막힐 때

- **페어**: 전원. 너는 모두의 통합 지점.
- **권한**: Gate \"안 됨\" 결정 권한은 너에게.

## 에이전트에 던질 첫 문장

```
당신은 HateSpeachStudy v2_15seed 파이프라인의 QA + Server + Author 페어 담당 에이전트입니다.

상황:
나는 사람이고 매일 preflight 명령을 직접 실행하고, Full Run Gate 통과 여부를
판단하고, 서버 시간을 관리하고, 후반에는 카드 4(보고서 담당)와 글쓰기 페어로
활동한다. 당신은 cli.py / manifest.py / artifacts.py 패치, preflight 자동화
스크립트, status 명령 정확성을 책임진다.

목표:
1. 매일 dry-run/status/compile 명령 묶음(daily.sh)으로 stage 깨짐 조기 감지.
2. Full Run Gate 6조건 통과 전 full 120 benchmark 차단.
3. 4명 (Benchmark/Statistics/XAI Core/XAI Bundle+Report)의 변경사항 통합.
4. 서버 실행 시 smoke → full 순서 강제. 실패 시 unit만 재실행.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/05_qa_server_agent.md
v2/docs/agent_tasks/06_integration_lead_agent.md
v2/docs/agent_tasks/07_review_agent.md
v2/docs/11_team_tasking_and_server_run_plan.md (657줄, 전체)
v2/docs/06_execution_runbook.md

수정 가능 파일:
v2/pipeline/cli.py / manifest.py / artifacts.py
v2/run.sh
v2/scripts/validate_commit_message.py
v2/configs/v2_15seed.json

수정 금지:
v2/runtime/
v2/pipeline/statistics.py / xai.py / xai_bundle.py / reporting.py

Full Run Gate 6조건 (다 통과 전엔 full 120 금지):
1. v2_runtime_import_smoke 통과
2. A_B seed 42 단일 smoke 성공
3. A_B/D_B seed 42 paired smoke 성공
4. metrics/history/config/predictions/checkpoint 생성
5. aggregate가 smoke 결과 읽고 paired_tests row 생성
6. checkpoint_path/predictions_path가 v2 내부 경로

매일 preflight 명령 묶음을 daily.sh로 자동화해주세요.
```

## 완료 보고 양식

```
[v2 integration report - D{N}]
Today's preflight: ok / fail
Stage status (1~5): 1=완료/진행중 ...
Blocking issue: 없음 / [구체적]
Full Run Gate: 1✓ 2✓ 3? 4? 5? 6?
Ready for server smoke: yes / no
Tomorrow plan:
```

---
---

# 부록 A. 5개 카드 한 페이지 요약 (회의실 벽 포스터)

```
[1. Benchmark]      서버 학습 직접 돌리는 사람
                    D3 목표: A_B seed 42 smoke ok
                    AI 부담: 코드 패치 / NVIDIA 호환
                    주 4~6h

[2. Statistics]     통계 CSV 숫자 직접 검증하는 사람
                    D3 목표: paired_statistics_smoke 통과
                    AI 부담: statistics.py / Holm 보정
                    주 3~5h

[3. XAI Core]       SHAP/LIME 결과 토큰 단위로 sanity check
                    D3 목표: primary 1 seed × 50 sample smoke
                    AI 부담: xai.py / 토크나이저 분기
                    주 4~6h

[4. XAI Bundle + Report]  보고서 / 발표 자료 직접 쓰는 사람
                          D4 목표: placeholder report + 보고서 골격
                          D10 목표: final + 발표 자료
                          AI 부담: xai_bundle.py / reporting.py
                          주 7~10h (가장 무거움)

[5. QA + Server + Author 페어]  매일 preflight + Gate + 글쓰기 보조
                                  D0~D10 상시
                                  AI 부담: cli.py / daily.sh 자동화
                                  주 5~7h
```

---

# 부록 B. 카드 5의 \"Author 페어 보조\" 작업표 (D8~D10)

| 날짜 | 카드 4 | 카드 5 (보조) |
|---|---|---|
| D8 오전 | report markdown 본문 초안 | preflight + status 확인 |
| D8 오후 | XAI 해석 문단 통합 | 보고서 본문 검수 (숫자 일치 / 컬럼명) |
| D9 오전 | dashboard HTML 점검 | dashboard 직접 열어서 깨진 부분 보고 |
| D9 오후 | 발표 자료 ppt 초안 | 표 정렬 + 캡션 작성 보조 |
| D10 오전 | 교수 Q&A 시나리오 5개 | 답변 1문단씩 같이 다듬기 |
| D10 오후 | 리허설 | 시간 측정 + 슬라이드 전환 점검 |

---

# 부록 C. 공통 — 첫 에이전트 호출 직전에 박을 컨텍스트

모든 카드의 \"에이전트 첫 문장\" 앞에 붙임:

```
프로젝트: HateSpeachStudy v2_15seed
저장소 위치: <repo root>
작업 폴더: v2/
Run ID: v2_15seed
산출물 루트: outputs/experiments/v2_15seed/
CLI 진입점: ./run.sh e2e ...
Config: configs/v2_15seed.json
설계 문서 위치: v2/docs/

원칙:
- v1은 archive. 절대 import 안 함.
- 모든 산출물은 outputs/experiments/v2_15seed/ 아래.
- 사람이 명령 실행 / 결과 검수 / 글쓰기를 하고, AI 에이전트가 코드 작성을 한다.
- 자기 담당 stage 밖 파일 수정은 사유 + 변경 범위 보고.
- full benchmark / full XAI는 카드 5(QA)의 승인 전 절대 실행 금지.

내 역할: [카드 N 본문 통째로 붙여넣기]
```

---

# 부록 D. 매주 1회 교차 리뷰 (07 Review 책임 분담)

```
Week 1: 카드 1 ↔ 카드 5 (CLI/adapter 정합)
Week 2: 카드 2 ↔ 카드 4 (통계 → 보고서 컬럼 정합)
Week 3: 카드 3 ↔ 카드 4 (XAI → bundle 입력 계약 정합)
Week 4: 전체 모임 (Full Run Gate 점검 + 발표 리허설)
```

리뷰 양식: `v2/docs/agent_tasks/07_review_agent.md` 5장 \"리뷰 출력 형식\" 그대로.

---

# 부록 E. 분배 직후 단톡 공지 템플릿

회의에서 분담 정해지면 단톡에 박을 1단락:

```
[v2_15seed 업무 분담 확정]

카드 1. Benchmark        — [이름]
카드 2. Statistics       — [이름]
카드 3. XAI Core         — [이름]
카드 4. XAI Bundle + Report — [이름]
카드 5. QA + Server + Author 페어 — [이름]

원칙:
1. 산출물은 무조건 v2/outputs/experiments/v2_15seed/ 안.
2. v1/ 폴더는 read-only archive. AI에 v1 만지라고 시키지 마.
3. 코드는 AI가, 사람은 명령 실행 / 결과 검수 / 글쓰기.
4. full benchmark/full XAI는 카드 5 승인 전 절대 실행 금지.
5. 막히면 옆 번호에게 핑. 카드 5는 누구든 받음.

D0 = [YYYY-MM-DD]
D3 마일스톤: 카드 1의 A_B seed 42 smoke 통과
D10 마일스톤: 발표 자료 + 보고서 최종

오늘 D0 할 일:
- 각자 자기 카드 정독 → 단톡 답글에 1줄 요약 박기
- AI 에이전트에 카드 본문 + 첫 문장 던지기
- 막힐 때 단톡에 질문

다음 모임: [YYYY-MM-DD HH:MM]
```

---

작성: 2026-05-17
참조: `18_team_role_cards.md` (archive, 코드 작성 가정), `14_team_assignment_matrix.md`, `15_runtime_code_validation_matrix.md`, `11_team_tasking_and_server_run_plan.md`, `agent_tasks/*.md`
