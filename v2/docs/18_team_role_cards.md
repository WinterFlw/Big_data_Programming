# 18. Team Role Cards (Archive — \"코드 신규 작성 가정\" 시나리오)

> **주의**: 이 문서는 \"5명이 코드를 신규로 작성한다\"는 가정으로 짜인 초기 카드 세트다.
> 실제 v2 상태는 코드베이스가 대부분 갖춰져 있고 사람은 **검증·실행·판단·글쓰기·운영** 중심이다.
> → 현행 분배 시나리오는 `19_team_role_tracks.md`를 본다. 이 문서는 reference로만 보관.
>
> 살릴 부분: 각 카드의 \"에이전트에 던질 첫 문장\" 블록은 19번에서도 그대로 재사용된다.

---

# (원본) 5개 역할을 카드 형태로 정리. 인원 미배정 — 회의에서 카드 보고 본인이 가져갈 것 선택.
> 각 카드는 self-contained. 자기 카드 하나 통째로 AI 에이전트에 복사해서 던지면 즉시 작업 시작 가능.

---

## 0. 카드 한눈에 비교

| 카드 | 역할 | 작업량 | 책임 무게 | 추천 배경 |
|---|---|---|---|---|
| **A** | Benchmark | 높음 | 중 | 학습 코드 / GPU / adapter 익숙 |
| **B** | Statistics | 중 | 중 | 통계학 / scipy 익숙 |
| **C** | XAI Core | 높음 | 중 | SHAP·LIME / 토크나이저 / 모델 내부 익숙 |
| **D** | XAI Bundle + Report | 매우 높음 | 매우 높음 | 글쓰기 / 데이터 정돈 / 발표 자료 |
| **E** | QA + Server + Integration | 중 | 높음 | CLI / 서버 운영 / 체계화 |

작업량 ≠ 어려움. 카드 D는 무겁지만 코드 신규 작성보다 \"기존 결과를 잘 묶고 잘 말하는\" 일에 가깝다.

---

## 0.1 의존 관계 (누가 막히면 누가 멈추나)

```
A (Benchmark) ──┬──→ B (Statistics) ────┐
                │                          ├──→ D (XAI Bundle + Report)
                └──→ C (XAI Core) ─────────┘

E (QA + Server) ──→ 모든 stage 관문 / 통합
```

- A가 첫 주 막히면 전원 멈춤 → A가 D3까지 smoke 통과 필수.
- E는 매일 검증 명령 돌리며 다른 사람 작업이 안 깨지는지 확인.
- B와 C는 A의 smoke 결과 들어오기 전까지는 fake metrics로 unit test 진행.

---

## 0.2 정하는 방법 (회의 5분)

1. 5개 카드 같이 훑기.
2. 카드 D부터 정하기. 가장 무겁고 책임 크다. 발표 자료까지 책임지는 자리.
3. 그 다음 E. 관문 + 통합 + 서버 운영. 다른 사람 일 잘 받아낼 사람.
4. A/B/C 셋은 흥미·배경 기준으로 자유롭게.

피해야 할 매칭:
- A와 E를 한 명에게 — 둘 다 \"막히면 안 됨\" 자리라 부담 폭증.
- D를 발표 부담 큰 사람에게 — 코드도 같이 보는 자리라 글쓰기만 잘하면 곤란.

---

## 0.3 공통 원칙 (5명 다 지킬 것)

```
산출물은 v2/outputs/experiments/v2_15seed/ 안에만.
v1/ 폴더는 read-only archive. 절대 import 금지.
자기 파일 밖 수정 시 단톡에 사유 1줄 + 변경 범위.
full benchmark / full XAI는 E의 승인 전 절대 실행 금지.
모든 stage는 resume 가능해야 함 (실패 unit만 재실행).
```

---
---

# 카드 A — Benchmark

## 한 줄 정의
`./run.sh e2e benchmark --execute` 명령이 실제 학습 1개를 돌리고, 결과 파일을 v2 정규 경로에 정확히 떨어뜨리도록 만든다.

## 너의 책임
1. v2 CLI ↔ runtime 학습 코드 연결 (adapter 작성).
2. condition × seed 단위 학습 실행 + resume 지원.
3. NVIDIA CUDA 환경 호환 (device 통일, AMP, cudnn deterministic).

## 너의 파일 (수정 OK)
```
v2/pipeline/runner.py
v2/pipeline/training_adapter.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py
v2/runtime/utils.py        ← device 통일 부분만
v2/configs/v2_15seed.json  ← 필요 시
```

## 참고만 (읽기 OK, 수정 X)
```
v2/runtime/experiment_core.py
v2/runtime/run_experiments.py
```

## 절대 건드리지 마
```
v2/pipeline/statistics.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py
v1/  (폴더 전체)
```

## 첫 주 D0~D3 체크리스트
- [ ] **D0**: 아래 문서 정독 → 자기 책임 한 줄로 단톡에 박기.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/01_benchmark_agent.md`
  - `v2/docs/02_e2e_pipeline.md`
  - `v2/docs/15_runtime_code_validation_matrix.md` 1~2번 섹션
- [ ] **D1**: `training_adapter.py`가 v1을 import 안 하는지 확인. `RUNTIME_DIR`가 `v2/runtime`인지 검증.
- [ ] **D2**: dry-run 통과 (아래 명령).
- [ ] **D3**: A_B seed 42 단일 smoke 성공 → 단톡 `[smoke ok] A_B seed 42` 보고.

## 첫 실행 명령 (복붙)
```bash
cd v2
./run.sh e2e --help
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force

# dry-run (D2 목표)
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run

# smoke (D3 목표, GPU 서버에서)
PYTHON_BIN=/path/to/venv/bin/python \
  ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume

# 검증
python3 - <<'PY'
from pipeline.training_adapter import _load_runtime_core, _condition_spec, RUNTIME_DIR
core = _load_runtime_core()
spec = _condition_spec(core, 'A_B')
assert spec.model_name == 'bert-base-uncased'
assert str(RUNTIME_DIR).endswith('/v2/runtime')
assert '/v1/' not in str(core.__file__)
print('v2_runtime_import_smoke: ok')
PY
```

## 완료 기준 (이게 되면 끝)
A_B seed 42 smoke 실행 후 아래 6개 파일 생성:
```
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/metrics.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/history.csv
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/run_config.json
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stdout.log
outputs/experiments/v2_15seed/benchmark/runs/a_b/seed_42/stderr.log
outputs/experiments/v2_15seed/benchmark/checkpoints/a_b_seed_42.pt
```

## NVIDIA CUDA 환경 체크리스트
`utils.py` 만질 때 같이 박을 것:
```python
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 시드 고정 필수
```
AMP는 `device.type == "cuda"`일 때만 활성화. MPS에서는 끔.

## 막힐 때
- **페어**: 카드 E (CLI/manifest 의존). 매일 dry-run/status 명령 결과 공유.
- **금지**: condition별 hyperparameter 임의 변경 / 기존 `outputs/runs/` 또는 `checkpoints/`를 canonical로 사용 / full 120 benchmark 임의 실행.

## 에이전트에 던질 첫 문장 (그대로 복사)
```
당신은 HateSpeachStudy v2_15seed 파이프라인의 Benchmark Adapter 담당 에이전트입니다.

목표:
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
명령이 실제 학습 1개를 실행하고, run_id 내부에 metrics.json/history.csv/
run_config.json/stdout.log/stderr.log/checkpoint를 정확히 생성하게 만드는 것.

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
1. training_adapter가 v1 import하지 않는지 확인 (위 검증 PY 블록 실행).
2. dry-run 통과 (--dry-run).
3. NVIDIA 환경: get_device / set_seed / AMP 조건 확인.
4. A_B seed 42 단일 smoke (--execute --resume).

완료 시 위 6개 산출물 파일 경로 모두 보고.
```

## 완료 보고 양식
```
[v2 작업 완료]
담당: Benchmark
수정한 파일:
실행한 명령어:
생성/변경된 산출물:
통과한 검증:
남은 위험:
다음 사람이 이어받을 부분: (B/C가 smoke 결과로 작업 시작 가능 여부)
```

---
---

# 카드 B — Statistics

## 한 줄 정의
A가 떨어뜨린 metrics.json 120개(8조건 × 15시드)를 읽어, 15 seed mean/std와 핵심 A_B vs D_B paired t-test + effect size + 95% CI를 CSV로 출력한다. Holm 보정은 여러 비교를 함께 보여줄 때의 보조 확인값으로만 둔다.

## 너의 책임
1. condition별 mean/std/CI 집계.
2. 핵심 A_B vs D_B same-seed paired comparison.
3. 나머지 6쌍과 Holm-Bonferroni는 부록/보조 분석으로 정리.
4. 결측/실패 run 처리 (빈 결과에서도 aggregate 안 깨지게).

## 너의 파일 (수정 OK)
```
v2/pipeline/statistics.py
v2/pipeline/schema.py
v2/pipeline/runner.py   ← 필요 시
```

## 절대 건드리지 마
```
v2/runtime/         (전체)
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py
v1/
```

## 첫 주 D0~D3 체크리스트
- [ ] **D0**: 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/02_statistics_agent.md`
  - `v2/docs/03_validation_and_statistics.md`
  - `v2/docs/07_output_and_report_contract.md`
- [ ] **D1**: 현재 `statistics.py` 함수 목록 훑기. fake metrics 2-3개로 unit test.
- [ ] **D2**: `aggregate` 명령이 결과 0건에서도 안 깨지는지 검증.
- [ ] **D3**: A의 smoke 결과 들어오면 paired_tests row 1개 생성 확인.

## 첫 실행 명령 (복붙)
```bash
cd v2

# aggregate (빈 입력 OK)
./run.sh e2e aggregate --run-id v2_15seed

# fake metrics로 paired test smoke
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
```

## 완료 기준
4개 CSV 생성:
```
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
outputs/experiments/v2_15seed/benchmark/paired_tests.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
```

## 필수 컬럼 (이 중 하나라도 빠지면 D가 보고서 못 씀)
```
comparison, metric, condition_a, condition_b, n_pairs,
mean_diff, std_diff, ci_low, ci_high,
test_name, p_value, p_value_holm, effect_size, significant_0_05
```

## paired 비교 우선순위
```
본문 핵심:
A_B:D_B

보조/부록:
A_B:D_B,  A_B:B_B,  A_B:C_B,
B_B:D_B,  C_B:D_B,
A_R:D_R,  D_B:D_R
```

## 금지 / 권장 표현
- 금지: \"p-value가 유의하므로 모델이 완전히 개선되었다\"
- 금지: \"XAI가 성능 향상을 증명한다\"
- 권장: 너는 CSV만 만든다. 해석 문장은 카드 D가 쓴다.

## 막힐 때
- **페어**: 카드 A (입력 metrics.json 스키마 의존). A의 smoke 결과 들어오자마자 paired row 검증.
- **페어**: 카드 D (출력 컬럼 스키마 의존). D가 어떤 컬럼을 읽는지 사전 협의.

## 에이전트에 던질 첫 문장
```
당신은 HateSpeachStudy v2_15seed 파이프라인의 Statistics 담당 에이전트입니다.

목표:
outputs/experiments/v2_15seed/benchmark/runs/ 아래의 condition × seed
metrics.json을 읽어서 benchmark_summary.csv / paired_tests.csv /
paired_tests_holm.csv 를 생성.

원칙:
- same-seed paired comparison만 인정. 다른 seed끼리 비교 금지.
- p-value만 보고하지 말 것 — mean_diff, 95% CI, Cohen's dz를 함께 출력.
- Holm 보정 p는 여러 조건 비교표의 보조 확인값으로만 둔다.
- 빈 결과/부분 결과에서도 aggregate가 실패하지 않게.

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

본문 핵심 paired 비교:
A_B:D_B

보조 paired 비교:
A_B:B_B, A_B:C_B, B_B:D_B, C_B:D_B, A_R:D_R, D_B:D_R

작업 순서:
1. fake metrics로 unit test (위 PY 블록).
2. 빈 입력에서 aggregate 안 깨지는지 확인.
3. A 담당의 smoke 결과 들어오면 paired row 생성 확인.

해석 문장은 작성하지 말 것 — Report 담당이 함.
```

## 완료 보고 양식
```
[v2 작업 완료]
담당: Statistics
수정한 파일:
실행한 명령어:
생성/변경된 산출물:
통과한 검증: 핵심 paired test row 생성 확인 / CI·effect size 컬럼 채워짐 / adjusted p-value 보조 산출 확인
남은 위험:
다음 사람이 이어받을 부분: (D가 paired_tests_holm.csv 읽고 보고서 row 생성)
```

---
---

# 카드 C — XAI Core

## 한 줄 정의
학습된 모델이 \"어디를 보고 그렇게 판단했는지\"를 SHAP/LIME/masking으로 뜯어 CSV로 떨어뜨린다. xai-primary / xai-deep / xai-ablation 3 stage.

## 너의 책임
1. **xai-primary**: A_B vs D_B × 15 seed × 200 fixed sample → seed-level metrics.
2. **xai-deep**: median seed × 500 sample → 정성 case + 시각화.
3. **xai-ablation**: 8조건 × median seed × 50 sample → ablation 매트릭스.
4. SHAP CPU 강제 유지 (NVIDIA에서도). 워드 단위 집계 BERT(##)/RoBERTa(Ġ) 분기.

## 너의 파일 (수정 OK)
```
v2/pipeline/xai.py
v2/pipeline/schema.py   ← XAI 스키마 추가만
v2/pipeline/runner.py   ← 필요 시
```

## 참고만 (읽기 OK, 수정 X)
```
v2/runtime/experiment_xai.py
```

## 절대 건드리지 마
```
v2/pipeline/xai_bundle.py   (카드 D 영역)
v2/pipeline/statistics.py
v2/pipeline/reporting.py
v2/runtime/experiment_core.py
v1/
```

## 첫 주 D0~D3 체크리스트
- [ ] **D0**: 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/03_xai_agent.md`
  - `v2/docs/04_xai_protocol.md`
  - `v2/docs/08_xai_report_template.md`
  - `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md` ← D의 입력 계약
- [ ] **D1**: 현재 `xai.py` 어디까지 구현되어 있는지 훑기. SHAP CPU 강제 패턴 점검.
- [ ] **D2**: stratified sample 200개 추출이 seed 간 동일 (seed 고정) 검증.
- [ ] **D3**: A의 smoke checkpoint 들어오면 primary XAI 1 seed × 50 sample smoke.

## 첫 실행 명령 (복붙)
```bash
cd v2

./run.sh e2e xai-primary --run-id v2_15seed --dry-run
./run.sh e2e xai-bundle --run-id v2_15seed   # D가 너 결과 잘 받는지 확인용

# 실제 smoke (A의 smoke checkpoint 있을 때)
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
```

## 완료 기준
7개 산출물 파일 생성:
```
outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv
outputs/experiments/v2_15seed/xai/primary/paired_xai_tests.csv
outputs/experiments/v2_15seed/xai/primary/seed_stability.csv
outputs/experiments/v2_15seed/xai/deep/case_summary.csv
outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv
outputs/experiments/v2_15seed/xai/xai_summary.json
```

## 필수 메트릭 (10개)
```
SHAP-LIME Overlap@5
SHAP-LIME Overlap@10
Rationale Precision@5
Rationale Recall@5
Rationale F1@5
Comprehensiveness
Sufficiency
Leave-one-out Drop
Top-k Jaccard across seeds
Rank correlation across seeds
```

## 토큰 처리 함정 (자주 실수)
서브워드 접두 잘못 판별하면 워드 단위 집계 무너짐:
```
BERT WordPiece:     "##" 접두 = 이어붙는 조각
RoBERTa BPE:        "Ġ"  접두 = 새 단어 시작
SentencePiece:      "▁"  접두 = 새 단어 시작
word-level token:   접두 없음
```

## 금지 / 권장 표현
- 금지: \"XAI가 VADER 추가의 효과를 증명했다\"
- 금지: \"모델이 맥락을 완전히 이해한다\"
- 권장: \"v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향을 보였다\"
- 권장: XAI는 사후 검증 도구. 모델 설계 근거 아님.

## 막힐 때
- **페어**: 카드 A (checkpoint 의존). 너는 A의 checkpoint를 load해서 XAI 돌림.
- **페어**: 카드 D (출력 계약 의존). D가 어떤 컬럼/JSON 구조를 기대하는지 사전 협의.
- **금지**: full XAI 임의 실행. smoke까지만.

## 에이전트에 던질 첫 문장
```
당신은 HateSpeachStudy v2_15seed 파이프라인의 XAI Core 담당 에이전트입니다.

목표:
xai-primary / xai-deep / xai-ablation 산출물 생성.
- primary: A_B vs D_B × 15 seed × 200 fixed sample → seed-level metrics
- deep:    median seed × 500 sample → case 분석
- ablation: 8조건 × median seed × 50 sample → ablation 매트릭스

원칙:
- XAI는 모델 설계 근거가 아니라 사후 검증 도구.
- 같은 sample set을 seed별 checkpoint에 동일하게 적용 (seed 간 변동 ≠ sample 변동).
- SHAP은 CPU 강제 (PartitionExplainer + transformer는 GPU에서 메모리 누수 잦음).
- 워드 단위 집계 시 BERT(##) / RoBERTa(Ġ) 접두 분기 필수.

먼저 읽을 문서:
v2/docs/agent_tasks/00_common_agent_rules.md
v2/docs/agent_tasks/03_xai_agent.md
v2/docs/04_xai_protocol.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md (Bundle 담당의 입력 계약)

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
3. A 담당의 smoke checkpoint 들어오면 primary 1 seed × 50 sample smoke.

권장 표현 톤:
"v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향을 보였다는
사후 근거로 해석한다."
```

## 완료 보고 양식
```
[v2 작업 완료]
담당: XAI Core
수정한 파일:
실행한 명령어:
생성/변경된 산출물: (7개 파일 경로)
통과한 검증: sample seed 고정 / 워드 단위 집계 정확
남은 위험:
다음 사람이 이어받을 부분: (D가 evidence bundle로 통합)
```

---
---

# 카드 D — XAI Bundle + Report

## 한 줄 정의
B의 통계 CSV + C의 XAI CSV를 받아, evidence bundle로 묶고 최종 report/dashboard를 만든다. 발표/제출의 절반이 너 손을 거친다.

## 너의 책임
1. **xai-bundle**: C가 만든 raw XAI를 \"누가 봐도 검증 가능한 evidence 패키지\"로 통합 (재계산 X).
2. **report**: `final_report.md` + `final_report.docx` 생성.
3. **dashboard**: `dashboard/index.html` 생성 — 실행 상태 + 핵심 표.
4. **교수 질문 방어**: TF-IDF baseline 대비 강점을 \"XAI evidence\"로 설명.

## 너의 파일 (수정 OK)
```
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py
v2/pipeline/schema.py   ← 필요 시
```

## 입력만 (읽기 OK, 수정 X)
```
outputs/experiments/v2_15seed/benchmark/*   ← B의 산출물
outputs/experiments/v2_15seed/xai/primary/*
outputs/experiments/v2_15seed/xai/deep/*
outputs/experiments/v2_15seed/xai/ablation/*
```

## 절대 건드리지 마
```
v2/pipeline/xai.py
v2/pipeline/statistics.py
v2/runtime/         (전체)
v1/
```

## 첫 주 D0~D4 체크리스트
- [ ] **D0**: 핵심 문서 정독. **09번 문서는 정독 필수 (696줄)**.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md` ★★★
  - `v2/docs/agent_tasks/04_report_dashboard_agent.md`
  - `v2/docs/07_output_and_report_contract.md`
  - `v2/docs/08_xai_report_template.md`
- [ ] **D1**: 현재 `xai_bundle.py` / `reporting.py` 어디까지 있는지 훑기.
- [ ] **D2**: 빈 입력에서 placeholder bundle / placeholder report 만드는지 검증.
- [ ] **D3**: B/C의 smoke 결과 들어오면 실제 row 1개 들어간 mini report.
- [ ] **D4**: 발표 흐름 초안 작성 (full run 결과 나오기 전이라도 골격은 미리).

## 첫 실행 명령 (복붙)
```bash
cd v2

./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

# bundle JSON 유효성 검증
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json >/dev/null
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/dev/null
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_interpretation_cards.json >/dev/null
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/dev/null
```

## 완료 기준
**xai-bundle (15개 산출물)**:
```
outputs/experiments/v2_15seed/xai/evidence_bundle/
  evidence_inventory.csv
  xai_run_metadata.json
  xai_sample_manifest.csv
  xai_predictions.csv
  method_agreement.csv
  faithfulness_metrics.csv
  context_metrics.csv
  plausibility_metrics.csv
  subgroup_xai_metrics.csv
  xai_risk_flags.csv
  xai_claims.json
  xai_interpretation_cards.json
  xai_dashboard_bundle.json
  token_attributions.jsonl
  README.md
```

**report (3개)**:
```
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

## 09 문서의 핵심 원칙 (반드시 따를 것)

**저장은 full, 노출은 요약**
- bundle엔 모든 raw 증거 (full).
- dashboard/report엔 핵심 주장과 대표 사례만 (요약).
- 나중에 교수 질문 들어오면 raw artifact로 backtracking 가능해야.

**XAI는 사후 검증**
- xai-bundle은 SHAP/LIME 재계산 X — 통합만.
- claim 하나하나 source artifact 파일 경로 연결 필수.
- 통계적 미확증 내용을 확정 claim으로 쓰지 마.

**TF-IDF 대비 강점 프레이밍**
- \"정확도가 압도적으로 높다\"가 아님.
- \"근거 토큰 / 설명 충실도 / human rationale 정렬도 / 단어 의존성 / subgroup 취약성까지 함께 산출\".

## 금지 표현
```
"XAI가 모델 개선 원인을 증명했다"
"XAI 결과를 보고 VADER를 설계했다"
"Attention이 곧 모델의 진짜 이유다"
"모델이 완전히 맥락을 이해한다"
```

## 권장 표현
```
"통제된 ablation 조건 간 설명 패턴 차이를 사후 검증한다"
"여러 독립 지표가 같은 방향을 보이는지 확인한다"
"성능 차이가 작더라도 판단 근거의 투명성과 취약성 분석을 제공한다"
"v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향을 보였다"
```

## 가장 위험한 함정
15 seed 중 일부 실패 가능성 항상 있음.
\"부분 결과로 report 안 깨지게\" 만드는 게 너의 가장 중요한 일.
모든 row에 \"결과 없으면 자동 placeholder\" 패턴 필수.

## 막힐 때
- **페어**: 카드 B (paired_tests_holm.csv 입력). 컬럼 스키마 사전 협의.
- **페어**: 카드 C (XAI CSV 입력). xai_summary.json 구조 사전 협의.
- **페어**: 카드 E (CLI 진입점). xai-bundle/report/dashboard stage 등록 협의.

## 에이전트에 던질 첫 문장
```
당신은 HateSpeachStudy v2_15seed 파이프라인의 XAI Evidence Bundle + Report 담당 에이전트입니다.

목표:
1. xai-primary/deep/ablation 산출물을 통합해 full XAI evidence bundle 생성
   (재계산 X, 통합만).
2. benchmark/statistics 산출물 + evidence bundle 읽어서 final_report.md /
   final_report.docx / dashboard/index.html 생성.

핵심 원칙 (09 문서):
- 저장은 full, 노출은 요약. bundle은 모든 raw 증거, dashboard/report는
  핵심 주장만.
- XAI는 사후 검증. 모델 설계 근거나 인과 증명이 아님.
- TF-IDF 대비 강점 = "정확도 압도"가 아니라 "근거 추적/검증/투명성".

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
xai-bundle: 15개 파일 (evidence_inventory.csv 등) - 09 문서의 contract 그대로
report: final_report.md / final_report.docx / dashboard/index.html

작업 순서:
1. 빈 입력에서도 placeholder bundle/report가 깨지지 않게.
2. B/C의 smoke 결과 들어오면 row 1개 실제 채움.
3. 모든 claim에 source artifact 파일 경로 연결.
4. 통계 미확증 내용은 확정 claim으로 쓰지 말 것.

금지 표현:
"XAI가 모델 개선 원인을 증명했다"
"XAI 결과를 보고 VADER를 설계했다"

권장 표현:
"통제된 ablation 조건 간 설명 패턴 차이를 사후 검증한다"
"여러 독립 지표가 같은 방향을 보인다"
```

## 완료 보고 양식
```
[v2 작업 완료]
담당: XAI Bundle + Report
수정한 파일:
실행한 명령어:
생성/변경된 산출물: (bundle 15개 + report 3개)
통과한 검증: JSON 유효성 / claim source 연결 / placeholder 동작
남은 위험:
다음 사람이 이어받을 부분: (E가 full run gate 검증)
```

---
---

# 카드 E — QA + Server + Integration

## 한 줄 정의
모든 stage가 깨지지 않게 매일 검증하고, 서버 시간 낭비 안 되게 관문을 지킨다. 4명의 변경사항을 통합한다.

## 너의 책임
1. **preflight**: 매일 dry-run / status / compile 명령 묶음 실행.
2. **manifest / status**: `plan` stage로 manifest 생성, `status`로 completed/failed/planned 정확 보고.
3. **Full Run Gate**: 6조건 통과 전까지 full 120 benchmark 절대 금지.
4. **Integration**: 다른 사람 PR 합칠 때 산출물 계약 안 깨지는지 검증.
5. **서버 운영**: 첫 실행은 smoke. 실패는 unit만 재실행.

## 너의 파일 (수정 OK)
```
v2/pipeline/cli.py
v2/pipeline/manifest.py
v2/pipeline/artifacts.py
v2/run.sh
v2/scripts/validate_commit_message.py
v2/configs/v2_15seed.json
```

## 협의 후 (필요 시)
```
v2/pipeline/runner.py
v2/pipeline/schema.py
```

## 절대 건드리지 마
```
v2/runtime/                     (전체)
v2/pipeline/statistics.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py
v1/
```

## 첫 주 D0~D10 체크리스트
- [ ] **D0**: 모든 사람 D0 요약 모아서 단톡에 \"현재 분담 정리\" 박기. 아래 문서 정독.
  - `v2/docs/agent_tasks/00_common_agent_rules.md`
  - `v2/docs/agent_tasks/05_qa_server_agent.md`
  - `v2/docs/agent_tasks/06_integration_lead_agent.md`
  - `v2/docs/agent_tasks/07_review_agent.md`
  - `v2/docs/11_team_tasking_and_server_run_plan.md` ★ (657줄)
  - `v2/docs/06_execution_runbook.md`
- [ ] **D1**: 매일 검증 명령 묶음을 단톡 자동 보고 형식으로 정리.
- [ ] **D2**: A의 smoke 준비 도와주기 (CLI/manifest 점검).
- [ ] **D3**: A의 smoke 통과 확인 + `execution_status.csv` 신뢰성 검증.
- [ ] **D4**: Full Run Gate 6조건 모두 통과 점검. paired smoke 확인.
- [ ] **D5-D6**: 서버에서 full 120 benchmark 실행. 모니터링.
- [ ] **D7+**: 실패 unit 회수. B/C/D stage 도움.

## 매일 돌릴 검증 명령 (복붙용, daily.sh로 묶어두면 편함)
```bash
cd v2

# 컴파일 / config 유효성
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json

# CLI 도움말 / status / dry-run
./run.sh e2e --help
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run

# 빈 입력 stage 안 깨지는지
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

## Full Run Gate (이 6조건 다 통과 전엔 full 120 절대 금지)
```
1. v2_runtime_import_smoke 통과               (A)
2. A_B seed 42 단일 smoke 성공                (A)
3. A_B/D_B seed 42 paired smoke 성공          (A)
4. metrics.json/history.csv/run_config.json/predictions.csv/checkpoint 생성
5. aggregate가 smoke 결과 읽고 paired_tests row 생성   (B)
6. checkpoint_path / predictions_path가 v2 내부 경로
```

## 서버 실행 순서
**첫 접속 시**:
```bash
git status --short
python3 --version
nvidia-smi
python3 -m compileall pipeline scripts/validate_commit_message.py
./run.sh e2e status --run-id v2_15seed
```

**smoke run**:
```bash
PYTHON_BIN=/path/to/venv/bin/python \
  ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
```

**smoke 통과 후 full**:
```bash
PYTHON_BIN=/path/to/venv/bin/python \
  ./run.sh e2e benchmark --run-id v2_15seed --resume
```

## 즉시 중단 기준 (전체 멈춰)
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

## 너만 할 수 있는 한 가지
다른 사람이 자기 파일 밖을 건드리면 \"왜?\" 물어보고 정리해줘.
모든 통합 충돌의 마지막 결정자.

## 막힐 때
- **페어**: 전원. 너는 모두의 통합 지점.
- **권한**: full run gate에서 \"안 됨\" 결정 권한은 너에게.

## 에이전트에 던질 첫 문장
```
당신은 HateSpeachStudy v2_15seed 파이프라인의 QA + Server + Integration 담당 에이전트입니다.

목표:
1. 매일 dry-run/status/compile 명령으로 stage 깨짐 조기 감지.
2. Full Run Gate 6조건 통과 전까지 full 120 benchmark 차단.
3. 4명 (Benchmark/Statistics/XAI Core/XAI Bundle+Report)의 변경사항을
   산출물 계약이 깨지지 않게 통합.
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

매일 검증 명령 (daily preflight):
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
./run.sh e2e --help
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

Full Run Gate 6조건 (다 통과 전엔 full 120 금지):
1. v2_runtime_import_smoke 통과
2. A_B seed 42 단일 smoke 성공
3. A_B/D_B seed 42 paired smoke 성공
4. metrics/history/config/predictions/checkpoint 생성
5. aggregate가 smoke 결과 읽고 paired_tests row 생성
6. checkpoint_path/predictions_path가 v2 내부 경로

서버 실행 정책:
- full benchmark / full XAI 임의 실행 절대 금지.
- 첫 실행은 smoke (A_B/D_B seed 42).
- 실패 시 전체 재실행 X — failed unit만.

즉시 중단 기준 (있으면 stage 멈춤):
split hash 다름, sample order 다름, VADER가 A/B 조건 들어감,
attention loss가 rationale 없는 샘플에 적용, metrics schema 조건마다 다름,
NaN loss 반복, checkpoint가 run_id 외부.
```

## 완료 보고 양식
```
[v2 integration report]
Merged roles:
Files touched:
Commands run:
Pass:
Fail:
Blocking issue:
Ready for server smoke: yes/no
```

---
---

# 부록 A. 5개 카드 한 페이지 요약 (포스트잇용)

```
A. Benchmark
   → 학습 1개 실행되게 만들기. NVIDIA 환경 호환.
   → 파일: runner.py / training_adapter.py / artifacts.py / schema.py / utils.py
   → D3 목표: A_B seed 42 smoke 통과

B. Statistics
   → 120 metrics.json → mean/std + 핵심 paired t-test + Cohen's d CSV
   → 파일: statistics.py / schema.py
   → D3 목표: fake metrics로 paired row 검증

C. XAI Core
   → SHAP/LIME 돌려서 7개 CSV + JSON 떨어뜨리기
   → 파일: xai.py / schema.py
   → D3 목표: primary 1 seed × 50 sample smoke

D. XAI Bundle + Report
   → C 결과 + B 통계 → evidence bundle 15개 + report 3개
   → 파일: xai_bundle.py / reporting.py
   → D4 목표: placeholder report에 row 1개 채움

E. QA + Server + Integration
   → 매일 preflight, Full Run Gate, 통합, 서버 운영
   → 파일: cli.py / manifest.py / artifacts.py / run.sh / config.json
   → D10까지 상시
```

---

# 부록 B. 모두에게 공통 — 첫 에이전트 호출 직전에 박을 컨텍스트

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
- 자기 담당 파일 밖 수정은 사유 + 변경 범위 보고.
- full benchmark / full XAI는 QA 담당 승인 전 절대 실행 금지.

내 역할: [카드 X 본문 통째로 붙여넣기]
```

---

# 부록 C. 매주 1회 교차 리뷰 (07 Review 책임 분담)

```
주 1회 정기 리뷰 (15분):
  Week 1: A ↔ E (CLI/adapter 정합)
  Week 2: B ↔ D (통계 → 보고서 컬럼 정합)
  Week 3: C ↔ D (XAI → bundle 입력 계약 정합)
  Week 4: 전체 모임 (full run gate 점검)
```

리뷰 양식: `v2/docs/agent_tasks/07_review_agent.md` 5장 \"리뷰 출력 형식\" 그대로.

---

작성: 2026-05-17
참조: `v2/docs/14_team_assignment_matrix.md`, `v2/docs/15_runtime_code_validation_matrix.md`, `v2/docs/agent_tasks/*.md`
