# 20. Claude Code Completion Brief

> Claude Code (또는 다른 코딩 에이전트)에게 던지는 \"v2 코드 완성\" 작업 명세.
> 사람은 명령 실행 / 결과 검수만 한다. 코드 작성은 에이전트가 한다.
> 이 문서 전체를 통째로 던지거나, 작업 #N 섹션 하나씩 떼서 던져도 된다.
>
> **상태 (2026-05-17)**: 작업 #1~#14 모두 완료. 본 문서는 이제 검토 기록·재현 가이드용. 5인 분배는 [`21_team_role_dispatch.md`](21_team_role_dispatch.md), 회의용 카드는 [`22_stage_briefs.md`](22_stage_briefs.md) 참조.

---

## 작업 진행 상태 (요약)

| 라운드 | 작업 | 우선순위 | 완료 시점 | 커밋 |
|---|---|---|---|---|
| 1차 | #1 cudnn 결정성 | P2 | 완료 | `2bdb54a` |
| 1차 | #2 ANOVA 2-way/3-way | P1 | 완료 | `2bdb54a` |
| 1차 | #3 failed/completed + daily.sh | P1 | 완료 | `2bdb54a` |
| 1차 | #4 XAI Core adapter | P0 | 완료 | `99df433` |
| 1차 | #5 XAI Bundle + Report | P0 | 완료 | `99df433` + `fd5e8e7` |
| 2차 | #6 statsmodels 의존 | P2 | 완료 | `25f52be` |
| 2차 | #7 Bootstrap CI | P1 | 완료 | `25f52be` |
| 2차 | #8 XAI 4축 (ablation) | P1 | 완료 | `25f52be` |
| 2차 | #9 token_attributions.jsonl | P2 | 완료 | `25f52be` |
| 2차 | #10 subgroup × context | P1 | 완료 | `25f52be` |
| 3차 | **#11 primary 4축** | P1 | 완료 | `480c4eb` |
| 3차 | **#12 AMP autocast** | P1 | 완료 | `480c4eb` |
| 3차 | **#13 Gate 자동 판정** | P1 | 완료 | `480c4eb` |
| 3차 | **#14 ANOVA effect size + sample-level subgroup** | P1 | 완료 | `480c4eb` |

코드 완성도 (5 Stage 종합): **모두 100%**. NVIDIA 서버 smoke 시작 가능 상태.

---

## 0. 컨텍스트 (모든 작업 공통)

```
프로젝트: HateSpeachStudy v2_15seed
저장소: WinterFlw/Big_data_Programming
작업 폴더: v2/
Run ID: v2_15seed
산출물 루트: outputs/experiments/v2_15seed/
CLI 진입점: ./run.sh e2e ...
Config: configs/v2_15seed.json
설계 문서: v2/docs/
```

### 원칙

```
v1/은 archive. 절대 import 안 함.
모든 산출물은 outputs/experiments/v2_15seed/ 아래.
runtime/은 학습/XAI 무거운 코드 (v1에서 이식, 거의 완성).
pipeline/은 orchestration + adapter (얇은 코드, 일부 placeholder).
완성된 작업: training_adapter, runner, statistics 기본, cli, manifest, artifacts.
빠진 작업: XAI adapter, bundle/report 채움, ANOVA, CUDA 시드 고정 강화.
```

### 검증 기본 명령 (작업 후 매번 돌릴 것)

```bash
cd v2
python3 -m compileall pipeline
python3 -m json.tool configs/v2_15seed.json >/dev/null && echo "config ok"
./run.sh e2e --help
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

각 명령이 깨지지 않고 결과를 출력해야 한다. 깨지면 작업 #N의 변경이 회귀를 일으킨 것.

---

## 작업 우선순위 매트릭스

### 1차 라운드 (코드 골격 완성)

| # | 작업 | 우선순위 | 분량 | 의존 | 상태 |
|---|---|---|---|---|:---:|
| 1 | NVIDIA CUDA 시드 고정 강화 (`utils.py`) | P2 | 5줄 | 없음 | ✓ |
| 2 | ANOVA 2-way / 3-way 추가 (`statistics.py`) | P1 | 100줄 | 없음 | ✓ |
| 3 | failed/completed CSV 분리 + daily.sh (`artifacts.py` + 신규) | P1 | 80줄 | 없음 | ✓ |
| 4 | **XAI Core adapter** (`pipeline/xai.py` 갈아엎기) | **P0** | **400~600줄** | runtime/experiment_xai.py 호출 | ✓ |
| 5 | **XAI Bundle + Report 채움** (`xai_bundle.py` + `reporting.py`) | **P0** | **300~500줄** | 작업 #4 산출물 | ✓ |

### 2차 라운드 (완성도 채움)

| # | 작업 | 우선순위 | 분량 | 의존 | 상태 |
|---|---|---|---|---|:---:|
| 6 | statsmodels 의존 명시 (`runtime/requirements.txt`) | P2 | 1줄 | 없음 | ✓ |
| 7 | Bootstrap CI (`pipeline/statistics.py`) | P1 | 60줄 | 작업 #2 | ✓ |
| 8 | XAI 4축 메트릭 (ablation) 실제 계산 | P1 | 80줄 | 작업 #4 | ✓ |
| 9 | `token_attributions.jsonl` cache→jsonl 평탄화 | P2 | 50줄 | 작업 #4 | ✓ |
| 10 | subgroup × context (source × target 분해, context window) | P1 | 90줄 | 작업 #5 | ✓ |

### 3차 라운드 (정밀 완성)

| # | 작업 | 우선순위 | 분량 | 의존 | 상태 |
|---|---|---|---|---|:---:|
| 11 | **primary seed_level_metrics에도 4축** (schema 18컬럼) | P1 | 70줄 | 작업 #4 + #8 | ✓ |
| 12 | **AMP autocast** (`runtime/experiment_core.py` 학습 루프) | P1 | 30줄 | 작업 #1 | ✓ |
| 13 | **Gate 자동 판정** (`scripts/gate_check.py` 신규 + daily.sh 통합) | P1 | 180줄 | 작업 #3 | ✓ |
| 14 | **ANOVA effect size + sample-level subgroup** | P1 | 180줄 | 작업 #2 + #5 + #10 | ✓ |

권장 진행 순서: 1 → 2 → 3 (가벼운 거 먼저) → 4 → 5 → 6~10 (완성도) → 11~14 (정밀).

---

# 작업 #1 — NVIDIA CUDA 시드 고정 강화

## 무엇

`v2/runtime/utils.py`의 `set_seed` 함수에 `cudnn.deterministic` / `cudnn.benchmark` 설정 추가.

## 왜

현재 `set_seed`는 random/numpy/torch seed만 고정. NVIDIA GPU에서 같은 seed로 두 번 돌려도 cudnn의 비결정적 알고리즘 선택 때문에 결과가 미세하게 달라질 수 있음. 시드 변동 측정이 본 연구의 핵심 결과물 (15 seed paired test)이라 결정성 보장 필수.

## 파일 / 위치

`v2/runtime/utils.py` 라인 115~130 부근 `set_seed` 함수.

## 현재 코드

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.mps, \"manual_seed\") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
```

## 변경 후 (추가할 부분)

`torch.cuda.manual_seed_all(seed)` 뒤에 아래 두 줄 추가:

```python
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

전체 함수는:

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.mps, \"manual_seed\") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
```

## 검증

```bash
python3 -c \"
import sys
sys.path.insert(0, 'runtime')
from utils import set_seed
import torch
set_seed(42)
assert torch.backends.cudnn.deterministic == True or not torch.cuda.is_available()
assert torch.backends.cudnn.benchmark == False or not torch.cuda.is_available()
print('cudnn settings ok')
\"
```

## 주의

- `cudnn.benchmark = False`는 약간 (5~10%) 느려질 수 있다. 그러나 시드 재현성이 본 연구의 메인 메시지라 절대 끄지 말 것.
- MPS는 영향 없음.

---

# 작업 #2 — ANOVA 2-way / 3-way 추가

## 무엇

`v2/pipeline/statistics.py`에 ANOVA 함수 2개 추가, `aggregate()` 함수가 ANOVA CSV 2개를 추가로 생성하도록 확장.

## 왜

`v2/docs/02_e2e_pipeline.md` 6장의 산출물 명세에 `anova_2way_bert.csv` / `anova_3way.csv`가 명시되어 있지만 현재 코드는 paired test만 함. ANOVA는 \"Attn 주효과 / VADER 주효과 / 교호작용\" 분해 보고에 필요.

## 파일 / 위치

`v2/pipeline/statistics.py`. 기존 함수 보존하고 끝에 추가.

## 추가할 함수 명세

### `compute_two_way_anova(rows, family='BERT', metric='macro_f1') -> list[dict]`

BERT 또는 RoBERTa 패밀리 4개 조건 (A/B/C/D)을 `attention_loss` × `vader` 2-way로 ANOVA.

입력: `rows` (collect_benchmark_runs의 출력), `family` ('BERT' or 'RoBERTa'), `metric`.
출력 컬럼: `family, metric, factor, sum_sq, df, F, p_value`. factor는 `C(attention_loss)`, `C(vader)`, `C(attention_loss):C(vader)`, `Residual` 4개.

구현 힌트:
```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df = pd.DataFrame([r for r in rows if r['backbone'] == family and r['status'] == 'completed' and r[metric] != ''])
df[metric] = df[metric].astype(float)
df['attention_loss'] = df['use_attention_loss'].astype(str)
df['vader'] = df['use_sentiment'].astype(str)

model = ols(f'{metric} ~ C(attention_loss) * C(vader)', data=df).fit()
table = anova_lm(model, typ=2)
```

ANOVA가 충분한 표본 없을 때 (예: 한 조합에 시드 1개 미만)는 빈 리스트 반환.

### `compute_three_way_anova(rows, metric='macro_f1') -> list[dict]`

`backbone` × `attention_loss` × `vader` 3-way. 8조건 전체 대상.

출력 컬럼: `metric, factor, sum_sq, df, F, p_value`. factor는 단일/2차/3차 교호작용 모두.

구현 힌트:
```python
formula = f'{metric} ~ C(backbone) * C(attention_loss) * C(vader)'
```

### `aggregate()` 수정

기존 `aggregate()` 끝에 추가:

```python
anova_bert = compute_two_way_anova(rows, family='BERT')
anova_roberta = compute_two_way_anova(rows, family='RoBERTa')
anova_3way = compute_three_way_anova(rows)

anova_bert_path = write_csv(
    benchmark_dir / 'anova_2way_bert.csv',
    anova_bert,
    ['family', 'metric', 'factor', 'sum_sq', 'df', 'F', 'p_value']
)
anova_roberta_path = write_csv(
    benchmark_dir / 'anova_2way_roberta.csv',
    anova_roberta,
    ['family', 'metric', 'factor', 'sum_sq', 'df', 'F', 'p_value']
)
anova_3way_path = write_csv(
    benchmark_dir / 'anova_3way.csv',
    anova_3way,
    ['metric', 'factor', 'sum_sq', 'df', 'F', 'p_value']
)

return {
    \"benchmark_runs\": run_path,
    \"benchmark_summary\": summary_path,
    \"paired_tests\": paired_path,
    \"paired_tests_holm\": holm_path,
    \"anova_2way_bert\": anova_bert_path,
    \"anova_2way_roberta\": anova_roberta_path,
    \"anova_3way\": anova_3way_path,
}
```

## 검증

```bash
cd v2
./run.sh e2e aggregate --run-id v2_15seed
ls outputs/experiments/v2_15seed/benchmark/anova_*.csv

python3 - <<'PY'
import sys; sys.path.insert(0, '.')
from pipeline.statistics import compute_two_way_anova
rows = []
import random
random.seed(0)
# 8조건 × 3시드 fake metrics
for seed in [42, 52, 62]:
    for cond, attn, vad, bb in [
        ('A_B', False, False, 'BERT'),
        ('B_B', True, False, 'BERT'),
        ('C_B', False, True, 'BERT'),
        ('D_B', True, True, 'BERT'),
    ]:
        rows.append({
            'condition': cond, 'seed': seed, 'status': 'completed',
            'backbone': bb, 'use_attention_loss': attn, 'use_sentiment': vad,
            'macro_f1': 0.6 + (0.05 if attn else 0) + (0.03 if vad else 0) + random.gauss(0, 0.01)
        })
result = compute_two_way_anova(rows, family='BERT')
assert len(result) >= 3, f'expected at least 3 factor rows, got {len(result)}'
assert any('attention_loss' in r['factor'] for r in result)
print('two_way_anova_smoke: ok')
PY
```

## 주의

- `statsmodels`는 이미 `runtime/requirements.txt`에 들어 있을 가능성 높음. 없으면 `pip install statsmodels`.
- 결측 시드/조건에서 안 깨지게 try/except로 감싸고, 실패 시 빈 리스트 반환.
- `df` 컬럼명이 pandas DataFrame 변수명과 겹치지 않게 (예: ANOVA 출력 `df`는 \"degrees of freedom\").

---

# 작업 #3 — failed/completed CSV 분리 + daily.sh

## 무엇

1. `v2/pipeline/artifacts.py`에 `write_failed_completed_csv()` 함수 추가. `execution_status.csv`에서 failed/completed unit을 별도 파일로 추출.
2. `v2/scripts/daily.sh` 신규 작성. QA 담당이 매일 한 번 돌릴 preflight 스크립트.

## 왜

`v2/docs/agent_tasks/05_qa_server_agent.md` 의 완료 기준에 `failed_runs.csv`, `completed_runs.csv`가 명시됨. 현재는 `execution_status.csv` 하나에 다 들어 있어 \"실패한 것만 빠르게 뽑기\"가 grep 필요. QA가 매일 모니터링하니까 별도 파일이 편함.

## 작업 #3a — failed/completed CSV 분리

### 파일

`v2/pipeline/artifacts.py` 끝에 추가.

### 추가 함수

```python
def write_failed_completed_csv(manifest: dict[str, Any], units: list[RunUnit]) -> dict[str, Path]:
    \"\"\"Extract failed and completed units into separate CSVs for QA monitoring.\"\"\"
    root = experiment_root(manifest[\"run_id\"])
    columns = [
        \"run_id\", \"condition\", \"seed\", \"backbone\", \"model_name\",
        \"use_attention_loss\", \"use_sentiment\", \"status\", \"run_dir\",
    ]
    failed_rows = []
    completed_rows = []
    for unit in units:
        metadata = unit.metadata
        status = unit_status(unit)
        row = {
            \"run_id\": unit.run_id,
            \"condition\": unit.condition,
            \"seed\": unit.seed,
            \"backbone\": metadata[\"backbone\"],
            \"model_name\": metadata[\"model_name\"],
            \"use_attention_loss\": metadata[\"use_attention_loss\"],
            \"use_sentiment\": metadata[\"use_sentiment\"],
            \"status\": status,
            \"run_dir\": display_path(unit.run_dir),
        }
        if status == \"failed\":
            failed_rows.append(row)
        elif status == \"completed\":
            completed_rows.append(row)

    failed_path = write_csv(root / \"failed_runs.csv\", failed_rows, columns)
    completed_path = write_csv(root / \"completed_runs.csv\", completed_rows, columns)
    return {\"failed\": failed_path, \"completed\": completed_path}
```

### `runner.status()` 수정

`v2/pipeline/runner.py`의 `status()` 함수가 위 함수를 호출하도록:

```python
from .artifacts import (
    build_run_units, ensure_experiment_tree, status_counts,
    write_stage_marker, write_unit_plan, write_failed_completed_csv,
)

def status(run_id: str = DEFAULT_RUN_ID, manifest_path: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path, run_id=run_id)
    errors = validate_manifest(manifest)
    if errors:
        return {\"valid\": False, \"errors\": errors}
    ensure_experiment_tree(run_id)
    units = build_run_units(manifest)
    status_path = write_unit_plan(manifest, units)
    counts = status_counts(units)
    extras = write_failed_completed_csv(manifest, units)
    return {
        \"valid\": True,
        \"counts\": counts,
        \"execution_status\": status_path,
        \"failed_runs\": extras[\"failed\"],
        \"completed_runs\": extras[\"completed\"],
    }
```

## 작업 #3b — daily.sh 신규

### 파일

`v2/scripts/daily.sh` 새로 생성. 실행 권한 (`chmod +x v2/scripts/daily.sh`).

### 내용

```bash
#!/usr/bin/env bash
# Daily preflight for v2_15seed pipeline (QA stage owner's daily tool).
# Run from repo root: ./v2/scripts/daily.sh

set -e

cd \"$(cd \"$(dirname \"${BASH_SOURCE[0]:-$0}\")\" && pwd -P)/..\"

echo \"=== compile pipeline ===\"
python3 -m compileall pipeline scripts/validate_commit_message.py

echo \"\"
echo \"=== config validation ===\"
python3 -m json.tool configs/v2_15seed.json > /tmp/v2_config_check.json && echo \"config ok\"

echo \"\"
echo \"=== CLI help ===\"
./run.sh e2e --help > /dev/null && echo \"cli help ok\"

echo \"\"
echo \"=== plan / status ===\"
./run.sh e2e plan --run-id v2_15seed --force | tail -5
./run.sh e2e status --run-id v2_15seed | tail -5

echo \"\"
echo \"=== dry-run benchmark ===\"
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run | tail -5

echo \"\"
echo \"=== scaffolding stages (placeholder safe) ===\"
./run.sh e2e aggregate --run-id v2_15seed | tail -5
./run.sh e2e xai-bundle --run-id v2_15seed | tail -3
./run.sh e2e report --run-id v2_15seed | tail -3
./run.sh e2e dashboard --run-id v2_15seed | tail -3

echo \"\"
echo \"=== failed runs ===\"
if [ -f outputs/experiments/v2_15seed/failed_runs.csv ]; then
    failed_count=$(tail -n +2 outputs/experiments/v2_15seed/failed_runs.csv | wc -l)
    echo \"failed_runs.csv: $failed_count rows\"
else
    echo \"failed_runs.csv: not yet generated\"
fi

echo \"\"
echo \"=== completed runs ===\"
if [ -f outputs/experiments/v2_15seed/completed_runs.csv ]; then
    completed_count=$(tail -n +2 outputs/experiments/v2_15seed/completed_runs.csv | wc -l)
    echo \"completed_runs.csv: $completed_count rows\"
else
    echo \"completed_runs.csv: not yet generated\"
fi

echo \"\"
echo \"[daily preflight ok]\"
```

## 검증

```bash
cd v2
chmod +x scripts/daily.sh
./scripts/daily.sh

# failed/completed 파일 생성 확인
ls outputs/experiments/v2_15seed/failed_runs.csv outputs/experiments/v2_15seed/completed_runs.csv
```

## 주의

- daily.sh의 `pwd -P`는 심볼릭 링크 해결. 절대 경로 이동 필수.
- failed_runs.csv는 처음엔 헤더만 (실제 실패 run 없음).
- runner.status()가 위 두 파일을 생성하니까 daily.sh가 그 다음에 읽으면 됨.

---

# 작업 #4 — XAI Core adapter (★ 가장 큰 작업)

## 무엇

`v2/pipeline/xai.py`를 placeholder에서 \"실제 SHAP/LIME 호출\" adapter로 갈아엎는다. `pipeline/training_adapter.py`가 `runtime/experiment_core.py`를 호출하는 패턴을 그대로 따라, `pipeline/xai.py`가 `runtime/experiment_xai.py`의 함수를 호출하도록.

## 왜

- runtime/experiment_xai.py (99KB)에 SHAP, LIME, 서브워드 집계, predict_probabilities 등 완성된 함수가 다 있음.
- pipeline/xai.py (5KB)는 빈 CSV 헤더만 박는 placeholder.
- 분배 가능한 형태로 만들려면 pipeline/xai.py가 실제로 결과 row를 채워야 한다.

## 파일 / 위치

- 갈아엎기: `v2/pipeline/xai.py` (현재 5KB → 최종 400~600줄 예상).
- 참고만: `v2/runtime/experiment_xai.py` (수정 금지).
- 새 모듈 가능: `v2/pipeline/xai_sampling.py` (sample 추출 분리하면 좋음).

## 단계별 작업

### 4-1. runtime 모듈 import 패턴

`pipeline/training_adapter.py`의 `_load_runtime_core()` 패턴을 그대로 복제:

```python
import sys
from pathlib import Path

from .paths import BASE_DIR

RUNTIME_DIR = BASE_DIR / \"runtime\"

def _load_runtime_xai() -> Any:
    \"\"\"Import v2/runtime experiment_xai after making runtime importable.\"\"\"
    runtime_path = str(RUNTIME_DIR)
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    for module_name in [\"utils\", \"experiment_xai\"]:
        module = sys.modules.get(module_name)
        module_file = getattr(module, \"__file__\", \"\") if module else \"\"
        if module_file and not str(module_file).startswith(runtime_path):
            del sys.modules[module_name]
    import experiment_xai  # type: ignore[import-not-found]
    return experiment_xai
```

### 4-2. Stratified sample 추출 (seed 간 동일성 보장)

`pipeline/xai_sampling.py` 신규:

```python
def select_primary_samples(runtime_xai, manifest, sample_size=200) -> list[dict]:
    \"\"\"Stratified 200 samples from test split. Seed-independent (fixed across all seeds).\"\"\"
    # runtime_xai.load_test_dataset() 또는 동등 함수 호출
    # label-stratified split (3-class 비율 보존)
    # seed=0으로 numpy.random 고정해서 모든 seed에서 동일 sample 선택
    # 반환: [{\"sample_id\", \"text\", \"label\", \"rationale_mask\", \"target\", \"source\"}, ...]
    ...

def select_deep_samples(runtime_xai, manifest, sample_size=500) -> list[dict]:
    \"\"\"Median seed × 500 stratified samples for qualitative cases.\"\"\"
    ...

def select_ablation_samples(runtime_xai, manifest, sample_size=50) -> list[dict]:
    \"\"\"50 samples for ablation matrix.\"\"\"
    ...
```

핵심: sample_id가 seed에 의존하지 않고 manifest에만 의존해야 한다. `numpy.random.RandomState(seed=0)` 고정.

### 4-3. SHAP/LIME 메트릭 계산

`runtime/experiment_xai.py`의 함수를 호출:

```python
def compute_primary_xai_metrics(runtime_xai, bundle, samples) -> list[dict]:
    \"\"\"For one (condition, seed) checkpoint, compute 10 metrics across the sample set.\"\"\"
    # 1. predict_probabilities for all samples
    probs = runtime_xai.predict_probabilities(bundle, [s['text'] for s in samples])

    # 2. SHAP explanations (CPU)
    shap_results = runtime_xai.run_shap_explanations(bundle, samples, predicted_labels=probs.argmax(-1))

    # 3. LIME explanations
    lime_results = runtime_xai.run_lime_explanations(bundle, samples)

    # 4. Aggregate to metrics
    metrics = {
        'shap_lime_overlap_at_5': mean([overlap_at_k(s['tokens'], l['tokens'], k=5) for s, l in zip(shap_results, lime_results)]),
        'shap_lime_overlap_at_10': ...,
        'rationale_precision_at_5': mean([precision_at_k(s['tokens'], sample['rationale_mask'], k=5) for s, sample in zip(shap_results, samples)]),
        'rationale_recall_at_5': ...,
        'rationale_f1_at_5': ...,
        'comprehensiveness': mean([comp_at_k(bundle, sample['text'], s['tokens'], k=5) for s, sample in zip(shap_results, samples)]),
        'sufficiency': ...,
        'loo_drop': ...,
        # topk_jaccard / rank_corr는 seed 간 비교라 여기서는 placeholder.
    }
    return metrics
```

helper 함수들 (overlap_at_k, precision_at_k, comp_at_k 등)은 runtime/experiment_xai.py에 이미 있을 수 있음 (grep으로 찾기). 없으면 새로 작성.

### 4-4. Seed stability 메트릭 (15시드 결과를 모은 후)

`compute_seed_stability(per_seed_metrics) -> dict`:
- Top-k Jaccard: 같은 sample, 다른 seed checkpoint의 top-5 token이 얼마나 일치.
- Rank correlation: SHAP token ranking의 Spearman corr (seed 간).

### 4-5. `plan_primary_xai()` 갈아엎기

기존 placeholder를 다음 흐름으로:

```python
def plan_primary_xai(manifest, dry_run=False):
    runtime_xai = _load_runtime_xai()
    root = experiment_root(manifest['run_id'])

    if dry_run:
        return {'status': 'dry-run', 'sample_size': manifest['xai']['primary']['sample_size']}

    # 1. Fixed sample set
    samples = select_primary_samples(runtime_xai, manifest, sample_size=manifest['xai']['primary']['sample_size'])
    sample_path = write_csv(root / 'xai' / 'samples' / 'primary_samples.csv', samples,
                            ['sample_id', 'label', 'case_type', 'source', 'target', 'text'])

    # 2. Per (condition, seed) — 다만 checkpoint 없으면 skip
    primary_rows = []
    for condition in manifest['xai']['primary']['models']:  # ['A_B', 'D_B']
        for seed in manifest['benchmark']['seeds']:
            ckpt = root / 'benchmark' / 'checkpoints' / f'{condition.lower()}_seed_{seed}.pt'
            if not ckpt.exists():
                continue
            bundle = runtime_xai.load_bundle_from_checkpoint(condition, seed, ckpt)
            metrics = compute_primary_xai_metrics(runtime_xai, bundle, samples)
            metrics.update({'run_id': manifest['run_id'], 'condition': condition, 'seed': seed, 'sample_count': len(samples)})
            primary_rows.append(metrics)

    metric_path = write_csv(root / 'xai' / 'primary' / 'seed_level_metrics.csv', primary_rows, XAI_SEED_METRIC_COLUMNS)

    # 3. Paired XAI tests (A_B vs D_B 같은 seed)
    paired_path = ... # statistics.py의 paired test 패턴 재사용

    # 4. Seed stability
    stability_path = ... # 위 4-4

    return {
        'samples': sample_path,
        'seed_metrics': metric_path,
        'paired_tests': paired_path,
        'seed_stability': stability_path,
    }
```

### 4-6. plan_deep_xai / plan_ablation_xai 동일 패턴

deep은 median seed 1개 × 500 sample. ablation은 8조건 × median seed × 50 sample.

### 4-7. checkpoint 없을 때 (smoke 단계) graceful skip

모든 함수가 \"checkpoint 0개\"에서도 빈 CSV 정상 생성해야 한다. training이 아직 안 끝났을 때도 dry-run 가능.

## 검증

```bash
cd v2

# dry-run 동작
./run.sh e2e xai-primary --run-id v2_15seed --dry-run
./run.sh e2e xai-deep --run-id v2_15seed --dry-run
./run.sh e2e xai-ablation --run-id v2_15seed --dry-run

# checkpoint 0개에서 빈 결과 생성
./run.sh e2e xai-primary --run-id v2_15seed
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed

# 산출물 파일 7개 다 생성됐는지
ls -la outputs/experiments/v2_15seed/xai/samples/primary_samples.csv \\
       outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv \\
       outputs/experiments/v2_15seed/xai/primary/paired_xai_tests.csv \\
       outputs/experiments/v2_15seed/xai/primary/seed_stability.csv \\
       outputs/experiments/v2_15seed/xai/deep/case_summary.csv \\
       outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv \\
       outputs/experiments/v2_15seed/xai/xai_summary.json

# (실제 GPU smoke 후) 1 seed × 50 sample smoke
# benchmark smoke checkpoint 있을 때:
./run.sh e2e xai-primary --run-id v2_15seed --resume
column -t -s, outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv | head -5
```

## 주의

- SHAP은 무조건 CPU로 (`bundle.model.to('cpu')` 후 호출, 끝나면 device 복귀).
- BERT WordPiece (`##`) / RoBERTa BPE (`Ġ`) 서브워드 집계 — runtime/experiment_xai.py의 `_aggregate_subword_scores` 그대로 사용.
- 첫 PR에서 모든 메트릭 완벽 구현 안 되어도 됨. 골격 + 1~2개 메트릭부터 시작해서 점진 확장.
- 실패 케이스 (checkpoint 손상, OOM 등)는 sample-level skip + risk_flags row 기록.
- 절대 SHAP 결과를 캐싱 없이 매번 재계산하지 말 것 (한 sample SHAP 계산이 수 초 ~ 수십 초).

## 권장 캐싱

`outputs/experiments/v2_15seed/xai/.cache/` 안에 (condition, seed, sample_id) → attribution 저장. 재실행 시 캐시 hit이면 skip.

---

# 작업 #5 — XAI Bundle + Report 채움

## 무엇

1. `v2/pipeline/xai_bundle.py`를 placeholder에서 \"실제 row 채움\" 로직으로 갈아엎기.
2. `v2/pipeline/reporting.py`의 markdown/docx에 benchmark + paired test + XAI 결과 표 자동 삽입.

작업 #4가 끝나야 입력이 들어옴. 다만 빈 입력에서도 placeholder는 유지.

## 작업 #5a — `xai_bundle.py` 채움

### 현재 상태

`build_xai_evidence_bundle`이 15개 파일을 만들지만 모두 빈 CSV / `\"status\": \"planned\"` JSON.

### 목표

각 파일이 실제 row를 가지도록 (입력이 있을 때만). 입력 없으면 기존 placeholder 유지.

### 각 파일 채움 로직

#### `xai_predictions.csv`
- 입력: `xai/primary/seed_level_metrics.csv` + sample 메타.
- 채움: sample_id별 (condition, seed) 예측 결과 row.
- 컬럼: `sample_id, condition, seed, true_label, predicted_label, probability, checkpoint_path`.

#### `method_agreement.csv`
- 입력: SHAP/LIME 결과 (xai_primary 산출물 또는 별도 캐시).
- 채움: 각 sample × (condition, seed)별 SHAP-LIME overlap@5/@10, rank correlation.
- 컬럼: `sample_id, condition, seed, overlap_at_5, overlap_at_10, rank_corr, notes`.

#### `faithfulness_metrics.csv`
- 입력: 작업 #4의 comprehensiveness/sufficiency/LOO drop.
- 컬럼: `sample_id, condition, seed, comprehensiveness, sufficiency, loo_drop`.

#### `plausibility_metrics.csv`
- 컬럼: `sample_id, condition, seed, rationale_precision_at_5, rationale_recall_at_5, rationale_f1_at_5`.

#### `context_metrics.csv`
- subgroup별 context window / sensitivity.
- 컬럼: `sample_id, condition, target, source, context_window, context_sensitivity`.

#### `subgroup_xai_metrics.csv`
- target/source 그룹별 메트릭 평균.
- 컬럼: `subgroup, condition, seed, metric, value`.

#### `xai_risk_flags.csv`
- 실패/이상치 sample 표시.
- 컬럼: `sample_id, condition, seed, flag_type, severity, evidence, recommended_report_note`.

#### `xai_claims.json`
- 통계적으로 확증된 주장만 포함.
- 각 claim:
```json
{
  \"id\": \"claim_001\",
  \"text\": \"D_B is more aligned with human rationale than A_B (Token F1 +XX%, Holm-adjusted p < 0.05).\",
  \"strength\": \"strong\",
  \"source_artifacts\": [\"xai/primary/seed_level_metrics.csv\", \"benchmark/paired_tests_holm.csv\"]
}
```

#### `xai_interpretation_cards.json`
- 사람이 읽는 해석 카드. 각 card는 \"이게 무엇이고 어떻게 해석하나\".

#### `xai_dashboard_bundle.json`
- dashboard가 우선 소비할 요약 JSON.
- 핵심 필드: `summary_cards`, `primary`, `seed_stability`, `deep_cases`, `ablation`, `artifact_links`.

#### `token_attributions.jsonl`
- 각 줄: 한 (sample_id, condition, seed)의 token 단위 attribution.
- 형식: `{\"sample_id\": ..., \"condition\": ..., \"seed\": ..., \"tokens\": [...], \"shap_scores\": [...], \"lime_scores\": [...]}`.

## 작업 #5b — `reporting.py` 채움

### 현재 상태

markdown/docx 모두 정적 텍스트. \"Benchmark Status: total/completed/failed\"만 동적.

### 목표

`benchmark_summary.csv` / `paired_tests_holm.csv` / `xai_dashboard_bundle.json`을 읽어 표/문장을 자동 삽입.

### 추가할 섹션

#### Benchmark Summary 표
```python
def _render_benchmark_summary_table(root: Path) -> str:
    summary_path = root / 'benchmark' / 'benchmark_summary.csv'
    if not summary_path.exists():
        return '_no benchmark results yet — placeholder_'
    rows = read_csv_rows(summary_path)
    lines = ['| Condition | N seeds | Macro F1 (mean ± std) | 95% CI |',
             '|---|---|---|---|']
    for row in rows:
        if row.get('n_seeds', '0') == '0':
            continue
        lines.append(f\"| {row['condition']} | {row['n_seeds']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | [{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}] |\")
    return '\\n'.join(lines)
```

#### Paired Test 표
`paired_tests_holm.csv` 7쌍 비교 × 3 metric. 보통 macro_f1만 표시하고 나머지는 부록 링크.

#### XAI Claims 섹션
`xai_dashboard_bundle.json`의 `summary_cards` 읽어서 markdown 변환.

#### TF-IDF baseline 비교 (있으면)
TF-IDF 결과 CSV가 있으면 비교 표 자동 생성.

### Markdown 본문 재구성

```python
def _report_markdown_text(manifest, root, counts):
    return f\"\"\"# v2 Final Report

## Run

- run_id: `{manifest['run_id']}`
- manifest_hash: `{manifest_hash(manifest)}`
- output_root: `{manifest['output_root']}`

## Benchmark Status

- total units: {counts['total']}
- completed: {counts['completed']}
- failed: {counts['failed']}
- planned: {counts['planned']}

## Benchmark Summary

{_render_benchmark_summary_table(root)}

## Paired Tests (Holm-corrected)

{_render_paired_tests_table(root, metric='macro_f1')}

## XAI Evidence Summary

{_render_xai_claims(root)}

## Seed Stability

{_render_seed_stability(root)}

## Limitations

{_render_limitations_text(root)}

## Reproducibility

[명령어 + manifest hash + commit hash + 환경]
\"\"\"
```

### dashboard HTML 강화 (선택)

기존 `generate_dashboard`에 표 더 추가:
- benchmark_summary 표
- paired tests 표
- XAI summary cards

## 검증

```bash
cd v2

# 빈 입력 (placeholder)
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

# 15개 bundle 파일 + 3개 report 파일
ls outputs/experiments/v2_15seed/xai/evidence_bundle/ | wc -l   # 15
ls outputs/experiments/v2_15seed/reports/
ls outputs/experiments/v2_15seed/dashboard/

# JSON 유효성
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json > /dev/null
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json > /dev/null

# (작업 #4의 산출물 있을 때) report에 실제 row 들어가는지
head -50 outputs/experiments/v2_15seed/reports/final_report.md
```

## 주의

- **저장은 full, 노출은 요약** (09 문서 원칙). bundle엔 모든 raw, dashboard/report엔 핵심 주장만.
- 모든 claim에 `source_artifacts` 필드 필수. 출처 없는 claim 금지.
- 통계 미확증 (paired test가 안 돌았거나 결과 없음) 내용을 확정 claim으로 쓰지 마.
- DOCX 출력은 minimal Word (현재 reporting.py 패턴 유지). 한글 폰트 박혀 있지 않으면 영문 보고서로.
- placeholder 유지: 입력 없을 때 \"_no XAI evidence yet — populate by running xai-primary/deep/ablation_\" 같은 메시지.

---

# 부록 A — 작업 완료 후 보고 양식

각 작업이 끝날 때마다 다음 형식으로 보고:

```
[v2 작업 완료 #{N}]
작업 이름:
수정한 파일:
추가한 함수:
실행한 검증 명령:
통과한 검증:
산출물 변화: (예: anova_2way_bert.csv 생성, paired_tests_holm.csv 컬럼 추가 없음)
남은 위험:
다음 작업 의존성: (작업 #4 끝나야 작업 #5의 실제 row 채움 의미 있음)
```

---

# 부록 B — 작업 의존 그래프

```
작업 #1 (cudnn 시드) ────┐
                          ├─→ Stage 1 사람 검증 가능
작업 #3a (failed/completed) ─┐
                              ├─→ Stage 5 사람 검증 가능 (daily.sh)
작업 #3b (daily.sh) ──────────┘

작업 #2 (ANOVA) ──→ Stage 2 사람 검증 가능

작업 #4 (XAI adapter) ──→ Stage 3 사람 검증 가능
                       │
                       ▼
작업 #5 (Bundle + Report) ──→ Stage 4 사람 검증 가능
```

작업 #1, #2, #3은 병렬 가능. 작업 #4 끝나야 #5 실효성 있음.

---

# 부록 C — 작업 종료 후 5명 분배 시점

모든 작업 (#1~#5) 완료되면:

1. **Pilot (1번 사람)**: NVIDIA 서버에서 A_B seed 42 smoke 실행 → 6개 산출물 ls 확인.
2. **Stat Auditor (2번)**: `aggregate` 실행 → benchmark_summary.csv + anova_2way_bert.csv 직접 열어서 숫자 검증.
3. **XAI Curator (3번)**: `xai-primary --resume` 실행 → primary_samples.csv가 seed 간 동일한지 확인 + SHAP top-5 token 1~2개 sanity check.
4. **Author (4번)**: `xai-bundle` + `report` + `dashboard` 실행 → final_report.md 본문 열어서 표/문장 확인 + 발표 자료 골격 시작.
5. **QA Conductor (5번)**: `daily.sh` 매일 실행 → Full Run Gate 6조건 점검 → full 120 benchmark GO/STOP 판단.

이 시점부터 `v2/docs/19_team_role_tracks.md`의 \"D0~D10 체크리스트\" 흐름 그대로.

---

작성: 2026-05-17
참조: `v2/docs/02_e2e_pipeline.md`, `v2/docs/15_runtime_code_validation_matrix.md`, `v2/docs/19_team_role_tracks.md`, `v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md`
