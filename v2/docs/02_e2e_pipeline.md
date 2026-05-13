# 02. End-to-End Pipeline Design

---

## 1. 새 output root

새 실험은 기존 `outputs/reports`, `outputs/xai`, `outputs/runs`를 직접 덮어쓰지 않는다.

기본 root:

```text
outputs/experiments/v2_15seed/
```

권장 구조:

```text
outputs/experiments/v2_15seed/
  manifest.json
  config.json
  data/
    split_profile.json
    split_hash.json
  benchmark/
    runs/
      a_b/seed_42/
      ...
    checkpoints/
    benchmark_runs.csv
    benchmark_summary.csv
    paired_tests.csv
    paired_tests_holm.csv
    anova_2way_bert.csv
    anova_3way.csv
  freeze/
    runs/
    freeze_summary.csv
  xai/
    samples/
    primary/
    deep/
    ablation/
    cases/
    seed_stability.csv
    xai_summary.json
  reports/
    final_report.md
    final_report.docx
  dashboard/
    index.html
```

이렇게 분리하면 기존 3-seed 결과와 새 15-seed 결과가 섞이지 않는다.

---

## 2. End-to-end stage

새 파이프라인은 다음 stage로 나눈다.

| Stage | 역할 | 주요 산출물 |
|---|---|---|
| `plan` | manifest 생성, seed/condition/hyperparameter 고정 | `manifest.json` |
| `data` | split 존재성, hash, 분포 검증 | `split_profile.json` |
| `benchmark` | 8조건 × 15 seed 학습 | run별 metrics/checkpoint |
| `aggregate` | 통계 집계, paired test, ANOVA | summary/test csv |
| `freeze` | 선택: freeze study 반복 | freeze summary |
| `xai-primary` | A_B vs D_B × 15 seed × 200 samples | seed-level XAI metrics |
| `xai-deep` | median seed × 500 samples | case plots, detailed XAI |
| `xai-ablation` | 8조건 × median seed × 50 samples | ablation XAI matrix |
| `report` | 최종 markdown/docx 생성 | final report |
| `dashboard` | run_id 기준 dashboard 생성 | index.html |

---

## 3. CLI 설계

권장 CLI:

```bash
./run.sh e2e plan --run-id v2_15seed
./run.sh e2e data --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e freeze --run-id v2_15seed --resume
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

한 번에 실행:

```bash
./run.sh e2e all --run-id v2_15seed --resume
```

---

## 4. Batch runner 설계

배치 실행 단위는 다음이다.

```text
unit = condition x seed
```

예:

```text
A_B seed 42
A_B seed 52
D_R seed 182
```

완료 판정 기준:

```text
metrics.json exists
history.csv exists
predictions.pkl exists
checkpoint.pt exists
```

`--resume` 옵션은 완료된 unit을 건너뛴다. `--force` 옵션은 기존 unit을 다시 실행한다.

---

## 5. Manifest 기반 실행

실험 실행값은 코드 내부에 흩뿌리지 않고 manifest에 기록한다.

필수 항목:

```text
run_id
created_at
seeds
conditions
primary_metric
common hyperparameters
attention alpha
xai sample policy
output root
```

manifest는 사람이 읽는 설정 파일이면서, 재현성을 위한 audit log다.

---

## 6. Aggregate stage

benchmark가 끝나면 모든 run을 다시 읽어 집계한다.

산출물:

```text
benchmark_runs.csv
benchmark_summary.csv
paired_tests.csv
paired_tests_holm.csv
anova_2way_bert.csv
anova_3way.csv
per_class_summary.csv
subgroup_source_summary.csv
subgroup_target_summary.csv
```

기존 run을 재사용할 수 있어야 하므로 aggregate는 학습 없이 독립 실행 가능해야 한다.

---

## 7. Latest export

새 run_id의 산출물을 확인한 뒤, 필요할 때만 기존 dashboard나 top-level reports로 복사한다.

```text
source: outputs/experiments/v2_15seed/dashboard/index.html
target: outputs/dashboard/index.html
```

기본은 run_id 내부를 canonical output으로 둔다.

