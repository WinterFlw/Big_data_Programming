# 02. Statistics Agent Brief

> 역할: 15 seed benchmark 결과를 통계적으로 집계하고, 학부 수준에 맞게 핵심 paired test와 효과크기 중심으로 정리한다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 Statistics 담당 에이전트입니다.
목표는 outputs/experiments/v2_15seed/benchmark/runs/ 아래의 condition x seed 결과를 읽어 benchmark_summary.csv, paired_tests.csv, paired_tests_holm.csv를 생성하는 것입니다.
같은 seed를 기준으로 핵심 A_B vs D_B paired comparison을 수행해야 하며, p-value만이 아니라 mean difference, 95% CI, effect size를 함께 출력해야 합니다.
Holm-adjusted p-value와 ANOVA는 여러 조건을 함께 볼 때의 보조/부록 분석으로만 다룹니다.
```

---

## 2. 반드시 읽을 문서

```text
docs/03_validation_and_statistics.md
docs/07_output_and_report_contract.md
docs/11_team_tasking_and_server_run_plan.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 소유 파일

우선 수정 가능:

```text
pipeline/statistics.py
pipeline/schema.py
```

필요 시 수정 가능:

```text
pipeline/runner.py
```

가급적 수정하지 않을 파일:

```text
runtime/experiment_core.py
pipeline/xai.py
pipeline/xai_bundle.py
pipeline/reporting.py
```

---

## 4. 구현 목표

아래 기능을 구현한다.

```text
completed run metrics 수집
condition별 mean/std/CI 집계
same-seed paired differences 계산
paired t-test 또는 Wilcoxon fallback
Cohen's dz 또는 paired effect size 계산
Holm-Bonferroni 보정 산출값 생성 (해석은 보조)
missing/failed run 처리
```

---

## 5. 비교 우선순위

```text
본문 핵심:
A_B vs D_B

보조/부록:
A_B vs B_B
A_B vs C_B
B_B vs D_B
C_B vs D_B
A_R vs D_R
D_B vs D_R
```

---

## 6. 완료 기준

아래 명령이 성공해야 한다.

```bash
./run.sh e2e aggregate --run-id v2_15seed
```

생성 파일:

```text
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/benchmark_summary.csv
outputs/experiments/v2_15seed/benchmark/paired_tests.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
```

필수 컬럼:

```text
comparison
metric
condition_a
condition_b
n_pairs
mean_diff
std_diff
ci_low
ci_high
test_name
p_value
p_value_holm
effect_size
significant_0_05
```

---

## 7. 테스트 방법

full run 결과가 없을 때는 fake metrics를 2-3개 run_dir에 넣어 unit test처럼 검증한다.

주의:

```text
fake metrics는 최종 결과로 커밋하지 않는다.
테스트용 파일을 만들었다면 삭제하거나 clearly marked sample fixture로 분리한다.
```

---

## 8. 해석 금지

통계 담당은 CSV를 만든다. 최종 해석 문장은 Report 담당이 결과를 보고 작성한다.

금지 표현:

```text
p-value가 유의하므로 모델이 완전히 개선되었다.
XAI가 성능 향상을 증명한다.
```
