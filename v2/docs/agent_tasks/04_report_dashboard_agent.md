# 04. Report and Dashboard Agent Brief

> 역할: benchmark/statistics/XAI 산출물을 읽어 최종 보고서와 dashboard를 생성한다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 Report/Dashboard 담당 에이전트입니다.
목표는 outputs/experiments/v2_15seed/ 내부 산출물만 읽어서 final_report.md, final_report.docx, dashboard/index.html을 생성하는 것입니다.
결과가 아직 없는 항목은 placeholder로 명확히 표시하고, 임의의 성능 수치나 p-value를 만들어 쓰면 안 됩니다.
```

---

## 2. 반드시 읽을 문서

```text
docs/07_output_and_report_contract.md
docs/08_xai_report_template.md
docs/03_validation_and_statistics.md
docs/04_xai_protocol.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 소유 파일

우선 수정 가능:

```text
pipeline/reporting.py
```

필요 시 수정 가능:

```text
pipeline/schema.py
pipeline/runner.py
```

가급적 수정하지 않을 파일:

```text
runtime/experiment_core.py
runtime/experiment_xai.py
pipeline/statistics.py
pipeline/xai.py
pipeline/xai_bundle.py
```

---

## 4. 구현 목표

보고서:

```text
run metadata
model definition
8-condition matrix
benchmark summary
paired tests with Holm correction
XAI primary summary
seed stability summary
case analysis
limitations
reproducibility commands
```

Dashboard:

```text
execution status
condition summary table
seed-level metric distribution
paired test table
XAI summary table
artifact links
```

---

## 5. 완료 기준

아래 명령이 성공해야 한다.

```bash
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

생성 파일:

```text
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

---

## 6. 보고서 문장 원칙

결과가 없는 상태:

```text
아직 full benchmark 결과가 없으므로 본 항목은 실행 후 자동 갱신된다.
```

결과가 있는 상태:

```text
D_B 조건은 A_B 대비 macro-F1에서 [mean_diff] 차이를 보였고, paired test의 Holm-adjusted p-value는 [p]였다.
```

XAI:

```text
XAI 분석은 성능 개선의 인과적 증명이 아니라, 모델 판단 패턴이 연구 가설과 일관적인지 확인하는 사후 근거로 해석한다.
```

---

## 7. 금지 사항

```text
없는 결과를 추정해서 쓰지 않는다.
best seed 수치를 대표 성능처럼 쓰지 않는다.
XAI case 하나로 전체 결론을 내리지 않는다.
top-level outputs/reports를 canonical input으로 쓰지 않는다.
```
