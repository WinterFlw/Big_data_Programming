# 10. Team Dispatch Prompts

> 목적: 팀장이 팀원 또는 개인 AI 에이전트에게 그대로 붙여 넣을 수 있는 업무 하달 문장을 제공한다.

---

## 1. 공통 시작 문장

모든 팀원에게 먼저 아래 문장을 보낸다.

```text
우리는 HateSpeachStudy의 v2_15seed end-to-end 파이프라인을 구현 중입니다.
이번 목표는 정확도 표만 만드는 것이 아니라 benchmark -> aggregate -> xai-primary -> xai-deep -> xai-ablation -> xai-bundle -> report -> dashboard까지 완성하는 것입니다.
실행 기준 코드는 v2/runtime과 v2/pipeline입니다. v1은 archive/reference로만 봅니다.
모든 새 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 저장해야 합니다.
기존 v1/outputs, v1/checkpoints, top-level outputs를 canonical output으로 쓰지 마세요.
업무 범위와 코드 검증 책임은 v2/docs/14_team_assignment_matrix.md와 v2/docs/15_runtime_code_validation_matrix.md를 따르세요.
자기 담당 파일 밖을 수정해야 하면 먼저 이유와 변경 범위를 설명해주세요.
```

---

## 2. Benchmark 담당에게 보낼 문장

```text
당신은 v2 Benchmark Adapter 담당입니다.

목표:
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
명령이 실제 학습 1개를 실행하고, run_id 내부에 metrics/history/config/log/checkpoint를 남기게 해주세요.

기간:
D1-D2 구현, D3 서버 smoke 준비.

주 수정 파일:
v2/pipeline/runner.py
v2/pipeline/artifacts.py
v2/pipeline/schema.py

필요하면 새 파일:
v2/pipeline/training_adapter.py

참고만 할 파일:
v2/runtime/experiment_core.py
v2/runtime/run_experiments.py

수정하지 말 파일:
v2/pipeline/statistics.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py

완료 기준:
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
./run.sh e2e status --run-id v2_15seed

주의:
condition별 hyperparameter를 임의로 바꾸지 마세요.
같은 seed에서 condition마다 split/sample order가 달라지면 안 됩니다.
checkpoint는 반드시 v2/outputs/experiments/v2_15seed/ 아래에도 남아야 합니다.
```

---

## 3. Statistics 담당에게 보낼 문장

```text
당신은 v2 Statistics 담당입니다.

목표:
120개 benchmark run 결과를 읽어서 condition별 summary와 핵심 A_B vs D_B same-seed paired test, 95% CI, effect size를 생성해주세요.
Holm 보정과 ANOVA는 여러 조건을 동시에 보여줄 때의 보조/부록 분석으로만 다룹니다.

기간:
D1-D3 구현 및 smoke aggregate, D6 full 결과 통계 확정.

주 수정 파일:
v2/pipeline/statistics.py
v2/pipeline/schema.py

반드시 읽을 문서:
v2/docs/03_validation_and_statistics.md
v2/docs/07_output_and_report_contract.md

수정하지 말 파일:
v2/pipeline/runner.py
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/reporting.py

완료 기준:
./run.sh e2e aggregate --run-id v2_15seed

필수 산출물:
benchmark_runs.csv
benchmark_summary.csv
paired_tests.csv
paired_tests_holm.csv

주의:
p-value만 보고하지 말고 mean_diff, 95% CI, effect_size를 같이 남겨주세요.
같은 seed끼리 비교하는 paired design을 깨면 안 됩니다.
```

---

## 4. XAI Core 담당에게 보낼 문장

```text
당신은 v2 XAI Core 담당입니다.

목표:
xai-primary, xai-deep, xai-ablation 산출물을 생성해주세요.
XAI는 모델 설계 근거가 아니라 사후 검증이며, 같은 sample set을 seed 간 고정해야 합니다.

기간:
D2-D4 코드 준비, D7-D8 benchmark 완료 후 실제 XAI 실행.

주 수정 파일:
v2/pipeline/xai.py

필요하면 새 파일:
v2/pipeline/xai_sampling.py
v2/pipeline/xai_methods.py
v2/pipeline/xai_metrics.py

참고만 할 파일:
v2/runtime/experiment_xai.py

수정하지 말 파일:
v2/pipeline/xai_bundle.py
v2/pipeline/statistics.py
v2/pipeline/reporting.py

완료 기준:
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed

필수 산출물:
xai/samples/primary_samples.csv
xai/primary/seed_level_metrics.csv
xai/primary/paired_xai_tests.csv
xai/primary/seed_stability.csv
xai/deep/case_summary.csv
xai/ablation/xai_ablation_metrics.csv
xai/xai_summary.json
```

---

## 5. XAI Evidence Bundle 담당에게 보낼 문장

```text
당신은 v2 XAI Evidence Bundle 담당입니다.

목표:
xai-primary, xai-deep, xai-ablation 결과를 읽어서 report/dashboard가 우선 소비할 evidence bundle을 생성해주세요.
이 stage는 SHAP/LIME을 다시 계산하지 않습니다.

기간:
D2-D3 placeholder bundle 검증, D8 실제 XAI 결과 기반 bundle 완성.

주 수정 파일:
v2/pipeline/xai_bundle.py

필요 시 협의 후 수정:
v2/pipeline/runner.py
v2/pipeline/cli.py
v2/docs/07_output_and_report_contract.md
v2/docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md

수정하지 말 파일:
v2/pipeline/xai.py
v2/pipeline/statistics.py
v2/pipeline/reporting.py

완료 기준:
./run.sh e2e xai-bundle --run-id v2_15seed
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_run_metadata.json >/tmp/xai_meta_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json >/tmp/xai_claims_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_interpretation_cards.json >/tmp/xai_cards_check.json
python3 -m json.tool outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json >/tmp/xai_dashboard_check.json

필수 산출물:
xai/evidence_bundle/evidence_inventory.csv
xai/evidence_bundle/xai_run_metadata.json
xai/evidence_bundle/xai_sample_manifest.csv
xai/evidence_bundle/xai_predictions.csv
xai/evidence_bundle/method_agreement.csv
xai/evidence_bundle/faithfulness_metrics.csv
xai/evidence_bundle/context_metrics.csv
xai/evidence_bundle/plausibility_metrics.csv
xai/evidence_bundle/subgroup_xai_metrics.csv
xai/evidence_bundle/xai_risk_flags.csv
xai/evidence_bundle/xai_claims.json
xai/evidence_bundle/xai_interpretation_cards.json
xai/evidence_bundle/xai_dashboard_bundle.json
xai/evidence_bundle/token_attributions.jsonl
xai/evidence_bundle/README.md

주의:
claim은 반드시 source artifact와 연결되어야 합니다.
통계적으로 확인되지 않은 내용을 확정적 claim으로 쓰지 마세요.
```

---

## 6. Report/Dashboard 담당에게 보낼 문장

```text
당신은 v2 Report/Dashboard 담당입니다.

목표:
benchmark/statistics/XAI evidence bundle을 읽어서 final_report.md, final_report.docx, dashboard/index.html을 생성해주세요.
report/dashboard는 raw XAI case보다 xai_claims.json과 xai_dashboard_bundle.json을 우선 읽어야 합니다.

기간:
D2-D4 placeholder 출력, D8-D9 최종 출력.

주 수정 파일:
v2/pipeline/reporting.py

필요하면 새 파일:
v2/pipeline/report_sections.py
v2/pipeline/dashboard_renderer.py

반드시 읽을 문서:
v2/docs/07_output_and_report_contract.md
v2/docs/08_xai_report_template.md

수정하지 말 파일:
v2/pipeline/xai.py
v2/pipeline/xai_bundle.py
v2/pipeline/statistics.py

완료 기준:
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

필수 산출물:
reports/final_report.md
reports/final_report.docx
dashboard/index.html
```

---

## 7. QA/Server 담당에게 보낼 문장

```text
당신은 v2 QA/Integration/Server 담당입니다.

목표:
서버 실행 전에 compile, config, manifest, dry-run, smoke 기준을 검증하고, 서버에서는 status와 실패 run을 관리해주세요.

기간:
D0-D10 상시 담당.

주 수정 파일:
v2/pipeline/cli.py
v2/pipeline/artifacts.py
v2/run.sh
v2/scripts/validate_commit_message.py

필요 시 협의 후 수정:
v2/pipeline/runner.py
v2/pipeline/schema.py
v2/configs/v2_15seed.json

완료 기준:
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
./run.sh e2e --help
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed

서버 smoke:
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e status --run-id v2_15seed
```

---

## 8. 완료 보고 공통 양식

```text
[v2 작업 완료]
담당:
기간:
수정한 파일:
실행한 명령어:
생성/변경된 산출물:
통과한 검증:
남은 위험:
다음 사람이 이어받을 부분:
```
