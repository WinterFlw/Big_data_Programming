# v2 Workspace

> v2는 현재 연구의 단일 기준 작업 공간이다. 팀원은 v1을 열지 않고 `v2/` 안의 코드와 문서만 기준으로 작업한다.

---

## 1. 원칙

```text
v1/은 archive다.
v2 실행 중 v1 코드를 import하지 않는다.
학습/추론/XAI runtime은 v2/runtime/에 둔다.
end-to-end orchestration은 v2/pipeline/에 둔다.
모든 새 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 둔다.
```

과거 루트에 섞여 있던 1차 파이프라인, 기존 산출물, 발표 문서는 `../v1/`로 이동했다. v2 실행에 필요한 학습/XAI/대시보드 코드는 `runtime/` 아래에 복사되어 있으므로, v2는 독립 실행 기준으로 검증한다.

---

## 2. 폴더 구조

```text
v2/
  README.md
  run.sh
  requirements.txt
  configs/
  docs/
  ai_skills/
  pipeline/
  runtime/
  outputs/
  scripts/
```

| 위치 | 책임 |
|---|---|
| `configs/` | `v2_15seed` 실행 설정 |
| `docs/` | 모델 정의, E2E 설계, 통계, XAI, 서버 실행, 업무 분담 |
| `docs/agent_tasks/` | 팀원이 AI 에이전트에게 줄 역할별 지시서 |
| `ai_skills/` | Codex, Claude, Gemini, Cursor, Antigravity 공용 AI 작업 지시서 |
| `pipeline/` | stage orchestration, manifest/status, adapter, 통계, report/dashboard |
| `runtime/` | 실제 학습, 평가, 추론, XAI, 대시보드 실행 코드 |
| `outputs/` | v2 canonical 산출물 위치 |
| `scripts/` | 커밋 훅과 보조 검증 스크립트 |

---

## 3. 빠른 실행

repo root에서 실행:

```bash
./v2/run.sh e2e status --run-id v2_15seed
./v2/run.sh e2e plan --run-id v2_15seed --force
./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

`v2/` 안에서 실행:

```bash
cd v2
./run.sh e2e status --run-id v2_15seed
```

서버 virtualenv를 명시할 때:

```bash
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e status --run-id v2_15seed
```

---

## 4. 현재 코드 상태 (2026-05-17 기준 — 5 Stage 모두 100%)

| Stage | 완성도 | 핵심 |
|---|:---:|---|
| 1. Benchmark | 100% | training adapter + cudnn 결정성 + AMP autocast |
| 2. Statistics | 100% | 15 seed mean/std + 핵심 paired t-test + CI/effect size, Holm/ANOVA는 보조 분석 |
| 3. XAI Core | 100% | sample 결정성 + 4축 메트릭 (CI/MSS/IS/AttnEntropy) + attribution 캐싱 + sample-level metric |
| 4. XAI Bundle + Report | 100% | 15파일 자동 채움 + 통계 확증 claim + token jsonl + 진짜 subgroup 분해 |
| 5. QA + Server | 100% | failed/completed 분리 + daily.sh + gate_check.py (Full Run Gate 6조건 자동) |

작업 #1~#14 모두 origin/main에 push. 자세한 작업 이력은 [`docs/agent_tasks/20_claude_code_completion_brief.md`](docs/agent_tasks/20_claude_code_completion_brief.md) 참조.

현재 가능한 것:

```text
manifest 생성
execution_status.csv + failed_runs.csv + completed_runs.csv 자동 갱신
120개 condition × seed planned unit 생성
benchmark --execute adapter (AMP fp16 자동 활성, CUDA에서만)
metrics/history/config/predictions/checkpoint v2 output 정규화 경로
aggregate/statistics CSV 7개 (핵심 paired t-test와 CI/effect size 중심, anova_*.csv와 adjusted p-value는 보조 검증용)
XAI 4축 (Attribution / Faithfulness / Context Learning / Plausibility) 모두 자동 계산
xai-bundle 14파일 자동 채움 + token_attributions.jsonl + source × target 진짜 subgroup 분해
final_report.md/docx + dashboard/index.html 자동 표 채움
daily.sh + gate_check.py: Full Run Gate 6조건 자동 점검 → [Gate: GO/STOP]
```

아직 실제로 수행하지 않은 것 (= NVIDIA 서버에서 사람이 할 일):

```text
A_B seed 42 실제 학습 smoke (Pilot D1~D2)
8 conditions × 15 seeds full benchmark (D3 Gate 통과 후)
실제 결과 기반 final report/dashboard 검수 (Author D7~D10)
발표 자료 26p 통합 + 리허설
```

다음 gate는 아래 명령이다.

```bash
# 환경 설치 (statsmodels 포함)
pip install -r runtime/requirements.txt

# A_B seed 42 smoke
PYTHON_BIN=/path/to/venv/bin/python ./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume

# 매일 preflight + Gate
./scripts/daily.sh   # 마지막 줄 "[daily preflight ok — Gate: GO]"
```

---

## 5. End-to-End Flow

```text
benchmark
-> aggregate
-> xai-primary
-> xai-deep
-> xai-ablation
-> xai-bundle
-> report
-> dashboard
```

v2의 목표는 단순 성능표가 아니다. 15 seed 통계 검증과 full XAI evidence bundle을 함께 남겨, “얼마나 나아졌는가”와 “무엇을 더 남기는가”를 동시에 답한다.

핵심 산출물:

```text
outputs/experiments/v2_15seed/benchmark/benchmark_runs.csv
outputs/experiments/v2_15seed/benchmark/paired_tests_holm.csv
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_claims.json
outputs/experiments/v2_15seed/xai/evidence_bundle/xai_dashboard_bundle.json
outputs/experiments/v2_15seed/reports/final_report.md
outputs/experiments/v2_15seed/reports/final_report.docx
outputs/experiments/v2_15seed/dashboard/index.html
```

---

## 6. 먼저 읽을 문서

처음 합류한 사람:

```text
docs/00_reading_order.md
docs/01_model_definition.md
docs/02_e2e_pipeline.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/14_team_assignment_matrix.md
docs/15_runtime_code_validation_matrix.md
docs/17_korean_reading_file_index.md
docs/한글_필독문서_업무도입_가이드.docx
```

팀원에게 업무를 나눌 사람:

```text
docs/14_team_assignment_matrix.md
docs/15_runtime_code_validation_matrix.md
docs/agent_tasks/10_team_dispatch_prompts.md
docs/16_portable_ai_agent_skills_guide.md
docs/20_role_file_review_matrix.md
docs/v2_end_to_end_team_brief.docx
docs/role_guides/
```

AI 에이전트를 쓸 사람:

```text
CLAUDE.md 또는 GEMINI.md
docs/16_portable_ai_agent_skills_guide.md
ai_skills/README.md
ai_skills/common_project_rules.md
ai_skills/<자기 역할>/SKILL.md
docs/agent_tasks/README.md
docs/agent_tasks/00_common_agent_rules.md
docs/agent_tasks/<자기 역할 문서>.md
docs/agent_tasks/08_handoff_template.md
```

---

## 7. 5명 기준 역할

| 사람 | 역할 | 기준 문서 |
|---|---|---|
| 1번 | E2E Gate 총괄 | `v2/run.sh`, `v2/pipeline/cli.py`, `v2/pipeline/runner.py`, `v2/pipeline/manifest.py`, `v2/pipeline/artifacts.py`, `v2/scripts/daily.sh`, `v2/scripts/gate_check.py` |
| 2번 | 학습 실행 / 실험 관리 | `v2/pipeline/training_adapter.py`, `v2/runtime/experiment_core.py`, `v2/runtime/run_experiments.py`, `v2/runtime/utils.py` |
| 3번 | 결과 분석 / 통계 해석 | `v2/pipeline/statistics.py`, `v2/pipeline/schema.py`, `v2/outputs/experiments/v2_15seed/benchmark/*.csv` |
| 4번 | XAI 설명 / evidence bundle | `v2/pipeline/xai.py`, `v2/pipeline/xai_sampling.py`, `v2/pipeline/xai_bundle.py`, `v2/runtime/experiment_xai.py` |
| 5번 | 발표자료 / 최종 보고서 제작 | `v2/pipeline/reporting.py`, `v2/runtime/dashboard_app.py`, `v2/runtime/experiment_dashboard.py`, `v2/outputs/experiments/v2_15seed/reports/`, `v2/outputs/experiments/v2_15seed/dashboard/` |

작업 하달의 큰 틀은 연구 산출물 기준으로 나누고, 각 담당자가 실제로 볼 코드와 검증 명령은 역할별 Word 업무지시서와 `docs/15_runtime_code_validation_matrix.md`를 함께 따른다.
1번은 전체 코드리뷰 담당이 아니라 `docs/20_role_file_review_matrix.md` 기준으로 각 담당자의 1차 리뷰 결과를 취합하고 full run gate만 판단한다.

---

## 8. Full Run 금지 조건

아래가 통과하기 전에는 full 120 benchmark를 시작하지 않는다.

```text
v2_runtime_import_smoke 통과
A_B seed 42 단일 smoke 성공
A_B/D_B seed 42 paired smoke 성공
metrics.json/history.csv/run_config.json/predictions.csv/checkpoint 생성
aggregate가 smoke 결과를 읽고 paired row 생성
checkpoint와 predictions_path가 v2/outputs/experiments/v2_15seed/ 내부를 가리킴
report/dashboard stage가 실패하지 않음
```

---

## 9. 커밋 메시지 훅

```bash
./v2/scripts/install_commit_msg_hook.sh
```

커밋 메시지 형식:

```text
feat(benchmark): wire v2 smoke-run training adapter

Why:
- ...

What:
- ...

Validation:
- ...
```
