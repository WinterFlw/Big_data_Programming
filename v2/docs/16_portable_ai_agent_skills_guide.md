# 16. Portable AI Agent Skills Guide

> 목적: 팀원들이 Codex, Claude, Gemini, Cursor, Antigravity 등 서로 다른 AI 도구를 쓰더라도 같은 v2 기준으로 작업하도록 공용 Skill/지시서 사용법을 정리한다.

---

## 1. 왜 필요한가

이번 v2 작업은 단순 코드 수정이 아니다.

```text
학습 runtime 검증
benchmark adapter 검증
15 seed 통계 설계
XAI evidence bundle
report/dashboard 산출물
제한된 서버 실행 기회
```

이 요소들이 동시에 얽혀 있으므로 AI에게 “이 코드 봐줘”라고만 지시하면 아래 문제가 생긴다.

```text
v1 출력과 v2 출력을 섞는다.
full benchmark를 너무 빨리 실행하려 한다.
p-value만 보고 성급하게 결론을 쓴다.
XAI를 인과적 증명처럼 과장한다.
자기 담당 범위를 넘어 다른 사람 파일을 고친다.
```

따라서 모든 AI 도구에는 같은 공통 규칙과 역할별 Skill을 먼저 읽힌다.

---

## 2. 파일 구조

```text
v2/
  CLAUDE.md
  GEMINI.md
  ai_skills/
    README.md
    common_project_rules.md
    hatespeech-v2-e2e/
      SKILL.md
    hatespeech-v2-benchmark/
      SKILL.md
    hatespeech-v2-statistics/
      SKILL.md
    hatespeech-v2-xai/
      SKILL.md
    hatespeech-v2-report-dashboard/
      SKILL.md
    hatespeech-v2-review/
      SKILL.md
```

각 파일의 역할:

| 파일 | 용도 |
|---|---|
| `v2/CLAUDE.md` | Claude Code가 프로젝트 규칙을 자동으로 이해하도록 하는 입구 |
| `v2/GEMINI.md` | Gemini 계열 도구에 읽힐 프로젝트 규칙 |
| `ai_skills/README.md` | 팀원이 어떤 Skill을 골라야 하는지 보는 안내 |
| `ai_skills/common_project_rules.md` | 모든 AI 도구에 공통으로 적용되는 v2 원칙 |
| `ai_skills/*/SKILL.md` | 역할별 파일 범위, 검증 명령, 금지 사항 |

---

## 3. 도구별 사용법

### Claude Code

Claude에게 아래처럼 지시한다.

```text
Read v2/CLAUDE.md first.
Then read v2/ai_skills/common_project_rules.md.
Then read v2/ai_skills/hatespeech-v2-review/SKILL.md.
Review the v2 pipeline for server-run blockers.
Do not edit files unless necessary.
```

구현을 맡길 때는 `review` 대신 역할별 Skill을 지정한다.

```text
hatespeech-v2-benchmark
hatespeech-v2-statistics
hatespeech-v2-xai
hatespeech-v2-report-dashboard
```

### Gemini CLI / Gemini Code Assist

Gemini에게는 검토와 설명 작업을 맡기기 좋다.

```text
Read v2/GEMINI.md.
Read v2/ai_skills/common_project_rules.md.
Read v2/ai_skills/hatespeech-v2-statistics/SKILL.md.
Check whether the aggregate/statistics design is defensible.
List assumptions, risks, and required evidence.
Do not edit files unless explicitly asked.
```

### Cursor

Cursor에서는 프로젝트 전체가 아니라 `v2/` 중심으로 열고, Chat에 관련 Skill을 첨부한다.

```text
Read v2/ai_skills/common_project_rules.md and v2/ai_skills/hatespeech-v2-benchmark/SKILL.md.
Only inspect v2/pipeline/training_adapter.py, v2/pipeline/runner.py, and v2/runtime/experiment_core.py.
Check whether A_B seed 42 smoke can generate metrics, predictions, and checkpoint under v2 output root.
```

### Antigravity

Antigravity는 브라우저/터미널/에디터를 오가는 end-to-end 확인에 맞다.

```text
Read v2/ai_skills/common_project_rules.md and v2/ai_skills/hatespeech-v2-e2e/SKILL.md.
Run only cheap validation commands.
Do not launch full benchmark.
Verify report/dashboard artifacts are created under v2/outputs/experiments/v2_15seed/.
```

### Codex

Codex에서는 이 폴더를 그대로 읽어도 되고, 필요하면 역할별 폴더를 로컬 skill 디렉터리로 복사해도 된다.

```text
Use v2/ai_skills/hatespeech-v2-xai/SKILL.md.
Focus on xai-bundle contract and report/dashboard evidence inputs.
```

---

## 4. 역할별 선택 기준

| 상황 | 선택할 Skill |
|---|---|
| 전체 구조를 이해시키고 stage 순서를 관리해야 함 | `hatespeech-v2-e2e` |
| 학습이 실제로 돌아가는지, checkpoint/metrics/predictions가 남는지 봐야 함 | `hatespeech-v2-benchmark` |
| 15 seed mean/std, 핵심 paired t-test, CI, effect size를 검토해야 함 | `hatespeech-v2-statistics` |
| SHAP/LIME, seed stability, evidence bundle을 다뤄야 함 | `hatespeech-v2-xai` |
| 최종 보고서와 dashboard를 생성해야 함 | `hatespeech-v2-report-dashboard` |
| 서버 올리기 전 전체 리스크를 검토해야 함 | `hatespeech-v2-review` |

---

## 5. 팀장이 팀원에게 줄 기본 문장

```text
이번 작업은 HateSpeachStudy v2_15seed 파이프라인입니다.
각자 사용하는 AI 도구가 달라도 아래 파일을 먼저 읽히세요.

1. v2/ai_skills/common_project_rules.md
2. 자기 역할에 맞는 v2/ai_skills/hatespeech-v2-*/SKILL.md

v2/가 기준이고, v1/은 archive/reference입니다.
모든 새 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 둡니다.
full 120-run benchmark는 smoke gate 통과 전에는 실행하지 않습니다.
작업 완료 후 [v2 agent handoff] 형식으로 보고하세요.
```

---

## 6. 공통 검증 명령

AI가 코드를 수정했다면 먼저 아래 cheap validation을 실행한다.

```bash
python3 -m compileall v2/runtime v2/pipeline v2/scripts/validate_commit_message.py
python3 -m json.tool v2/configs/v2_15seed.json >/tmp/v2_config_check.json
PYTHON_BIN=python3 ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
PYTHON_BIN=python3 ./v2/run.sh e2e aggregate --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e xai-bundle --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e report --run-id v2_15seed
PYTHON_BIN=python3 ./v2/run.sh e2e dashboard --run-id v2_15seed
git diff --check
```

서버/GPU가 필요한 expensive validation은 아래 순서로만 진행한다.

```bash
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --execute --resume
PYTHON_BIN=/path/to/venv/bin/python ./v2/run.sh e2e aggregate --run-id v2_15seed
```

---

## 7. AI가 절대 하면 안 되는 것

```text
v1 코드를 v2 runtime dependency로 import하지 않는다.
v1/outputs 또는 root outputs를 canonical result로 쓰지 않는다.
smoke gate 없이 full 120 benchmark를 실행하지 않는다.
통계적으로 확인되지 않은 내용을 final report claim으로 쓰지 않는다.
XAI case study를 성능 개선의 인과 증명처럼 쓰지 않는다.
대용량 output/checkpoint를 커밋하지 않는다.
자기 담당 파일 밖을 이유 없이 수정하지 않는다.
```

---

## 8. 완료 보고 형식

모든 AI 도구는 작업 끝에 아래 형식을 사용한다.

```text
[v2 agent handoff]
Role:
Files changed:
Commands run:
Artifacts created/updated:
Validation passed:
Known limitations:
Next owner:
```

이 형식을 유지해야 팀장이 여러 AI 도구의 결과를 비교하고 통합할 수 있다.
