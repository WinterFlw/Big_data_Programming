# Portable AI Skills for HateSpeachStudy v2

이 폴더는 Codex, Claude, Gemini, Cursor, Antigravity처럼 서로 다른 AI 도구가 같은 기준으로 v2 작업을 수행하도록 만든 공용 작업 지시서 묶음이다.

핵심 목적은 단순하다.

```text
AI가 v1과 v2를 섞지 않게 한다.
AI가 full benchmark를 함부로 실행하지 않게 한다.
AI가 자기 역할 밖 파일을 과하게 수정하지 않게 한다.
AI가 통계와 XAI를 과장해서 해석하지 않게 한다.
AI가 작업 후 같은 형식으로 인수인계하게 한다.
```

## 1. 가장 먼저 읽힐 파일

모든 AI 도구에는 아래 순서로 읽히면 된다.

```text
1. v2/CLAUDE.md 또는 v2/GEMINI.md
2. v2/ai_skills/common_project_rules.md
3. v2/ai_skills/<역할>/SKILL.md
4. 필요 시 v2/docs/14_team_assignment_matrix.md
5. 필요 시 v2/docs/15_runtime_code_validation_matrix.md
```

Claude가 아니어도 `CLAUDE.md`를 읽혀도 된다. Gemini가 아니어도 `GEMINI.md`를 읽혀도 된다. 둘 다 프로젝트 공통 규칙을 담고 있으므로, 도구 이름보다 “현재 쓰는 AI가 잘 읽는 형식”을 우선한다.

## 2. 도구별 사용법

| 도구 | 읽힐 파일 | 사용 방식 |
|---|---|---|
| Codex | `common_project_rules.md` + 역할별 `SKILL.md` | 필요하면 `hatespeech-v2-*` 폴더를 로컬 skills 폴더로 복사 |
| Claude Code | `v2/CLAUDE.md` + 역할별 `SKILL.md` | Claude 시작 후 “이 파일들을 먼저 읽어라”라고 지시 |
| Gemini CLI | `v2/GEMINI.md` + 역할별 `SKILL.md` | 검토/설계/문서화 작업에 특히 유용 |
| Cursor | `common_project_rules.md` + 역할별 `SKILL.md` | Chat에 관련 파일을 첨부하고 좁은 범위로 요청 |
| Antigravity | `common_project_rules.md` + 역할별 `SKILL.md` | 터미널/브라우저/에디터 통합 확인에 사용 |

## 3. 역할별 Skill 선택표

| 팀 역할 | 읽힐 Skill | 주요 책임 |
|---|---|---|
| 전체 통합/팀장 | `hatespeech-v2-e2e/SKILL.md` | stage 순서, run gate, 전체 연결 |
| 학습/benchmark | `hatespeech-v2-benchmark/SKILL.md` | runtime 학습, adapter, metrics/checkpoint/predictions |
| 통계/aggregate | `hatespeech-v2-statistics/SKILL.md` | paired test, Holm correction, CI, effect size |
| XAI/evidence | `hatespeech-v2-xai/SKILL.md` | xai-primary/deep/ablation, xai-bundle |
| 보고서/dashboard | `hatespeech-v2-report-dashboard/SKILL.md` | final_report.md/docx, dashboard/index.html |
| 리뷰/preflight | `hatespeech-v2-review/SKILL.md` | v2-only path audit, smoke gate, 서버 실행 전 점검 |

## 4. 팀원이 그대로 붙여 넣을 시작 프롬프트

```text
우리는 HateSpeachStudy v2_15seed 파이프라인을 작업 중입니다.
먼저 v2/ai_skills/common_project_rules.md를 읽고, 그 다음 내 역할에 맞는 v2/ai_skills/<role>/SKILL.md를 읽으세요.

중요 규칙:
- v2/가 canonical workspace입니다.
- v1/은 archive/reference일 뿐입니다.
- 모든 새 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 둡니다.
- full 120-run benchmark는 smoke gate 전에는 실행하지 않습니다.
- 역할 밖 파일을 수정해야 하면 이유와 범위를 먼저 설명하세요.

작업 후에는 [v2 agent handoff] 형식으로 완료 보고하세요.
```

## 5. 공통 금지 사항

```text
v1/outputs 또는 v1/checkpoints를 canonical input/output으로 쓰지 않는다.
root outputs/reports, outputs/xai, outputs/runs를 새 결과 위치로 쓰지 않는다.
full benchmark를 임의로 실행하지 않는다.
p-value만 보고 결론을 쓰지 않는다.
XAI case 몇 개를 성능 개선의 인과 증명처럼 말하지 않는다.
자기 담당 파일 밖을 이유 없이 수정하지 않는다.
```

## 6. 공통 완료 보고

모든 AI 도구는 마지막에 아래 형식으로 보고하게 한다.

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
