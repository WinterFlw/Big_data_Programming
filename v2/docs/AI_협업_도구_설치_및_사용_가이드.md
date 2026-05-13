# AI 협업 도구 설치 및 사용 가이드

HateSpeachStudy v2_15seed 팀 운영용  
작성 기준일: 2026-05-13  
범위: Codex CLI, Claude Code, Gemini CLI, Cursor, Google Antigravity

## 1. 문서 목적

이 문서는 팀원들이 AI 도구를 각자 설치하고, v2 실험 폴더 안에서 안전하게 분업하기 위한 운영 가이드다. 핵심은 빠른 자동화가 아니라 재현 가능한 협업이다. AI가 코드를 많이 작성하더라도, 최종 책임은 담당자가 가진다.

## 2. 팀 공통 원칙

- 작업 루트는 기본적으로 `v2/`다. 기존 루트 파일을 바꾸는 작업은 리드에게 먼저 공유한다.
- 전체 15시드 벤치마크는 서버 기회가 한정되어 있으므로 승인 없이 실행하지 않는다.
- AI가 수정한 파일은 반드시 사람이 `git diff`로 검토한다.
- 데이터셋, API 키, 계정 토큰, 서버 접속 정보는 프롬프트나 커밋에 넣지 않는다.
- AI에게 “알아서 다 해줘”라고 맡기지 말고, 역할 문서와 산출물 계약을 같이 제공한다.
- 생성된 실험 결과는 `v2/outputs/` 아래에 두되, 대용량 결과물은 커밋하지 않는다.
- 커밋 전에는 커밋 메시지 정책과 검증 명령을 통과시킨다.

## 3. 도구 선택 기준

| 도구 | 주 용도 | 잘 맞는 작업 | 주의사항 |
|---|---|---|---|
| Codex CLI | 터미널 기반 코딩 에이전트 | v2 파이프라인 수정, 검증 명령 실행, 커밋 전 점검 | 실행 권한과 파일 변경 범위를 프롬프트에 명시한다. |
| Claude Code | 터미널 기반 코딩/리뷰 에이전트 | 코드 리뷰, 리팩터링, 버그 설명, 대안 구현 비교 | npm 설치 시 `sudo`를 쓰지 않는다. |
| Gemini CLI | Google 계열 CLI 에이전트 | 문서 초안, 통계/실험 설계 검토, 비교 설명 | 최신 설치법은 공식 GitHub 문서를 확인한다. |
| Cursor | AI 내장 IDE | 함수 단위 구현, diff 리뷰, 문서/코드 동시 편집 | 프로젝트 전체가 아니라 `v2/` 중심으로 열어 컨텍스트를 좁힌다. |
| Google Antigravity | 에이전트형 개발 환경 | 브라우저/터미널/에디터를 오가는 통합 작업, 프로토타입 검증 | 공개 프리뷰 도구이므로 결과는 반드시 로컬 검증한다. |

## 4. 공통 설치 전 준비

1. Git, Python 3.10 이상, Node.js LTS를 설치한다.
2. 저장소를 최신화한다.

```bash
cd /path/to/HateSpeachStudy
git pull --ff-only
```

3. v2 폴더의 기본 상태를 확인한다.

```bash
cd /path/to/HateSpeachStudy
./v2/run.sh e2e status --run-id v2_15seed
```

4. 커밋 메시지 검증 훅을 설치한다.

```bash
cd /path/to/HateSpeachStudy
./v2/scripts/install_commit_msg_hook.sh
```

5. AI 도구에 줄 기본 컨텍스트는 다음 세 파일이다.

- `v2/README.md`
- `v2/docs/00_reading_order.md`
- `v2/docs/agent_tasks/00_common_agent_rules.md`

## 5. Codex CLI 설치 및 사용

Codex CLI는 OpenAI의 터미널 기반 코딩 에이전트다. 저장소 수정, 명령 실행, 검증 루프에 가장 잘 맞는다.

### 설치

공식 문서 기준 기본 설치 명령은 다음과 같다.

```bash
npm install -g @openai/codex
```

설치 후 로그인한다.

```bash
codex login
codex --version
```

업데이트가 필요하면 다음 명령을 사용한다.

```bash
npm i -g @openai/codex@latest
```

### 기본 사용 흐름

```bash
cd /path/to/HateSpeachStudy
codex
```

자동화형 단일 요청은 다음처럼 실행할 수 있다.

```bash
codex exec "Read v2/docs/agent_tasks/01_benchmark_agent.md and inspect only v2/pipeline. Do not run the full benchmark."
```

### 권장 프롬프트

```text
Read v2/docs/agent_tasks/00_common_agent_rules.md first.
Your work scope is v2/ only.
Do not run the full 15-seed benchmark.
Before editing, summarize the files you plan to touch.
After editing, run targeted validation and show git diff summary.
```

## 6. Claude Code 설치 및 사용

Claude Code는 Anthropic의 터미널 기반 개발 도구다. 코드 설명, 리뷰, 리팩터링 후보 제안, 구현 대안 비교에 적합하다.

### 설치

공식 문서는 네이티브 설치 방식을 권장한다. npm을 사용할 경우 다음 명령을 사용한다.

```bash
npm install -g @anthropic-ai/claude-code
```

공식 문서는 npm 전역 설치에 `sudo`를 쓰지 말라고 안내한다. 설치 후 다음처럼 실행한다.

```bash
claude
```

### 권장 사용법

- 복잡한 변경 전에는 먼저 “어떤 파일을 바꿀지”만 묻게 한다.
- 리뷰 작업에서는 “버그, 재현성, 통계적 타당성, 누락 테스트” 순서로 보게 한다.
- 구현을 맡길 때는 `v2/docs/agent_tasks/`의 담당자별 문서를 같이 읽게 한다.

### 권장 프롬프트

```text
You are reviewing the v2 experiment pipeline.
Read v2/docs/00_reading_order.md and v2/docs/agent_tasks/07_review_agent.md.
Focus on bugs, reproducibility risks, statistical mistakes, and missing validation.
Do not rewrite unrelated files.
```

## 7. Gemini CLI 설치 및 사용

Gemini CLI는 Google Gemini 기반 명령줄 도구다. 문서 정리, 실험 설계 검토, 통계 해석 보조, 교차 검토에 사용할 수 있다.

### 설치

최신 설치 명령은 Google Gemini CLI 공식 GitHub 문서에서 확인한다. Node.js 기반 설치 또는 `npx` 실행 방식이 제공될 수 있으므로, 로컬 환경에 맞는 방식을 선택한다.

### 권장 사용법

- 통계 문서와 XAI 문서를 읽히고 “논리적 빈칸”을 찾게 한다.
- 실험 결과 해석 초안을 만들 때 사용한다.
- 커밋을 만드는 주 도구로 쓰기보다는, 검토자 또는 설명 보조 역할로 쓰는 편이 안전하다.

### 권장 프롬프트

```text
Read v2/docs/03_validation_and_statistics.md and v2/docs/04_xai_protocol.md.
Check whether the 15-seed plan is statistically defensible.
List assumptions, risks, and what evidence must be collected.
Do not edit files unless explicitly asked.
```

## 8. Cursor 설치 및 사용

Cursor는 AI 기능이 내장된 IDE다. 팀원이 코드를 직접 읽고 고치면서 AI 도움을 받기에 가장 접근성이 좋다.

### 설치

공식 다운로드 페이지에서 운영체제에 맞는 설치 파일을 받는다. macOS에서는 공식 페이지가 제공하는 원라인 설치 명령을 사용할 수 있다.

```bash
curl https://cursor.com/install -fsS | bash
```

단, curl 명령은 반드시 공식 Cursor 다운로드 페이지에서 확인한 뒤 실행한다.

### 프로젝트 열기

```bash
cd /path/to/HateSpeachStudy
cursor v2
```

Cursor CLI가 없다면 앱을 열고 `v2/` 폴더를 직접 연다.

### 권장 설정

- 채팅에는 현재 파일과 관련 문서만 첨부한다.
- 대형 자동 수정 대신 작은 변경 단위로 요청한다.
- 적용 전에는 제안 diff를 읽고, 적용 후에는 터미널에서 검증 명령을 실행한다.
- “Fix all” 같은 광범위한 자동 수정은 사용하지 않는다.

### 권장 프롬프트

```text
Use the existing v2 style.
Only modify files required for this task.
Keep comments detailed enough for teammates to understand the pipeline.
After changes, tell me exact commands to validate the change.
```

## 9. Google Antigravity 설치 및 사용

Google Antigravity는 Google이 공개한 에이전트 중심 개발 플랫폼이다. 에디터, 터미널, 브라우저를 함께 사용해 풀스택 작업을 처리하는 흐름에 맞춰져 있다.

### 설치

공식 다운로드 페이지에서 macOS, Windows, Linux용 설치 파일을 받는다.

```text
https://antigravity.google/download
```

### 권장 사용법

- 새 기능의 엔드투엔드 흐름을 한 번에 점검할 때 사용한다.
- 브라우저 확인, 터미널 명령, 코드 수정을 연결해야 하는 작업에 적합하다.
- 공개 프리뷰 성격이 있으므로 모든 변경은 로컬 검증과 `git diff` 리뷰를 거친다.

### 권장 프롬프트

```text
Open the v2 folder only.
Read v2/docs/agent_tasks/06_integration_lead_agent.md.
Map the end-to-end flow from config to manifest to report.
Do not run expensive benchmark commands.
Ask before deleting files or changing generated outputs.
```

## 10. 역할별 권장 조합

| 역할 | 1순위 도구 | 보조 도구 | 산출물 |
|---|---|---|---|
| Benchmark 담당 | Codex CLI | Cursor | 어댑터 구현, 실행 로그, manifest 초안 |
| Statistics 담당 | Cursor | Gemini CLI, Claude Code | 15시드 통계 검정, 효과크기, 신뢰구간 문서 |
| XAI 담당 | Codex CLI | Claude Code, Antigravity | SHAP/IG/LIME/attention rollout 결과와 케이스 분석 |
| Report 담당 | Cursor | Gemini CLI | 최종 표, 리포트 템플릿, 해석 초안 |
| QA/Server 담당 | Codex CLI | Antigravity | 서버 실행 체크리스트, dry-run 결과, 실패 대응표 |
| Integration Lead | Codex CLI | Claude Code | 병합 검토, 최종 실행 승인, 커밋/푸시 관리 |

## 11. 팀원에게 줄 공통 작업 지시문

```text
You are working on HateSpeachStudy v2.
First read:
1. v2/README.md
2. v2/docs/00_reading_order.md
3. v2/docs/agent_tasks/00_common_agent_rules.md

Scope:
- Work inside v2/ unless explicitly approved.
- Do not run the full 15-seed benchmark.
- Do not commit secrets, datasets, or large generated outputs.

Before edits:
- Explain which files you will touch.
- Explain what validation will prove the change works.

After edits:
- Run targeted validation.
- Provide git diff summary.
- Write a strict conventional commit message candidate.
```

## 12. 담당자별 AI 프롬프트

### Benchmark 담당

```text
Read v2/docs/agent_tasks/01_benchmark_agent.md.
Implement or review the benchmark adapter contract only.
Do not change statistics or XAI modules unless the contract is broken.
Prepare a small dry-run command and expected artifact list.
```

### Statistics 담당

```text
Read v2/docs/agent_tasks/02_statistics_agent.md and v2/docs/03_validation_and_statistics.md.
Check whether the metric aggregation supports paired tests across 15 seeds.
Ensure mean, standard deviation, confidence interval, p-value, and effect size can be reported.
```

### XAI 담당

```text
Read v2/docs/agent_tasks/03_xai_agent.md and v2/docs/04_xai_protocol.md.
Design the minimum XAI artifact set for SHAP, Integrated Gradients, LIME, attention rollout, and case review.
Make every output traceable to seed, model, dataset split, sample id, and method.
```

### Report 담당

```text
Read v2/docs/agent_tasks/04_report_dashboard_agent.md.
Check that the final report can explain model performance, uncertainty, and XAI evidence without overclaiming.
Prefer tables and concise interpretation over long prose.
```

### QA/Server 담당

```text
Read v2/docs/agent_tasks/05_qa_server_agent.md and v2/docs/06_execution_runbook.md.
Prepare server preflight, dry-run, full-run, monitoring, and recovery steps.
Do not launch the full 15-seed run without explicit approval.
```

## 13. 커밋 전 필수 검증

최소 검증 명령은 다음과 같다.

```bash
cd /path/to/HateSpeachStudy
python3 -m compileall v2/pipeline v2/scripts/validate_commit_message.py
./v2/run.sh e2e status --run-id v2_15seed
git diff --stat
```

변경 범위에 따라 다음을 추가한다.

- 파이프라인 수정: 작은 dry-run 또는 단위 스모크 테스트
- 통계 수정: toy 데이터 기반 paired test 검증
- XAI 수정: 1개 seed, 1개 샘플 기반 artifact 생성 검증
- 문서 수정: 관련 역할 문서와 실행 문서의 용어 일치 확인

## 14. AI 사용 시 금지 사항

- 서버 전체 실행을 승인 없이 시작하지 않는다.
- `rm -rf`, `git reset --hard`, 대량 이동, 대량 포맷팅을 AI에게 맡기지 않는다.
- 웹에서 본 임의 설치 스크립트를 실행하지 않는다.
- API 키를 채팅창, 로그, 문서, 커밋에 넣지 않는다.
- AI가 만든 통계 해석을 검정 없이 결론으로 쓰지 않는다.
- XAI 시각화를 “원인”이라고 단정하지 않는다. 설명 근거 또는 정성 분석 자료로 표현한다.

## 15. 문제 해결

### command not found

설치 경로가 shell PATH에 없을 가능성이 높다.

```bash
echo $PATH
npm prefix -g
```

터미널을 재시작한 뒤 다시 확인한다.

### npm 권한 오류

`sudo npm install -g ...`로 해결하지 않는다. Node 버전 관리자 또는 공식 설치 방식을 다시 확인한다.

### macOS에서 앱 실행 차단

시스템 설정의 개인정보 보호 및 보안에서 개발자 앱 실행 허용을 확인한다. 반드시 공식 다운로드 파일인지 먼저 확인한다.

### AI가 너무 많은 파일을 바꿈

즉시 중지하고 다음을 실행한다.

```bash
git diff --stat
git diff -- v2/
```

필요한 변경만 선별해서 남긴다. 사용자 작업이 섞여 있으면 임의로 되돌리지 않는다.

### 커밋 메시지가 거부됨

정책 문서를 확인한다.

```bash
sed -n '1,220p' v2/docs/13_commit_message_policy.md
```

예시:

```text
feat(v2): add xai artifact collector

- collect per-seed explanation metadata
- write method-level manifest entries
```

## 16. 공식 출처

설치 명령과 지원 운영체제는 바뀔 수 있으므로, 실행 전 공식 문서를 다시 확인한다.

- OpenAI Codex CLI documentation: https://developers.openai.com/codex/cli/
- Anthropic Claude Code setup: https://code.claude.com/docs/en/setup
- Cursor official download: https://cursor.com/download
- Google Antigravity download: https://antigravity.google/download
- Google Antigravity announcement: https://blog.google/technology/developers/gemini-3-developers/
- Google Gemini CLI getting started: https://github.com/google-gemini/gemini-cli/blob/main/docs/get-started/index.md

