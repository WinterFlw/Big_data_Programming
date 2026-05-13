# v2 Workspace

> 이 폴더가 v2 작업의 새 기준점이다. 팀원은 여기만 열고 시작한다.

---

## 1. 왜 새 폴더인가

과거 루트에 섞여 있던 1차 파이프라인, 기존 산출물, 발표 문서는 `../v1/`로 이동했다.
이 폴더는 `v2_15seed` 실험을 위해 필요한 현재 기준 작업 공간이다.

```text
v2/
  README.md
  run.sh
  configs/
  docs/
  pipeline/
  outputs/
  scripts/
```

---

## 2. 먼저 볼 것

처음 온 사람은 아래 순서로 읽는다.

```text
README.md
docs/00_reading_order.md
docs/01_model_definition.md
docs/02_e2e_pipeline.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/12_code_commenting_guide.md
docs/13_commit_message_policy.md
docs/agent_tasks/README.md
```

역할별 에이전트를 쓸 사람은 `docs/agent_tasks/`에서 자기 역할 문서를 고른다.

---

## 3. 실행 방법

이 폴더 안에서 실행한다.

```bash
cd v2
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

또는 repo root에서 실행한다.

```bash
./v2/run.sh e2e status --run-id v2_15seed
```

---

## 4. 폴더별 책임

| 폴더 | 역할 |
|---|---|
| `configs/` | v2 실행 manifest/config |
| `docs/` | v2 설계, 통계, XAI, 서버 실행, 팀 업무 문서 |
| `docs/agent_tasks/` | 팀원이 에이전트에게 줄 역할별 지시서 |
| `pipeline/` | v2 전용 orchestration 코드 |
| `outputs/` | v2 실행 산출물 |
| `scripts/` | 커밋 훅/검증 등 보조 스크립트 |

---

## 5. Canonical output

이 새 workspace 안에서는 canonical output이 아래다.

```text
v2/outputs/experiments/v2_15seed/
```

문서와 코드 안의 `outputs/experiments/v2_15seed/`는 `v2/` 기준 상대 경로다.

---

## 6. 현재 코드 상태

현재 가능한 것:

```text
manifest 생성
execution_status.csv 생성
120개 condition x seed planned unit 생성
benchmark aggregate 골격 생성
XAI 산출물 골격 생성
report/dashboard 초안 생성
```

아직 실제 학습 실행은 연결 전이다.
다음 구현 1순위는 `pipeline/runner.py`의 `benchmark --execute` adapter다.

---

## 7. 팀원에게 줄 최소 지시

```text
v2/ 폴더만 기준으로 작업하세요.
공통 규칙은 docs/agent_tasks/00_common_agent_rules.md를 읽고,
자기 역할 문서는 docs/agent_tasks/에서 고르세요.
모든 산출물은 v2/outputs/experiments/v2_15seed/ 아래에 저장해야 합니다.
코드 주석은 docs/12_code_commenting_guide.md 기준을 따르세요.
커밋 메시지는 docs/13_commit_message_policy.md 기준을 따르세요.
```

---

## 8. 커밋 메시지 훅 설치

v2 커밋 메시지 규칙을 로컬 Git hook으로 적용한다.

```bash
./v2/scripts/install_commit_msg_hook.sh
```

설치 후 커밋 메시지는 아래 형식을 만족해야 한다.

```text
feat(benchmark): wire v2 smoke-run training adapter

Why:
- ...

What:
- ...

Validation:
- ...
```
