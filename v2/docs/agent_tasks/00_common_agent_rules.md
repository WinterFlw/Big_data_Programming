# 00. Common Agent Rules

> 모든 팀원이 에이전트를 사용할 때 공통으로 적용하는 규칙이다.

---

## 1. 공통 컨텍스트

에이전트에게 먼저 알려야 할 프로젝트 상태:

```text
Project: HateSpeachStudy
Goal: v2_15seed end-to-end pipeline
Run ID: v2_15seed
Canonical output root: outputs/experiments/v2_15seed/
Main CLI: ./run.sh e2e ...
Config: configs/v2_15seed.json
Pipeline package: pipeline/
Runtime code: runtime/
Design docs: docs/
```

---

## 2. 반드시 읽을 문서

모든 에이전트는 아래 문서를 먼저 읽는다.

```text
docs/README.md
docs/00_reading_order.md
docs/02_e2e_pipeline.md
docs/07_output_and_report_contract.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/15_runtime_code_validation_matrix.md
docs/12_code_commenting_guide.md
docs/13_commit_message_policy.md
```

역할별 문서는 그 다음에 읽는다.

---

## 3. 공통 실행 명령

작업 전 상태 확인:

```bash
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
```

문법 확인:

```bash
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
```

---

## 4. 산출물 위치 원칙

모든 새 산출물은 아래에 둔다.

```text
outputs/experiments/v2_15seed/
```

아래 경로는 archive/reference 용도다. 결과 비교를 위해 읽을 수는 있지만,
새 v2 실행의 입력/출력 기준으로 삼지 않는다.

```text
outputs/reports/
outputs/xai/
outputs/runs/
checkpoints/
../v1/outputs/
../v1/checkpoints/
```

v2 canonical 결과로 쓰면 안 되는 경로:

```text
outputs/reports
outputs/xai
outputs/runs
checkpoints
../v1/outputs
../v1/checkpoints
```

---

## 5. 파일 수정 원칙

에이전트는 자기 역할 문서에 적힌 파일을 우선 수정한다.

역할 밖 파일 수정이 필요하면 아래 순서로 처리한다.

```text
1. 왜 필요한지 설명한다.
2. 어떤 함수/스키마에 영향이 있는지 적는다.
3. 최소 수정으로 제한한다.
4. 완료 보고에 반드시 명시한다.
```

---

## 6. 서버 비용 관련 규칙

서버 실행 기회가 제한되어 있으므로 아래를 지킨다.

```text
full benchmark를 임의 실행하지 않는다.
full XAI를 임의 실행하지 않는다.
먼저 dry-run과 1-seed smoke run을 통과시킨다.
실패 시 전체 재실행 대신 failed unit만 재실행하도록 설계한다.
```

---

## 7. 에이전트 완료 보고 필수 형식

모든 에이전트 작업은 아래 형식으로 끝난다.

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

---

## 8. 주석과 커밋 메시지

코드 주석:

```text
docs/12_code_commenting_guide.md 기준을 따른다.
의도, 제약, 서버 비용, 다음 구현 지점을 설명한다.
코드 한 줄을 그대로 번역하는 주석은 피한다.
```

커밋 메시지:

```text
docs/13_commit_message_policy.md 기준을 따른다.
Why, What, Validation 섹션을 반드시 포함한다.
실행하지 않은 검증은 Not run: 이유 형식으로 남긴다.
```

---

## 9. 공통 리뷰 기준

작업 결과는 아래 기준으로 평가한다.

```text
CLI가 깨지지 않는가?
run_id output root가 지켜지는가?
resume 가능한 구조인가?
CSV/JSON schema가 07_output_and_report_contract.md와 맞는가?
서버 full run 없이 로컬 검증 가능한가?
다른 담당자의 파일을 불필요하게 건드리지 않았는가?
```
