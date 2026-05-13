# 13. Commit Message Policy

> 목적: v2 작업의 커밋 메시지를 엄격하게 통일한다. 서버 실행, 통계 검증, XAI 결과가 섞이는 프로젝트이므로 커밋 로그가 실험 감사 로그 역할을 해야 한다.

---

## 1. 필수 형식

모든 일반 커밋은 아래 형식을 따른다.

```text
<type>(<scope>): <summary>

Why:
- ...

What:
- ...

Validation:
- ...
```

예시:

```text
feat(benchmark): wire v2 smoke-run training adapter

Why:
- Need one condition x seed run to validate server execution before full benchmark.

What:
- Connect benchmark --execute to the v2 RunUnit adapter.
- Redirect run artifacts into v2/outputs/experiments/v2_15seed.

Validation:
- python3 -m compileall v2/pipeline v2/scripts/validate_commit_message.py
- ./v2/run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute
```

---

## 2. 허용 type

```text
feat      새 기능
fix       버그 수정
docs      문서 수정
refactor  동작 변경 없는 구조 정리
test      테스트 추가/수정
chore     빌드/설정/정리
ci        CI 또는 자동화
perf      성능 개선
build     의존성/빌드 시스템
revert    되돌리기
```

---

## 3. 허용 scope

```text
v2
pipeline
benchmark
stats
xai
report
qa
docs
agent
config
scripts
outputs
root
```

---

## 4. 엄격 규칙

```text
header는 반드시 <type>(<scope>): <summary> 형식이다.
summary는 10자 이상 72자 이하로 쓴다.
summary 끝에 마침표를 찍지 않는다.
header 다음 줄은 반드시 빈 줄이다.
Why, What, Validation 섹션을 모두 포함한다.
Validation에는 실제 실행한 명령어를 적는다.
실행하지 못했다면 Validation에 "Not run:"과 이유를 적는다.
```

---

## 5. 금지 예시

```text
update
fix stuff
v2 changes
docs update.
feat: benchmark
feat(v2): done
```

---

## 6. 훅 설치

아래 명령으로 로컬 commit-msg 훅과 commit template을 설치한다.

```bash
./v2/scripts/install_commit_msg_hook.sh
```

설치 후 커밋 메시지는 아래 스크립트로 검증된다.

```bash
python3 v2/scripts/validate_commit_message.py <commit-msg-file>
```

---

## 7. 팀원에게 줄 문장

```text
커밋 메시지는 v2/docs/13_commit_message_policy.md 형식을 반드시 따르세요.
Why/What/Validation 섹션이 없으면 commit-msg hook에서 거절됩니다.
서버에서 실행하지 않은 검증은 실행한 척 쓰지 말고 "Not run:"으로 이유를 남기세요.
```
