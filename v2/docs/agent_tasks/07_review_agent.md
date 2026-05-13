# 07. Review Agent Brief

> 역할: 다른 팀원의 코드를 리뷰하고, 버그/리스크/누락된 검증을 먼저 찾는다.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 Review 담당 에이전트입니다.
코드를 새로 작성하기보다, 변경사항이 서버 실행을 망가뜨릴 위험이 있는지 리뷰하세요.
findings를 먼저 제시하고, 파일/라인 근거와 재현 명령어를 포함하세요.
```

---

## 2. 반드시 읽을 문서

```text
docs/07_output_and_report_contract.md
docs/10_code_implementation_notes.md
docs/11_team_tasking_and_server_run_plan.md
docs/agent_tasks/00_common_agent_rules.md
```

---

## 3. 리뷰 우선순위

P0:

```text
서버 full run을 실패하게 만드는 문제
기존 결과를 덮어쓰는 문제
데이터 leakage
condition 간 hyperparameter 통제 붕괴
same-seed paired design 붕괴
```

P1:

```text
resume/skip이 작동하지 않는 문제
CSV schema 불일치
training_adapter가 metrics/history/config/predictions/checkpoint 계약을 깨는 문제
XAI sample set이 seed마다 달라지는 문제
checkpoint path가 불안정한 문제
xai-bundle이 report/dashboard 입력 계약을 깨는 문제
```

P2:

```text
보고서 표현 과장
로그 부족
사용성 낮은 CLI help
문서와 코드 이름 불일치
```

---

## 4. 리뷰 명령

```bash
python3 -m compileall pipeline scripts/validate_commit_message.py
python3 -m json.tool configs/v2_15seed.json >/tmp/v2_config_check.json
./run.sh e2e --help
./run.sh e2e status --run-id v2_15seed
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B,D_B --seeds 42 --dry-run
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
python3 - <<'PY'
from pipeline.statistics import compute_paired_tests, apply_holm_correction
manifest = {'statistics': {'paired_tests': ['A_B:D_B']}}
rows = []
for seed, a, d in [(42, .60, .65), (52, .61, .66), (62, .62, .67)]:
    rows.append({'condition':'A_B','seed':seed,'status':'completed','macro_f1':a,'accuracy':a,'weighted_f1':a})
    rows.append({'condition':'D_B','seed':seed,'status':'completed','macro_f1':d,'accuracy':d,'weighted_f1':d})
paired = compute_paired_tests(manifest, rows)
corrected = apply_holm_correction(paired)
assert paired[0]['n_pairs'] == 3
assert round(float(paired[0]['mean_diff']), 4) == 0.05
assert corrected[0]['p_value_holm'] != ''
print('paired_statistics_smoke: ok')
PY
```

---

## 5. 리뷰 출력 형식

```text
Findings
- [P0/P1/P2] file:line - 문제 설명

Open questions
- 확인 필요한 사항

Validation
- 실행한 명령어와 결과

Summary
- 전체 판단
```

문제가 없으면 아래처럼 쓴다.

```text
No blocking findings.
Remaining risk: full GPU execution has not been validated yet.
```
