# 03. XAI Agent Brief

> 역할: XAI primary/deep/ablation 실행과 seed stability 분석을 구현한다.
>
> **현재 상태 (2026-05-17)**: 작업 #4 (adapter + xai_sampling.py) + #8 (ablation 4축) + #9 (token jsonl) + #11 (primary 4축) + #14 (sample-level) 적용 완료. XAI 코드 100%. agent는 토큰 sanity check + 학술 인용 매핑 + p4/p5 본문 작성 위주.
>
> **4번은 서버 안 들어감** — 2번이 push한 outputs/.../xai/.cache/*.json + xai/primary/*.csv를 git pull로 받아서 로컬에서 토큰 sanity check. SHAP/LIME 본 실행은 2번이 서버에서.

---

## 1. 에이전트에게 줄 첫 지시문

```text
당신은 HateSpeachStudy v2_15seed 파이프라인의 XAI 담당 에이전트입니다.
목표는 A_B와 D_B checkpoint를 중심으로 SHAP/LIME, rationale alignment, masking, seed stability 산출물을 생성하는 것입니다.
XAI는 모델 설계 근거가 아니라 사후 검증 도구입니다.
같은 sample set을 seed별 checkpoint에 동일하게 적용해야 합니다.
```

---

## 2. 반드시 읽을 문서

```text
docs/04_xai_protocol.md
docs/08_xai_report_template.md
docs/07_output_and_report_contract.md
docs/11_team_tasking_and_server_run_plan.md
docs/agent_tasks/00_common_agent_rules.md
docs/agent_tasks/09_e2e_xai_evidence_bundle_agent.md
```

---

## 3. 소유 파일

우선 수정 가능:

```text
pipeline/xai.py
```

필요 시 수정 가능:

```text
pipeline/schema.py
pipeline/runner.py
```

읽기 전용 참고:

```text
runtime/experiment_xai.py
```

가급적 수정하지 않을 파일:

```text
pipeline/statistics.py
pipeline/xai_bundle.py
pipeline/reporting.py
runtime/experiment_core.py
```

---

## 4. XAI stage 책임

Primary XAI:

```text
Models: A_B, D_B
Seeds: all 15 seeds
Samples: 200 fixed stratified samples
Outputs: seed-level metrics, paired xai tests, seed stability
```

Deep XAI:

```text
Models: A_B, D_B
Seed: median-performing seed
Samples: 500 stratified samples
Outputs: xai_details.json, case_summary.csv, case plots
```

Ablation XAI:

```text
Models: all 8 conditions
Seed: median-performing seed per condition
Samples: 50
Outputs: xai_ablation_metrics.csv
```

---

## 5. 완료 기준

아래 명령이 성공해야 한다.

```bash
./run.sh e2e xai-primary --run-id v2_15seed --resume
./run.sh e2e xai-deep --run-id v2_15seed
./run.sh e2e xai-ablation --run-id v2_15seed
```

생성 파일:

```text
outputs/experiments/v2_15seed/xai/samples/primary_samples.csv
outputs/experiments/v2_15seed/xai/primary/seed_level_metrics.csv
outputs/experiments/v2_15seed/xai/primary/paired_xai_tests.csv
outputs/experiments/v2_15seed/xai/primary/seed_stability.csv
outputs/experiments/v2_15seed/xai/deep/case_summary.csv
outputs/experiments/v2_15seed/xai/ablation/xai_ablation_metrics.csv
outputs/experiments/v2_15seed/xai/xai_summary.json
```

---

## 6. 필수 지표

```text
SHAP-LIME Overlap@5
SHAP-LIME Overlap@10
Rationale Precision@5
Rationale Recall@5
Rationale F1@5
Comprehensiveness
Sufficiency
Leave-one-out Drop
Top-k Jaccard across seeds
Rank correlation across seeds
```

---

## 7. 주의할 토큰 처리

SHAP/LIME token aggregation에서 아래를 구분한다.

```text
BERT subword: ## prefix
RoBERTa subword: Ġ prefix
SentencePiece: ▁ prefix
word-level token: prefix 없음
```

word-level token을 RoBERTa subword로 오판하면 안 된다.

---

## 8. 금지 표현

```text
XAI가 VADER 추가의 효과를 증명했다.
XAI 결과만으로 모델이 맥락을 완전히 이해한다고 볼 수 있다.
```

권장 표현:

```text
XAI 결과는 v2 조건의 판단 패턴이 human rationale과 더 정렬되는 경향을 보였다는 사후 근거로 해석한다.
```
