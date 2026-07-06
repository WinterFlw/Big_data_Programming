# Context-Aware Hate Speech Detection

HateXplain 데이터셋을 기반으로 혐오표현 탐지 모델의 **성능**, **재현성**, **설명 가능성(XAI)**을 함께 검증한 빅데이터프로그래밍 산학협력 프로젝트입니다.

핵심 아이디어는 단순히 욕설이나 특정 키워드에 반응하는 모델이 아니라, 사람이 혐오 판단의 근거로 표시한 토큰 단위 rationale을 학습 과정에 반영해 모델의 attention을 더 설명 가능한 방향으로 유도하는 것입니다.

## Project Summary

| 항목 | 내용 |
|---|---|
| Status | Completed project |
| Task | 3-class hate speech classification |
| Dataset | HateXplain |
| Labels | `hatespeech`, `offensive`, `normal` |
| Main models | BERT, RoBERTa |
| Proposed component | Rationale-aware attention loss |
| Auxiliary feature | VADER sentiment score |
| Experiment design | Backbone x Attention Loss x VADER, 8 conditions |
| Repetition | 15 training seeds, 120 total runs |
| Primary metric | Macro F1 |
| XAI methods | SHAP, LIME, rationale alignment, masking/faithfulness checks |

## Final Result

The best-performing condition in this experiment was **B_B: BERT + rationale-aware attention loss**.

| Condition | Backbone | Attention Loss | VADER | Macro F1 mean | Weighted F1 mean | Interpretation |
|---|---|---:|---:|---:|---:|---|
| A_B | BERT | No | No | 0.6798 | 0.6876 | BERT baseline |
| B_B | BERT | Yes | No | **0.6858** | **0.6935** | Best condition |
| C_B | BERT | No | Yes | 0.6825 | 0.6903 | Sentiment feature only |
| D_B | BERT | Yes | Yes | 0.6836 | 0.6910 | Attention + sentiment |
| A_R | RoBERTa | No | No | 0.6653 | 0.6723 | RoBERTa baseline |
| B_R | RoBERTa | Yes | No | 0.6763 | 0.6836 | Best RoBERTa condition |
| C_R | RoBERTa | No | Yes | 0.6698 | 0.6777 | Sentiment feature only |
| D_R | RoBERTa | Yes | Yes | 0.6743 | 0.6813 | Attention + sentiment |

### Statistical Findings

| Question | Result | Interpretation |
|---|---|---|
| Did B_B improve over A_B? | mean diff +0.0060, Holm-adjusted p = 0.0027 | Yes, small but statistically significant |
| Did attention loss matter? | 3-way ANOVA p ~= 6.2e-06, eta squared ~= 0.0955 | Meaningful factor |
| Did VADER matter? | 3-way ANOVA p ~= 0.5248, eta squared ~= 0.0017 | Weak independent effect |
| Which factor explained most variance? | Backbone eta squared ~= 0.3896 | Model backbone was the strongest factor |

This is **not claimed as an external state-of-the-art result**. The result means that, within this controlled 8-condition experiment, B_B was the strongest candidate.

## What Changed Compared With A Baseline

The baseline model uses only the text encoder output for classification. The proposed variant adds a training signal from human rationale annotations:

```text
text
-> BERT / RoBERTa encoder
-> optional VADER sentiment feature fusion
-> MLP classifier
-> cross-entropy loss
   + optional rationale-aware attention loss
```

The rationale-aware attention loss encourages the model to place more attention on tokens that human annotators marked as important for the hate/offensive/normal decision.

## XAI Analysis

XAI was used as a post-hoc verification layer, not as the primary performance claim.

Main observations:

- SHAP and LIME were used to inspect important tokens.
- Rationale alignment was checked with top-k token overlap against human rationale.
- Faithfulness-style checks tested how predictions changed when important tokens were masked or removed.
- XAI evidence suggested improved explanation stability in selected A_B vs D_B cases, but this part should be interpreted as qualitative/supporting evidence rather than full statistical proof.

Important distinction:

- **Final performance model:** `B_B`
- **Representative XAI comparison:** `A_B` vs `D_B`

The XAI comparison was limited by compute time, so it should not be overstated as a complete explanation of the final B_B result.

## Team Contribution Mapping

This section is reserved for mapping each team member to their project responsibilities and final artifacts. Do not include private contact information in this public README.

| Member | Role | Main Contributions | Related Files / Artifacts | Evidence To Check |
|---|---|---|---|---|
| TBD | Project lead / integration | Pipeline design, experiment integration, final review | `v2/run.sh`, `v2/pipeline/`, `README.md` | E2E run logs, final report, commit history |
| TBD | Data / preprocessing | Dataset preparation, label mapping, split validation | `v2/runtime/`, `v2/configs/v2_15seed.json` | split policy, preprocessing notes |
| TBD | Model training | BERT/RoBERTa training, seed runs, checkpoint handling | `v2/runtime/experiment_core.py`, `v2/pipeline/training_adapter.py` | completed runs, benchmark CSVs |
| TBD | Statistics / analysis | Macro F1 comparison, paired tests, Holm correction, ANOVA interpretation | `v2/pipeline/statistics.py`, `v2/outputs/experiments/v2_15seed/benchmark/` | summary tables, statistical results |
| TBD | XAI / reporting | SHAP/LIME interpretation, rationale alignment, report/dashboard materials | `v2/pipeline/xai*.py`, `v2/docs/04_xai_protocol.md`, `outputs/` | XAI visuals, report sections |

## Repository Structure

```text
.
├── README.md
├── hatespeech/                  # Earlier modularized code snapshot
├── v1/                          # Archived first pipeline and earlier outputs
├── v2/                          # Main reproducible experiment workspace
│   ├── configs/                 # v2_15seed experiment config
│   ├── docs/                    # Model, pipeline, statistics, XAI, runbook docs
│   ├── pipeline/                # E2E orchestration, manifest, statistics, reporting
│   ├── runtime/                 # Training, inference, dashboard, XAI runtime code
│   ├── scripts/                 # Server run, backup, gate check utilities
│   └── outputs/                 # Generated experiment outputs
└── outputs/                     # Optional local analysis/visualization artifacts
```

The canonical implementation is under [`v2/`](v2/). The `v1/` directory is kept as an archive for comparison and project history.

## Reproducing The Experiment

The full experiment requires a CUDA GPU environment. The final run was executed on a cloud GPU instance and used 15 seeds across 8 conditions.

```bash
git clone https://github.com/WinterFlw/Big_data_Programming.git
cd Big_data_Programming

python -m venv .venv
source .venv/bin/activate
pip install -r v2/requirements.txt

cd v2
./run.sh e2e status --run-id v2_15seed
./run.sh e2e plan --run-id v2_15seed --force
./run.sh e2e benchmark --run-id v2_15seed --conditions A_B --seeds 42 --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
```

For the full 120-run benchmark:

```bash
cd v2
./run.sh e2e benchmark --run-id v2_15seed --execute --resume
./run.sh e2e aggregate --run-id v2_15seed
./run.sh e2e xai-primary --run-id v2_15seed
./run.sh e2e xai-bundle --run-id v2_15seed
./run.sh e2e report --run-id v2_15seed
./run.sh e2e dashboard --run-id v2_15seed
```

Large checkpoints and some generated artifacts are intentionally excluded from version control. Re-running the pipeline recreates the standard CSV/JSON/report/dashboard outputs under `v2/outputs/experiments/v2_15seed/`.

## Key Documents

| Document | Purpose |
|---|---|
| [`v2/docs/01_model_definition.md`](v2/docs/01_model_definition.md) | Model definition and condition naming |
| [`v2/docs/02_e2e_pipeline.md`](v2/docs/02_e2e_pipeline.md) | End-to-end pipeline overview |
| [`v2/docs/03_validation_and_statistics.md`](v2/docs/03_validation_and_statistics.md) | Statistical validation plan |
| [`v2/docs/04_xai_protocol.md`](v2/docs/04_xai_protocol.md) | XAI protocol and interpretation rules |
| [`v2/docs/06_execution_runbook.md`](v2/docs/06_execution_runbook.md) | Server/GPU execution guide |
| [`v2/configs/v2_15seed.json`](v2/configs/v2_15seed.json) | Final experiment configuration |

## Limitations

- The performance gain is statistically significant but numerically small.
- VADER sentiment scores did not provide a strong independent improvement.
- XAI analysis was performed on representative checkpoints due to compute limits, not on every full-run checkpoint.
- The project uses HateXplain, so conclusions should be interpreted within that dataset distribution.
- The repository excludes large model checkpoints; exact checkpoint-level reproduction requires rerunning training.

## Takeaway

Rationale-aware attention supervision provided a small but consistent improvement over the BERT baseline in this controlled experiment. The project is most useful as a reproducible framework for comparing performance, statistical reliability, and explanation behavior in hate speech detection models.
