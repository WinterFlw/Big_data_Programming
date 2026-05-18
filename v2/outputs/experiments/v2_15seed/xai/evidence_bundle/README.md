# XAI Evidence Bundle

run_id: `v2_15seed`

This directory is the canonical XAI bundle for report/dashboard stages.
Files are populated automatically from primary/deep/ablation artifacts.
If a file appears empty, the corresponding upstream stage has not produced
its metric rows yet — re-run the relevant `./run.sh e2e xai-*` command.

Report and dashboard code should prefer:

- `xai_claims.json`
- `xai_dashboard_bundle.json`
- `xai_interpretation_cards.json`

Raw per-sample evidence remains available through the CSV/JSONL files in this
directory.
