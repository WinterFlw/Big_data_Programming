"""Self-contained interactive dashboard generator for experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_core import (
    BENCHMARK_RUNS_PATH,
    BENCHMARK_SUMMARY_PATH,
    CONFIG_PATH,
    DATA_PROFILE_PATH,
    FREEZE_STUDY_PATH,
    FREEZE_STUDY_MARKDOWN_PATH,
    TUNING_LOG_PATH,
)
from utils import OUTPUT_DIR, REPORT_DIR, TUNING_DIR, XAI_DIR, ensure_dir, load_json, save_json, save_text


DASHBOARD_DIR = OUTPUT_DIR / "dashboard"
DASHBOARD_HTML_PATH = DASHBOARD_DIR / "index.html"
DASHBOARD_BUNDLE_PATH = DASHBOARD_DIR / "dashboard_bundle.json"


def _load_optional_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _load_optional_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


def _relpath(path: Path) -> str:
    return str(path.relative_to(DASHBOARD_DIR.parent))


def _artifact_map() -> dict[str, dict[str, Any]]:
    candidates = {
        "config": CONFIG_PATH,
        "data_profile_json": DATA_PROFILE_PATH,
        "split_distribution_csv": REPORT_DIR / "split_distribution.csv",
        "split_distribution_png": REPORT_DIR / "split_distribution.png",
        "benchmark_runs_csv": BENCHMARK_RUNS_PATH,
        "benchmark_summary_csv": BENCHMARK_SUMMARY_PATH,
        "benchmark_summary_md": REPORT_DIR / "benchmark_summary.md",
        "benchmark_plot_png": REPORT_DIR / "model_comparison.png",
        "per_class_heatmap_png": REPORT_DIR / "per_class_f1_heatmap.png",
        "freeze_study_csv": FREEZE_STUDY_PATH,
        "freeze_study_md": FREEZE_STUDY_MARKDOWN_PATH,
        "tuning_log_csv": TUNING_LOG_PATH,
        "tuning_best_md": TUNING_DIR / "transformer_tuning_best.md",
        "xai_summary_json": XAI_DIR / "xai_summary.json",
        "xai_summary_md": XAI_DIR / "xai_summary.md",
        "xai_case_summary_csv": XAI_DIR / "case_summary.csv",
        "xai_overlap_csv": XAI_DIR / "overlap_at_5.csv",
        "xai_overlap_png": XAI_DIR / "overlap_at_5.png",
    }
    return {
        key: {
            "exists": path.exists(),
            "path": _relpath(path) if path.exists() else str(path.relative_to(OUTPUT_DIR)),
        }
        for key, path in candidates.items()
    }


def _group_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("model", dropna=False)["overlap_at_5"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "overlap_mean", "count": "samples"})
    )
    return grouped.where(pd.notnull(grouped), None).to_dict(orient="records")


def build_dashboard_bundle() -> dict[str, Any]:
    """Build a dashboard bundle from saved artifacts."""
    benchmark_summary = _load_optional_csv(BENCHMARK_SUMMARY_PATH)
    benchmark_runs = _load_optional_csv(BENCHMARK_RUNS_PATH)
    freeze_rows = _load_optional_csv(FREEZE_STUDY_PATH)
    tuning_rows = _load_optional_csv(TUNING_LOG_PATH)
    split_rows = _load_optional_csv(REPORT_DIR / "split_distribution.csv")
    xai_summary = _load_optional_json(XAI_DIR / "xai_summary.json")
    xai_case_rows = _load_optional_csv(XAI_DIR / "case_summary.csv")
    xai_overlap_rows = _load_optional_csv(XAI_DIR / "overlap_at_5.csv")

    available_images = []
    for image_path in [
        REPORT_DIR / "split_distribution.png",
        REPORT_DIR / "model_comparison.png",
        REPORT_DIR / "per_class_f1_heatmap.png",
        XAI_DIR / "overlap_at_5.png",
    ]:
        if image_path.exists():
            available_images.append(
                {
                    "label": image_path.stem.replace("_", " ").title(),
                    "path": _relpath(image_path),
                }
            )

    best_model = benchmark_summary[0]["model"] if benchmark_summary else None
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": _load_optional_json(CONFIG_PATH) or {},
        "data_profile": _load_optional_json(DATA_PROFILE_PATH) or {},
        "split_distribution": split_rows,
        "benchmark_summary": benchmark_summary,
        "benchmark_runs": benchmark_runs,
        "freeze_study": freeze_rows,
        "tuning_log": tuning_rows,
        "xai_summary": xai_summary or {},
        "xai_cases": xai_case_rows,
        "xai_overlap": xai_overlap_rows,
        "xai_overlap_grouped": _group_overlap(xai_overlap_rows),
        "artifacts": _artifact_map(),
        "images": available_images,
        "best_model": best_model,
    }


def _render_dashboard_html(bundle: dict[str, Any]) -> str:
    data_json = json.dumps(bundle, ensure_ascii=False)
    template = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hate Speech Study Dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 251, 245, 0.88);
      --ink: #1f2430;
      --muted: #6a7280;
      --line: rgba(31, 36, 48, 0.12);
      --accent: #cc6b2c;
      --accent-2: #0f766e;
      --good: #2f855a;
      --warn: #b7791f;
      --bad: #c53030;
      --shadow: 0 18px 50px rgba(65, 39, 16, 0.10);
      --radius: 20px;
      --sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      --serif: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(204, 107, 44, 0.18), transparent 28%),
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30%),
        linear-gradient(180deg, #fbf7f1 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px 20px 64px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,246,235,0.92));
      border: 1px solid rgba(204, 107, 44, 0.16);
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-family: var(--serif);
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 64ch;
      line-height: 1.6;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .meta-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .meta-card {{
      padding: 16px 18px;
    }}
    .meta-card .label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .meta-card .value {{
      font-size: 1.35rem;
      font-weight: 700;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 20px;
    }}
    .tab-btn {{
      border: none;
      border-radius: 999px;
      padding: 12px 16px;
      background: rgba(31, 36, 48, 0.06);
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
      transition: 180ms ease;
    }}
    .tab-btn.active {{
      background: linear-gradient(135deg, var(--accent), #e39c57);
      color: white;
      box-shadow: 0 10px 25px rgba(204, 107, 44, 0.28);
    }}
    .tab-section {{
      display: none;
      animation: fade 180ms ease;
    }}
    .tab-section.active {{
      display: block;
    }}
    @keyframes fade {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .grid.cols-2 {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .grid.cols-3 {{
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .panel {{
      padding: 22px;
    }}
    .panel h2, .panel h3 {{
      margin: 0 0 14px;
      font-family: var(--serif);
      letter-spacing: -0.03em;
    }}
    .panel .sub {{
      color: var(--muted);
      margin-top: -4px;
      margin-bottom: 16px;
      font-size: 0.95rem;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    select, input {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      min-width: 180px;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .stat {{
      padding: 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
    }}
    .stat .k {{
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 8px;
    }}
    .stat .v {{
      font-size: 1.45rem;
      font-weight: 800;
    }}
    .bar-list {{
      display: grid;
      gap: 10px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 180px 1fr 120px;
      gap: 12px;
      align-items: center;
    }}
    .bar-label {{
      font-weight: 700;
      font-size: 0.95rem;
    }}
    .bar-track {{
      height: 16px;
      background: rgba(31, 36, 48, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), #efb36f);
    }}
    .bar-value {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .pill {{
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .pill.ready {{ background: rgba(47, 133, 90, 0.12); color: var(--good); }}
    .pill.missing {{ background: rgba(197, 48, 48, 0.10); color: var(--bad); }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .image-card img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      display: block;
    }}
    .case-nav {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .nav-buttons {{
      display: flex;
      gap: 8px;
    }}
    .nav-buttons button {{
      border: none;
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(31,36,48,0.08);
      cursor: pointer;
      font-weight: 700;
    }}
    .token-box {{
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      min-height: 140px;
    }}
    .token-box h4 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .token-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .token {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(204, 107, 44, 0.10);
      color: #8d4c20;
      font-weight: 700;
      font-size: 0.9rem;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px 0;
      font-style: italic;
    }}
    .artifact-list {{
      display: grid;
      gap: 10px;
    }}
    .artifact-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.75);
    }}
    .artifact-item a {{
      color: var(--accent-2);
      text-decoration: none;
      word-break: break-all;
    }}
    .note {{
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }}
    @media (max-width: 980px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .grid.cols-2, .grid.cols-3 {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <h1>Experiment Dashboard</h1>
        <p>Repeated benchmark results, freeze-study comparison, tuning logs, and XAI cases in one interactive view. This page is self-contained and can be opened directly in a browser without a separate server.</p>
      </div>
      <div class="meta-grid">
        <div class="meta-card">
          <div class="label">Generated At</div>
          <div class="value" id="meta-generated-at">-</div>
        </div>
        <div class="meta-card">
          <div class="label">Best Model</div>
          <div class="value" id="meta-best-model">-</div>
        </div>
        <div class="meta-card">
          <div class="label">Seed Runs</div>
          <div class="value" id="meta-run-count">-</div>
        </div>
        <div class="meta-card">
          <div class="label">XAI Cases</div>
          <div class="value" id="meta-xai-cases">-</div>
        </div>
      </div>
    </section>

    <div class="tabs">
      <button class="tab-btn active" data-tab="overview">Overview</button>
      <button class="tab-btn" data-tab="benchmark">Benchmark</button>
      <button class="tab-btn" data-tab="freeze">Freeze Study</button>
      <button class="tab-btn" data-tab="tuning">Tuning</button>
      <button class="tab-btn" data-tab="xai">XAI</button>
      <button class="tab-btn" data-tab="artifacts">Artifacts</button>
    </div>

    <section class="tab-section active" data-tab-content="overview">
      <div class="grid cols-2">
        <div class="panel">
          <h2>Run Snapshot</h2>
          <div class="sub">Quick read of the current experiment bundle</div>
          <div class="stat-grid" id="overview-stats"></div>
        </div>
        <div class="panel">
          <h2>Dataset Split</h2>
          <div class="sub">70/10/20 stratified split summary</div>
          <div id="split-table"></div>
        </div>
      </div>
      <div class="grid cols-2" style="margin-top:18px;">
        <div class="panel">
          <h2>Topline Comparison</h2>
          <div class="sub">Interactive metric bars from the benchmark summary</div>
          <div class="controls">
            <select id="overview-metric"></select>
          </div>
          <div class="bar-list" id="overview-bars"></div>
        </div>
        <div class="panel">
          <h2>Saved Figures</h2>
          <div class="sub">Rendered PNG artifacts available from the current run</div>
          <div class="image-grid" id="overview-images"></div>
        </div>
      </div>
    </section>

    <section class="tab-section" data-tab-content="benchmark">
      <div class="panel">
        <h2>Benchmark Explorer</h2>
        <div class="controls">
          <select id="benchmark-metric"></select>
          <input id="benchmark-filter" placeholder="Filter model name">
        </div>
        <div class="bar-list" id="benchmark-bars"></div>
      </div>
      <div class="panel" style="margin-top:18px;">
        <h2>Benchmark Summary Table</h2>
        <div id="benchmark-table"></div>
      </div>
      <div class="panel" style="margin-top:18px;">
        <h2>Per-run Records</h2>
        <div class="sub">Seed-level outputs from repeated runs</div>
        <div id="run-table"></div>
      </div>
    </section>

    <section class="tab-section" data-tab-content="freeze">
      <div class="panel">
        <h2>Freeze vs Fine-tuning</h2>
        <div class="sub">Uses the tuned BERT+VADER settings to isolate encoder freezing effects</div>
        <div class="controls">
          <select id="freeze-metric"></select>
        </div>
        <div class="bar-list" id="freeze-bars"></div>
      </div>
      <div class="panel" style="margin-top:18px;">
        <h2>Freeze Study Table</h2>
        <div id="freeze-table"></div>
      </div>
    </section>

    <section class="tab-section" data-tab-content="tuning">
      <div class="panel">
        <h2>Tuning Log Explorer</h2>
        <div class="controls">
          <select id="tuning-model-filter"></select>
          <select id="tuning-parameter-filter"></select>
        </div>
        <div id="tuning-table"></div>
      </div>
    </section>

    <section class="tab-section" data-tab-content="xai">
      <div class="grid cols-2">
        <div class="panel">
          <h2>XAI Summary</h2>
          <div class="stat-grid" id="xai-stats"></div>
        </div>
        <div class="panel">
          <h2>Overlap@5</h2>
          <div class="sub">Mean overlap between SHAP Top-5 and LIME Top-5 tokens</div>
          <div class="bar-list" id="xai-overlap-bars"></div>
        </div>
      </div>
      <div class="panel" style="margin-top:18px;">
        <div class="case-nav">
          <div>
            <h2 style="margin-bottom:4px;">Case Browser</h2>
            <div class="sub" id="case-counter">No cases</div>
          </div>
          <div class="nav-buttons">
            <button id="case-prev" type="button">Previous</button>
            <button id="case-next" type="button">Next</button>
          </div>
        </div>
        <div id="case-body"></div>
      </div>
    </section>

    <section class="tab-section" data-tab-content="artifacts">
      <div class="panel">
        <h2>Artifact Browser</h2>
        <div class="sub">Quick links to the generated CSV, JSON, Markdown, and PNG outputs</div>
        <div class="artifact-list" id="artifact-list"></div>
      </div>
    </section>
  </div>

  <script id="dashboard-data" type="application/json">__DATA_JSON__</script>
  <script>
    const dashboard = JSON.parse(document.getElementById('dashboard-data').textContent);
    const metricOptions = [
      {{ key: 'macro_f1_mean', label: 'Macro F1' }},
      {{ key: 'macro_precision_mean', label: 'Macro Precision' }},
      {{ key: 'macro_recall_mean', label: 'Macro Recall' }},
      {{ key: 'accuracy_mean', label: 'Accuracy' }},
      {{ key: 'auroc_mean', label: 'AUROC' }}
    ];

    function formatValue(value, digits = 4) {{
      if (value === null || value === undefined || value === '') return 'N/A';
      if (typeof value === 'number') return value.toFixed(digits);
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed.toFixed(digits) : String(value);
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function setTab(tabName) {{
      document.querySelectorAll('.tab-btn').forEach((button) => {{
        button.classList.toggle('active', button.dataset.tab === tabName);
      }});
      document.querySelectorAll('.tab-section').forEach((section) => {{
        section.classList.toggle('active', section.dataset.tabContent === tabName);
      }});
    }}

    function mountMetricSelect(id) {{
      const select = document.getElementById(id);
      metricOptions.forEach((option) => {{
        const el = document.createElement('option');
        el.value = option.key;
        el.textContent = option.label;
        select.appendChild(el);
      }});
      select.value = metricOptions[0].key;
      return select;
    }}

    function renderBarList(targetId, rows, metricKey, labelKey = 'model') {{
      const host = document.getElementById(targetId);
      if (!rows.length) {{
        host.innerHTML = '<div class="empty">No data available yet.</div>';
        return;
      }}
      const values = rows.map((row) => Number(row[metricKey] ?? 0));
      const maxValue = Math.max(...values, 0.0001);
      host.innerHTML = rows.map((row) => {{
        const value = Number(row[metricKey] ?? 0);
        const width = Math.max((value / maxValue) * 100, 2);
        const stdKey = metricKey.replace('_mean', '_std');
        const stdText = row[stdKey] !== undefined && row[stdKey] !== null ? ` +/- ${formatValue(row[stdKey])}` : '';
        return `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(row[labelKey])}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <div class="bar-value">${formatValue(value)}${stdText}</div>
          </div>
        `;
      }}).join('');
    }}

    function renderTable(targetId, rows, columns) {{
      const host = document.getElementById(targetId);
      if (!rows.length) {{
        host.innerHTML = '<div class="empty">No data available yet.</div>';
        return;
      }}
      const head = columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join('');
      const body = rows.map((row) => {{
        return `<tr>${columns.map((column) => `<td>${escapeHtml(row[column.key] ?? 'N/A')}</td>`).join('')}</tr>`;
      }}).join('');
      host.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }}

    function renderOverview() {{
      document.getElementById('meta-generated-at').textContent = dashboard.generated_at || '-';
      document.getElementById('meta-best-model').textContent = dashboard.best_model || 'Pending';
      document.getElementById('meta-run-count').textContent = String(dashboard.benchmark_runs.length || 0);
      document.getElementById('meta-xai-cases').textContent = String(dashboard.xai_cases.length || 0);

      const stats = [
        {{ key: 'total_samples', label: 'Total Samples', value: dashboard.data_profile.total_samples ?? 'N/A' }},
        {{ key: 'split_count', label: 'Tracked Splits', value: dashboard.split_distribution.length || 0 }},
        {{ key: 'benchmark_models', label: 'Benchmarked Models', value: dashboard.benchmark_summary.length || 0 }},
        {{ key: 'artifact_count', label: 'Ready Artifacts', value: Object.values(dashboard.artifacts).filter(item => item.exists).length }}
      ];
      document.getElementById('overview-stats').innerHTML = stats.map((item) => `
        <div class="stat"><div class="k">${escapeHtml(item.label)}</div><div class="v">${escapeHtml(item.value)}</div></div>
      `).join('');

      renderTable('split-table', dashboard.split_distribution, [
        {{ key: 'split', label: 'Split' }},
        {{ key: 'samples', label: 'Samples' }},
        {{ key: 'hatespeech', label: 'Hatespeech' }},
        {{ key: 'offensive', label: 'Offensive' }},
        {{ key: 'normal', label: 'Normal' }}
      ]);

      const imageHost = document.getElementById('overview-images');
      if (!dashboard.images.length) {{
        imageHost.innerHTML = '<div class="empty">No images generated yet.</div>';
      }} else {{
        imageHost.innerHTML = dashboard.images.map((item) => `
          <div class="image-card">
            <div class="note" style="margin-bottom:8px; font-weight:700;">${escapeHtml(item.label)}</div>
            <img src="../${escapeHtml(item.path)}" alt="${escapeHtml(item.label)}">
          </div>
        `).join('');
      }}
    }}

    function renderBenchmark(metricKey, modelFilter = '') {{
      const rows = dashboard.benchmark_summary.filter((row) =>
        row.model.toLowerCase().includes(modelFilter.toLowerCase())
      );
      renderBarList('benchmark-bars', rows, metricKey);
      renderTable('benchmark-table', rows, [
        {{ key: 'model', label: 'Model' }},
        {{ key: 'macro_f1_display', label: 'Macro F1' }},
        {{ key: 'macro_precision_display', label: 'Macro Precision' }},
        {{ key: 'macro_recall_display', label: 'Macro Recall' }},
        {{ key: 'accuracy_display', label: 'Accuracy' }},
        {{ key: 'auroc_display', label: 'AUROC' }}
      ]);
      renderBarList('overview-bars', dashboard.benchmark_summary, document.getElementById('overview-metric').value);
    }}

    function renderRunTable() {{
      renderTable('run-table', dashboard.benchmark_runs, [
        {{ key: 'model', label: 'Model' }},
        {{ key: 'seed', label: 'Seed' }},
        {{ key: 'best_epoch', label: 'Best Epoch' }},
        {{ key: 'best_val_macro_f1', label: 'Best Val Macro F1' }},
        {{ key: 'macro_f1', label: 'Test Macro F1' }},
        {{ key: 'accuracy', label: 'Accuracy' }}
      ]);
    }}

    function renderFreeze(metricKey) {{
      const rows = dashboard.freeze_study;
      if (!rows.length) {{
        document.getElementById('freeze-bars').innerHTML = '<div class="empty">Freeze-study artifacts are not available yet.</div>';
        document.getElementById('freeze-table').innerHTML = '<div class="empty">Freeze-study artifacts are not available yet.</div>';
        return;
      }}

      const frame = rows.reduce((acc, row) => {{
        const model = row.model;
        if (!acc[model]) acc[model] = {{ model, values: [] }};
        if (row.macro_f1 !== null && row.macro_f1 !== undefined) {{
          acc[model].values.push(row);
        }}
        return acc;
      }}, {{}});

      const summary = Object.values(frame).map((entry) => {{
        const values = entry.values.map((row) => Number(row[metricKey.replace('_mean', '')] ?? 0));
        const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
        return {{ model: entry.model, [metricKey]: mean }};
      }});

      renderBarList('freeze-bars', summary, metricKey);
      renderTable('freeze-table', rows, [
        {{ key: 'model', label: 'Model' }},
        {{ key: 'seed', label: 'Seed' }},
        {{ key: 'best_epoch', label: 'Best Epoch' }},
        {{ key: 'macro_f1', label: 'Macro F1' }},
        {{ key: 'accuracy', label: 'Accuracy' }}
      ]);
    }}

    function renderTuning() {{
      const modelFilterEl = document.getElementById('tuning-model-filter');
      const parameterFilterEl = document.getElementById('tuning-parameter-filter');
      const rows = dashboard.tuning_log;

      const models = ['All Models', ...new Set(rows.map((row) => row.model))];
      const parameters = ['All Params', ...new Set(rows.map((row) => row.parameter))];

      modelFilterEl.innerHTML = models.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join('');
      parameterFilterEl.innerHTML = parameters.map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`).join('');

      function updateTable() {{
        const modelValue = modelFilterEl.value;
        const parameterValue = parameterFilterEl.value;
        const filtered = rows.filter((row) => {{
          const modelOk = modelValue === 'All Models' || row.model === modelValue;
          const paramOk = parameterValue === 'All Params' || row.parameter === parameterValue;
          return modelOk && paramOk;
        }});
        renderTable('tuning-table', filtered, [
          {{ key: 'model', label: 'Model' }},
          {{ key: 'parameter', label: 'Parameter' }},
          {{ key: 'candidate', label: 'Candidate' }},
          {{ key: 'val_macro_f1', label: 'Val Macro F1' }},
          {{ key: 'seed', label: 'Seed' }}
        ]);
      }}

      modelFilterEl.addEventListener('change', updateTable);
      parameterFilterEl.addEventListener('change', updateTable);
      updateTable();
    }}

    let currentCaseIndex = 0;
    function renderXai() {{
      const summary = dashboard.xai_summary || {{}};
      const stats = [
        {{ label: 'Baseline Model', value: summary.baseline_model || 'Pending' }},
        {{ label: 'Improved Model', value: summary.improved_model || 'Pending' }},
        {{ label: 'Baseline Macro F1', value: formatValue(summary.baseline_macro_f1) }},
        {{ label: 'Improved Macro F1', value: formatValue(summary.improved_macro_f1) }},
        {{ label: 'Baseline Overlap Mean', value: formatValue(summary.baseline_overlap_mean) }},
        {{ label: 'Improved Overlap Mean', value: formatValue(summary.improved_overlap_mean) }},
        {{ label: 'XAI Samples', value: summary.sample_count ?? 0 }},
        {{ label: 'Fixed Errors', value: summary.fixed_error_count ?? 0 }}
      ];
      document.getElementById('xai-stats').innerHTML = stats.map((item) => `
        <div class="stat"><div class="k">${escapeHtml(item.label)}</div><div class="v">${escapeHtml(item.value)}</div></div>
      `).join('');

      const overlapRows = dashboard.xai_overlap_grouped.map((row) => ({{
        model: row.model,
        overlap_mean: Number(row.overlap_mean ?? 0),
        overlap_mean_std: null
      }}));
      renderBarList('xai-overlap-bars', overlapRows, 'overlap_mean');

      function drawCase(index) {{
        const cases = dashboard.xai_cases;
        const host = document.getElementById('case-body');
        const counter = document.getElementById('case-counter');
        if (!cases.length) {{
          host.innerHTML = '<div class="empty">No XAI cases are available yet.</div>';
          counter.textContent = 'No cases';
          return;
        }}
        currentCaseIndex = (index + cases.length) % cases.length;
        const item = cases[currentCaseIndex];
        counter.textContent = `Case ${currentCaseIndex + 1} / ${cases.length} - ${item.category || 'case'}`;
        host.innerHTML = `
          <div class="note" style="margin-bottom:14px;">
            Sample ID: ${escapeHtml(item.sample_id)}<br>
            Baseline Overlap@5: ${escapeHtml(formatValue(item.baseline_overlap_at_5))}<br>
            Improved Overlap@5: ${escapeHtml(formatValue(item.improved_overlap_at_5))}
          </div>
          <div class="grid cols-2">
            <div class="token-box">
              <h4>Baseline Top Tokens</h4>
              <div class="token-list">
                ${String(item.baseline_top_tokens || '').split(',').filter(Boolean).map((token) => `<span class="token">${escapeHtml(token.trim())}</span>`).join('') || '<span class="empty">No tokens</span>'}
              </div>
            </div>
            <div class="token-box">
              <h4>Improved Model Top Tokens</h4>
              <div class="token-list">
                ${String(item.improved_top_tokens || '').split(',').filter(Boolean).map((token) => `<span class="token">${escapeHtml(token.trim())}</span>`).join('') || '<span class="empty">No tokens</span>'}
              </div>
            </div>
          </div>
        `;
      }}

      document.getElementById('case-prev').onclick = () => drawCase(currentCaseIndex - 1);
      document.getElementById('case-next').onclick = () => drawCase(currentCaseIndex + 1);
      drawCase(0);
    }}

    function renderArtifacts() {{
      const host = document.getElementById('artifact-list');
      const entries = Object.entries(dashboard.artifacts);
      host.innerHTML = entries.map(([name, item]) => `
        <div class="artifact-item">
          <div>
            <div style="font-weight:700; margin-bottom:4px;">${escapeHtml(name)}</div>
            <a href="../${escapeHtml(item.path)}">${escapeHtml(item.path)}</a>
          </div>
          <span class="pill ${item.exists ? 'ready' : 'missing'}">${item.exists ? 'READY' : 'MISSING'}</span>
        </div>
      `).join('');
    }}

    document.querySelectorAll('.tab-btn').forEach((button) => {{
      button.addEventListener('click', () => setTab(button.dataset.tab));
    }});

    const overviewMetric = mountMetricSelect('overview-metric');
    const benchmarkMetric = mountMetricSelect('benchmark-metric');
    const freezeMetric = mountMetricSelect('freeze-metric');

    overviewMetric.addEventListener('change', () => {{
      renderBarList('overview-bars', dashboard.benchmark_summary, overviewMetric.value);
    }});
    benchmarkMetric.addEventListener('change', () => {{
      renderBenchmark(benchmarkMetric.value, document.getElementById('benchmark-filter').value);
    }});
    document.getElementById('benchmark-filter').addEventListener('input', (event) => {{
      renderBenchmark(benchmarkMetric.value, event.target.value);
    }});
    freezeMetric.addEventListener('change', () => renderFreeze(freezeMetric.value));

    renderOverview();
    renderBenchmark(benchmarkMetric.value, '');
    renderRunTable();
    renderFreeze(freezeMetric.value);
    renderTuning();
    renderXai();
    renderArtifacts();
  </script>
</body>
</html>
"""
    return template.replace("{{", "{").replace("}}", "}").replace("__DATA_JSON__", data_json)


def run_dashboard() -> Path:
    """Generate the dashboard HTML and its JSON bundle."""
    ensure_dir(DASHBOARD_DIR)
    bundle = build_dashboard_bundle()
    save_json(bundle, DASHBOARD_BUNDLE_PATH)
    save_text(_render_dashboard_html(bundle), DASHBOARD_HTML_PATH)
    return DASHBOARD_HTML_PATH
