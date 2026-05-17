"""Report and dashboard scaffolding for v2 experiments.

The reporting stage should be able to run before full results exist. That lets
the team verify links, sections, and output paths locally, then rerun the same
command after the server fills benchmark and XAI artifacts.

v2.1 추가: benchmark_summary / paired_tests_holm / xai_dashboard_bundle 입력이
들어오면 markdown/docx 본문에 실제 표·문장을 자동으로 삽입한다. 입력이 비어 있으면
"_no XAI evidence yet — populate by running xai-primary/deep/ablation_" 같은
placeholder 메시지를 둔다.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile

from .artifacts import build_run_units, status_counts
from .manifest import manifest_hash
from .paths import experiment_root


def _artifact_state(root: Path, relative_path: str) -> str:
    """Return a compact status string for report/dashboard artifact links."""
    return "present" if (root / relative_path).exists() else "planned"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Tiny CSV reader: missing file → []."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Tiny JSON reader: missing or invalid → {}."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _fmt_float(value: Any, digits: int = 4) -> str:
    """Format a CSV cell that might be empty string into either '?.????' or '-'."""
    if value in ("", None):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


# ───────────────────────────────────────────────────────────
# 본문 섹션 자동 렌더링
# ───────────────────────────────────────────────────────────


def _render_benchmark_summary_table(root: Path) -> str:
    """benchmark_summary.csv 행을 mean ± std + 95% CI 마크다운 표로."""
    summary_rows = _read_csv_rows(root / "benchmark" / "benchmark_summary.csv")
    completed_rows = [row for row in summary_rows if row.get("n_seeds") and row["n_seeds"] not in ("", "0")]
    if not completed_rows:
        return "_no benchmark results yet — populate by running ./run.sh e2e benchmark --execute_"

    lines = [
        "| Condition | Backbone | N seeds | Macro F1 (mean ± std) | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for row in completed_rows:
        mean = _fmt_float(row.get("macro_f1_mean"))
        std = _fmt_float(row.get("macro_f1_std"))
        low = _fmt_float(row.get("macro_f1_ci_low"))
        high = _fmt_float(row.get("macro_f1_ci_high"))
        lines.append(
            f"| {row.get('condition', '?')} | {row.get('backbone', '?')} | {row.get('n_seeds', '?')} "
            f"| {mean} ± {std} | [{low}, {high}] |"
        )
    return "\n".join(lines)


def _render_paired_tests_table(root: Path, metric: str = "macro_f1") -> str:
    """paired_tests_holm.csv에서 metric에 해당하는 행만 추출해 표로."""
    paired_rows = _read_csv_rows(root / "benchmark" / "paired_tests_holm.csv")
    filtered = [row for row in paired_rows if row.get("metric") == metric and row.get("n_pairs") not in ("", "0")]
    if not filtered:
        return f"_no paired tests yet for metric `{metric}` — populate by running ./run.sh e2e aggregate_"

    lines = [
        "| Comparison | n pairs | Mean diff | p (raw) | p (Holm) | Cohen dz | sig@0.05 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in filtered:
        lines.append(
            f"| {row.get('comparison', '?')} | {row.get('n_pairs', '?')} | "
            f"{_fmt_float(row.get('mean_diff'))} | {_fmt_float(row.get('p_value'))} | "
            f"{_fmt_float(row.get('p_value_holm'))} | {_fmt_float(row.get('effect_size'))} | "
            f"{row.get('significant_0_05', '?')} |"
        )
    return "\n".join(lines)


def _render_anova_table(root: Path, name: str) -> str:
    """ANOVA CSV 한 파일을 마크다운 표로 렌더."""
    rows = _read_csv_rows(root / "benchmark" / f"{name}.csv")
    if not rows:
        return f"_no ANOVA rows yet for `{name}` — populate by running ./run.sh e2e aggregate after seeds complete_"

    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_float(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def _render_xai_claims(root: Path) -> str:
    """xai_dashboard_bundle.json + xai_claims.json에서 요약 카드·claim을 본문에."""
    bundle = _read_json(root / "xai" / "evidence_bundle" / "xai_dashboard_bundle.json")
    claims_payload = _read_json(root / "xai" / "evidence_bundle" / "xai_claims.json")
    parts: list[str] = []

    summary_cards = bundle.get("summary_cards") or []
    if summary_cards:
        parts.append("### Summary Cards")
        parts.append("")
        for card in summary_cards:
            title = card.get("title", "?")
            value = card.get("value", "?")
            source = card.get("source", "?")
            parts.append(f"- **{title}**: {value} (source: `{source}`)")
        parts.append("")

    claims = claims_payload.get("claims") or []
    if claims:
        parts.append("### Statistically-supported Claims")
        parts.append("")
        parts.append("| ID | Strength | Statement | Source |")
        parts.append("|---|---|---|---|")
        for claim in claims:
            sources = ", ".join(f"`{src}`" for src in claim.get("source_artifacts", []))
            parts.append(
                f"| {claim.get('id', '?')} | {claim.get('strength', '?')} | "
                f"{claim.get('text', '?')} | {sources} |"
            )

    if not parts:
        return (
            "_no XAI evidence yet — populate by running "
            "`./run.sh e2e xai-primary/xai-deep/xai-ablation` followed by `./run.sh e2e xai-bundle`._"
        )
    return "\n".join(parts)


def _render_seed_stability(root: Path) -> str:
    """seed_stability.csv 행을 표로."""
    rows = _read_csv_rows(root / "xai" / "primary" / "seed_stability.csv")
    if not rows:
        return "_no seed stability rows yet — populate by running ./run.sh e2e xai-primary after multiple seeds complete_"
    lines = [
        "| Condition | Metric | Mean | Std | CI Low | CI High |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('condition', '?')} | {row.get('metric', '?')} | "
            f"{_fmt_float(row.get('mean'))} | {_fmt_float(row.get('std'))} | "
            f"{_fmt_float(row.get('ci_low'))} | {_fmt_float(row.get('ci_high'))} |"
        )
    return "\n".join(lines)


def _render_limitations_text(root: Path) -> str:
    """xai_risk_flags.csv를 한계 서술의 출발점으로 사용."""
    rows = _read_csv_rows(root / "xai" / "evidence_bundle" / "xai_risk_flags.csv")
    if not rows:
        return (
            "_no automatic limitations recorded yet — limitations are populated_ "
            "_when XAI evidence is incomplete or paired test sample sizes are small._"
        )
    bullets = []
    for row in rows:
        bullets.append(
            f"- [{row.get('severity', '?')}] {row.get('flag_type', '?')}: "
            f"{row.get('evidence', '?')} (recommendation: {row.get('recommended_report_note', '?')})"
        )
    return "\n".join(bullets)


def _report_markdown_text(manifest: dict[str, Any], root: Path, counts: dict[str, int]) -> str:
    """Build report text. Tables are auto-filled from CSV/JSON when available."""
    return f"""# v2 Final Report

## Run

- run_id: `{manifest["run_id"]}`
- manifest_hash: `{manifest_hash(manifest)}`
- output_root: `{manifest["output_root"]}`

## Benchmark Status

- total units: {counts["total"]}
- completed: {counts["completed"]}
- failed: {counts["failed"]}
- planned: {counts["planned"]}

## Model

The v2 model family evaluates rationale-aware attention loss and VADER sentiment features through an 8-condition ablation matrix across BERT and RoBERTa backbones.

## Benchmark Summary

{_render_benchmark_summary_table(root)}

## Paired Tests (Holm-corrected, macro_f1)

The primary comparison uses same-seed paired tests across all seeds. Holm correction is applied for multiple comparisons.

{_render_paired_tests_table(root, metric="macro_f1")}

## ANOVA — BERT family (2-way)

Factors: attention loss × VADER.

{_render_anova_table(root, "anova_2way_bert")}

## ANOVA — RoBERTa family (2-way)

{_render_anova_table(root, "anova_2way_roberta")}

## ANOVA — Cross-family (3-way)

Factors: backbone × attention loss × VADER.

{_render_anova_table(root, "anova_3way")}

## XAI Evidence Summary

XAI is treated as post-hoc verification. Primary XAI compares A_B and D_B across all seeds, while deep XAI uses median-performing checkpoints for detailed cases.

{_render_xai_claims(root)}

## Seed Stability

Top-k Jaccard and rank correlation across seed checkpoints — high stability means explanations stay consistent under seed variation.

{_render_seed_stability(root)}

## Limitations

{_render_limitations_text(root)}

## XAI Evidence Bundle (file links)

Report and dashboard generation should prefer the evidence bundle before reading raw XAI case files.

- `xai/evidence_bundle/xai_claims.json`: {_artifact_state(root, "xai/evidence_bundle/xai_claims.json")}
- `xai/evidence_bundle/xai_dashboard_bundle.json`: {_artifact_state(root, "xai/evidence_bundle/xai_dashboard_bundle.json")}
- `xai/evidence_bundle/xai_interpretation_cards.json`: {_artifact_state(root, "xai/evidence_bundle/xai_interpretation_cards.json")}
- `xai/evidence_bundle/xai_run_metadata.json`: {_artifact_state(root, "xai/evidence_bundle/xai_run_metadata.json")}
- `xai/evidence_bundle/token_attributions.jsonl`: {_artifact_state(root, "xai/evidence_bundle/token_attributions.jsonl")}

## Reproducibility

- Manifest hash: `{manifest_hash(manifest)}`
- Conditions: {", ".join(manifest["benchmark"]["conditions"])}
- Seeds: {", ".join(str(seed) for seed in manifest["benchmark"]["seeds"])}
- Commands to reproduce (from `v2/`):
  - `./run.sh e2e plan --run-id {manifest["run_id"]}`
  - `./run.sh e2e benchmark --run-id {manifest["run_id"]} --execute`
  - `./run.sh e2e aggregate --run-id {manifest["run_id"]}`
  - `./run.sh e2e xai-primary --run-id {manifest["run_id"]}`
  - `./run.sh e2e xai-bundle --run-id {manifest["run_id"]}`
  - `./run.sh e2e report --run-id {manifest["run_id"]}`
"""


def _markdown_to_docx_paragraphs(markdown_text: str) -> list[tuple[str, str]]:
    """Convert the report scaffold into simple DOCX paragraph styles.

    This intentionally supports only headings, bullets, and body paragraphs.
    The goal is a dependency-free Word artifact that always exists in the v2
    output contract; richer formatting can be layered on after final results.
    """
    paragraphs: list[tuple[str, str]] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            paragraphs.append(("Title", line[2:].strip()))
        elif line.startswith("## "):
            paragraphs.append(("Heading1", line[3:].strip()))
        elif line.startswith("- "):
            paragraphs.append(("ListParagraph", line[2:].strip()))
        else:
            paragraphs.append(("Normal", line))
    return paragraphs


def _docx_paragraph_xml(style: str, text: str) -> str:
    """Render one minimal WordprocessingML paragraph."""
    style_xml = "" if style == "Normal" else f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
    safe_text = xml_escape(text)
    return f"<w:p>{style_xml}<w:r><w:t>{safe_text}</w:t></w:r></w:p>"


def _write_minimal_docx(path: Path, paragraphs: list[tuple[str, str]]) -> Path:
    """Write a valid DOCX without adding a heavy runtime dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document_body = "".join(_docx_paragraph_xml(style, text) for style, text in paragraphs)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {document_body}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
  </w:body>
</w:document>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
</w:styles>
"""
    with ZipFile(path, "w", ZIP_DEFLATED) as docx:
        docx.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
""",
        )
        docx.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""",
        )
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", styles_xml)
    return path


def generate_markdown_report(manifest: dict[str, Any]) -> Path:
    """Write a Markdown report scaffold for one run_id."""
    root = experiment_root(manifest["run_id"])
    units = build_run_units(manifest)
    counts = status_counts(units)
    report_path = root / "reports" / "final_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # This text is deliberately conservative. Until real results exist, it only
    # states the design and current artifact status. The future report builder
    # should replace the hook section with tables loaded from benchmark/XAI CSVs.
    text = _report_markdown_text(manifest, root, counts)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def generate_docx_report(manifest: dict[str, Any]) -> Path:
    """Write the Word counterpart required by the final report contract."""
    root = experiment_root(manifest["run_id"])
    units = build_run_units(manifest)
    counts = status_counts(units)
    text = _report_markdown_text(manifest, root, counts)
    docx_path = root / "reports" / "final_report.docx"
    return _write_minimal_docx(docx_path, _markdown_to_docx_paragraphs(text))


def generate_report_bundle(manifest: dict[str, Any]) -> dict[str, Path]:
    """Generate both canonical report formats for one run_id."""
    return {
        "markdown": generate_markdown_report(manifest),
        "docx": generate_docx_report(manifest),
    }


def generate_dashboard(manifest: dict[str, Any]) -> Path:
    """Write a minimal HTML dashboard for one run_id."""
    root = experiment_root(manifest["run_id"])
    units = build_run_units(manifest)
    counts = status_counts(units)
    dashboard_path = root / "dashboard" / "index.html"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)

    # Escape text that comes from the manifest so the dashboard remains safe if
    # a run_id or path contains punctuation.
    rows = "\n".join(
        f"<tr><th>{escape(key)}</th><td>{value}</td></tr>"
        for key, value in counts.items()
    )
    evidence_rows = "\n".join(
        f"<tr><th>{escape(relative_path)}</th><td>{_artifact_state(root, relative_path)}</td></tr>"
        for relative_path in [
            "xai/evidence_bundle/xai_claims.json",
            "xai/evidence_bundle/xai_dashboard_bundle.json",
            "xai/evidence_bundle/xai_interpretation_cards.json",
            "xai/evidence_bundle/xai_run_metadata.json",
            "xai/evidence_bundle/token_attributions.jsonl",
        ]
    )
    # benchmark summary와 XAI summary cards를 dashboard에도 노출.
    benchmark_rows_html = ""
    summary_rows = _read_csv_rows(root / "benchmark" / "benchmark_summary.csv")
    completed_rows = [r for r in summary_rows if r.get("n_seeds") and r["n_seeds"] not in ("", "0")]
    if completed_rows:
        body = "\n".join(
            (
                "<tr>"
                f"<td>{escape(row.get('condition', '?'))}</td>"
                f"<td>{escape(row.get('backbone', '?'))}</td>"
                f"<td>{escape(row.get('n_seeds', '?'))}</td>"
                f"<td>{_fmt_float(row.get('macro_f1_mean'))} ± {_fmt_float(row.get('macro_f1_std'))}</td>"
                "</tr>"
            )
            for row in completed_rows
        )
        benchmark_rows_html = (
            "<h2>Benchmark Summary</h2>"
            "<table><thead><tr><th>Condition</th><th>Backbone</th><th>N seeds</th>"
            "<th>Macro F1</th></tr></thead><tbody>"
            f"{body}</tbody></table>"
        )

    xai_cards_html = ""
    dashboard_bundle = _read_json(root / "xai" / "evidence_bundle" / "xai_dashboard_bundle.json")
    summary_cards = dashboard_bundle.get("summary_cards") or []
    if summary_cards:
        body = "\n".join(
            (
                "<tr>"
                f"<td>{escape(str(card.get('title', '?')))}</td>"
                f"<td>{escape(str(card.get('value', '?')))}</td>"
                f"<td><code>{escape(str(card.get('source', '?')))}</code></td>"
                "</tr>"
            )
            for card in summary_cards
        )
        xai_cards_html = (
            "<h2>XAI Summary Cards</h2>"
            "<table><thead><tr><th>Title</th><th>Value</th><th>Source</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>{escape(manifest["run_id"])} dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.5; }}
    table {{ border-collapse: collapse; min-width: 360px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{escape(manifest["run_id"])} dashboard</h1>
  <p>Output root: <code>{escape(manifest["output_root"])}</code></p>
  <table>
    <tbody>
      {rows}
    </tbody>
  </table>
  {benchmark_rows_html}
  {xai_cards_html}
  <h2>XAI Evidence Bundle</h2>
  <p>Report/dashboard stages should prefer these bundle artifacts before raw XAI case files.</p>
  <table>
    <tbody>
      {evidence_rows}
    </tbody>
  </table>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path
