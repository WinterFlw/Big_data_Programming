"""Report and dashboard scaffolding for v2 experiments.

The reporting stage should be able to run before full results exist. That lets
the team verify links, sections, and output paths locally, then rerun the same
command after the server fills benchmark and XAI artifacts.
"""

from __future__ import annotations

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


def _report_markdown_text(manifest: dict[str, Any], root: Path, counts: dict[str, int]) -> str:
    """Build conservative report text from the current artifact state."""
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

## Statistics

The primary comparison uses same-seed paired tests across 15 seeds. Holm correction is reserved for multiple comparisons.

## XAI

XAI is treated as post-hoc verification. Primary XAI compares A_B and D_B across all seeds, while deep XAI uses median-performing checkpoints for detailed cases.

## XAI Evidence Bundle

Report and dashboard generation should prefer the evidence bundle before reading raw XAI case files.

- `xai/evidence_bundle/xai_claims.json`: {_artifact_state(root, "xai/evidence_bundle/xai_claims.json")}
- `xai/evidence_bundle/xai_dashboard_bundle.json`: {_artifact_state(root, "xai/evidence_bundle/xai_dashboard_bundle.json")}
- `xai/evidence_bundle/xai_interpretation_cards.json`: {_artifact_state(root, "xai/evidence_bundle/xai_interpretation_cards.json")}
- `xai/evidence_bundle/xai_run_metadata.json`: {_artifact_state(root, "xai/evidence_bundle/xai_run_metadata.json")}
- `xai/evidence_bundle/token_attributions.jsonl`: {_artifact_state(root, "xai/evidence_bundle/token_attributions.jsonl")}

## Next Implementation Hooks

- Fill `benchmark/benchmark_runs.csv` from completed condition x seed runs.
- Fill `benchmark/paired_tests_holm.csv` after aggregation.
- Fill `xai/primary/seed_level_metrics.csv` after XAI execution.
- Fill `xai/evidence_bundle/xai_claims.json` from completed primary/deep/ablation XAI outputs.
- Replace conservative placeholder paragraphs after real benchmark/XAI results are populated.
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
