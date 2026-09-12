"""Read-only manuscript checks plus generated PDF QA previews/report."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--qa-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    qa_dir = args.qa_dir.resolve()
    qa_dir.mkdir(parents=True, exist_ok=True)

    abstract = (package / "sections/abstract.tex").read_text(encoding="utf-8")
    abstract = re.sub(r"\\(?:begin|end)\{abstract\}", "", abstract)
    abstract_words = len(abstract.split())
    assert 100 <= abstract_words <= 150, abstract_words

    table = (package / "tables/main_results.tex").read_text(encoding="utf-8")
    source = (package / "notes/source_zh.md").read_text(encoding="utf-8")
    source_rows = {}
    for row in source.splitlines():
        if not row.startswith("| "):
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) != 11:
            continue
        try:
            source_rows[cells[1]] = [float(c.replace("*", "")) for c in cells[2:]]
        except ValueError:
            pass
    checked_rows = []
    for row in table.splitlines():
        if "&" not in row or not row.endswith(r"\\"):
            continue
        label = row.split("&")[0].strip()
        key = "NOVA + 选择性边界修正" if label.startswith(r"\NOVA") else label
        if key not in source_rows:
            continue
        cells = row.split("&")[1:]
        nums = [float(re.sub(r"\\textbf\{([^}]+)\}", r"\1", c).strip().rstrip("\\").strip()) for c in cells]
        assert nums == source_rows[key], (key, nums, source_rows[key])
        checked_rows.append(key)
    assert len(checked_rows) == 11, checked_rows

    pdf = pymupdf.open(package / "main.pdf")
    report = {
        "abstract_words": abstract_words,
        "main_table_rows_verified": checked_rows,
        "page_count": len(pdf),
        "pages": [],
        "style_hashes": {},
    }
    for style in ("spconf.sty", "IEEEbib.bst"):
        report["style_hashes"][style] = hashlib.sha256((package / style).read_bytes()).hexdigest()
    for number, page in enumerate(pdf, 1):
        text = page.get_text()
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"] if span["text"].strip()]
        bbox = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
                max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
        item = {
            "page": number, "size_points": [page.rect.width, page.rect.height],
            "text_bbox": bbox, "min_span_size": min(s["size"] for s in spans),
            "text_start": text[:180], "text_end": text[-180:],
        }
        report["pages"].append(item)
        assert abs(page.rect.width - 612) < 1 and abs(page.rect.height - 792) < 1, item
        # Mathematical scripts can be smaller than the body/table font.
        # Visible prose/table text is reviewed separately in the rendered pages.
        assert bbox[0] >= 52 and bbox[2] <= 562, item
        # Font-metric rectangles include descent padding beyond visible ink.
        # The official style's 229 mm text height ends at 721.1 pt; permit
        # 5 pt of metric padding, then visually inspect rendered page edges.
        assert bbox[1] >= 65 and bbox[3] <= 727, item
        page.get_pixmap(matrix=pymupdf.Matrix(1.35, 1.35)).save(qa_dir / f"page-{number}.png")
    assert len(pdf) <= 5, len(pdf)
    if len(pdf) == 5:
        assert "REFERENCES" in pdf[4].get_text(), "Page 5 must begin the references"
    report["pdf_sha256"] = hashlib.sha256((package / "main.pdf").read_bytes()).hexdigest()
    (qa_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
