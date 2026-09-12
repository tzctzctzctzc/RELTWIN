"""Validate the completed v4 manuscript against its evidence and compiled PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import pymupdf
from render_reltwin_v4_tables import fmt2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--qa-dir", type=Path, required=True)
    p.add_argument("--zip", type=Path)
    args = p.parse_args()
    repo, package, qa = args.repository.resolve(), args.package.resolve(), args.qa_dir.resolve()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    assert evidence["bridge_complete"]
    assert len(evidence["models"]) == 15
    assert evidence["audio_groups"] == {"relation": 80, "public": 387}
    assert len(evidence["source_component_sizes"]) == 39
    assert not evidence["train_dev_class_overlap"] and not evidence["train_dev_source_overlap"]
    for filename in ("spconf.sty", "IEEEbib.bst"):
        assert sha(package / filename) == sha(repo / "paper/overleaf_icassp2027_reltwin_review_v3" / filename)
    for filename, source in (("source_zh.md", "ICASSP_DRAFT_ZH_v8_BINDING.md"), ("source_en.md", "ICASSP_DRAFT_EN_v8_BINDING.md")):
        assert sha(package / "notes" / filename) == sha(repo / "paper" / source)
    assert sha(package / "notes/evidence.json") == sha(args.evidence)
    manuscript = "\n".join(p.read_text(encoding="utf-8") for p in package.glob("sections/*.tex"))
    assert not re.search(r"pending|TODO|TBD|XX\.XX", manuscript, re.I)
    abstract = (package / "sections/abstract.tex").read_text(encoding="utf-8")
    abstract = re.sub(r"\\(?:begin|end)\{abstract\}", "", abstract)
    word_count = len(abstract.split())
    assert 100 <= word_count <= 150, word_count
    table = (package / "tables/training_results.tex").read_text(encoding="utf-8")
    for name in ("official", "sft_seed0", "sft_current_seed0", "no_exchange_seed0"):
        row = evidence["models"][name]
        values = [row["public"]["mIoU"], row["relation"]["mIoU"], row["relation"]["JointPairAcc@0.5"]]
        assert " & ".join(fmt2(x) for x in values) in table, name
    for stage in ("no_exchange", "rbee", "continue_rbee", "setpo"):
        for split in ("relation", "public"):
            row = evidence["stage_means"][stage][split]["mIoU"]
            assert f"${fmt2(row['mean'])}\\pm{fmt2(row['sample_sd'])}$" in table, (stage, split)
    main_table = (package / "tables/main_results.tex").read_text(encoding="utf-8")
    old_table = (repo / "paper/overleaf_icassp2027_reltwin_review_v3/tables/main_results.tex").read_text(encoding="utf-8")
    for method in ("WTATG", "AM-DETR", "Gemini-2.5-Flash", "Gemini-2.5-Pro", "Kimi-Audio", "Qwen2-Audio", "Audio Flamingo 3", "TimeAudio", "SpotSound-Q", "SpotSound-A"):
        current = next(line for line in main_table.splitlines() if line.startswith(method + " &"))
        previous = next(line for line in old_table.splitlines() if line.startswith(method + " &"))
        assert current.rstrip("\\") == " &".join(previous.split(" &")[:7]), method
    for split, row_name in (("spotsound", "SpotSound"), ("clotho", "Clotho")):
        filename = "spotsound_full400_report.json" if split == "spotsound" else "clotho_full6649_report.json"
        report = json.loads((repo / "results/nova_boundary_utility_v1" / filename).read_text(encoding="utf-8"))
        for key in ("nova_mbr_mIoU_percent", "R1@0.3_percent", "R1@0.5_percent"):
            assert f"{report[key]:.2f}" in main_table, (row_name, key)
    bridge = evidence["comparisons"]["cand_minus_current_sft_seed0"]
    bridge_tex = (package / "sections/bridge_results.tex").read_text(encoding="utf-8")
    for split, metric in (("relation", "mIoU"), ("relation", "JointPairAcc@0.5"), ("public", "mIoU")):
        value = bridge[split]["metrics"][metric]["audio_cluster"]["mean_delta_points"]
        assert f"{value:.2f}" in bridge_tex, (split, metric, value)
    log = (package / "main.log").read_text(encoding="utf-8", errors="replace")
    for bad in ("Overfull ", "undefined references", "Citation `", "LaTeX Error", "multiply defined"):
        assert bad not in log, bad
    pdf = pymupdf.open(package / "main.pdf")
    assert len(pdf) <= 5
    text = "\n".join(page.get_text() for page in pdf)
    assert "REFERENCES" in text
    if len(pdf) == 5:
        assert "REFERENCES" in pdf[4].get_text()[:60]
        assert "ACKNOWLEDGMENTS" not in pdf[4].get_text()
    qa.mkdir(parents=True, exist_ok=True)
    fonts = set()
    for index, page in enumerate(pdf):
        assert tuple(page.rect)[2:] == (612., 792.)
        for font in page.get_fonts(full=True):
            fonts.add(font[3])
            assert font[1] != "n/a" and font[2] != "Type3", font
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(qa / f"page-{index+1}.png")
    archive = None
    if args.zip:
        with zipfile.ZipFile(args.zip) as z:
            names = z.namelist()
            assert "main.tex" in names and "spconf.sty" in names and "authors.tex" in names
            assert not any(name.endswith((".aux", ".log", ".pdf", ".safetensors", ".pyc")) for name in names)
            assert not any(name.startswith("/") or ".." in name.split("/") for name in names)
            for name in names:
                if not name.endswith("/"):
                    assert z.read(name) == (package / name).read_bytes(), name
        archive = {"path": str(args.zip), "sha256": sha(args.zip), "files": len(names)}
    report = {"validated": True, "pages": len(pdf), "abstract_words": word_count, "fonts": sorted(fonts), "pdf_sha256": sha(package / "main.pdf"), "evidence_sha256": sha(args.evidence), "archive": archive,
              "checks": ["15 complete aligned models", "matched-runtime bridge included", "no pending claims", "published main-table transcription", "controlled numerical table", "input/source separation", "template byte identity", "no overfull or unresolved citations", "US Letter and <=5 pages", "fifth page references only", "font embedding and no Type3"],
              "manual_checks_remaining": ["visual inspection of rendered pages", "author approval and ORCID/contact confirmation", "submission portal inspection"]}
    (qa / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
