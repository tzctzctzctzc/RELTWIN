"""Validate manuscript numbers against frozen evidence; render PDF QA pages.

No model inference, statistical reruns, or edits to historical results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pymupdf


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tex_numbers(line: str) -> list[float]:
    cells = line.split("&")[1:]
    return [float(re.sub(r"\\textbf\{([^}]+)\}", r"\1", c)
                  .strip().rstrip("\\").strip().strip("$")) for c in cells]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("paper/ICASSP_DRAFT_ZH_v6_RELTWIN.md"))
    parser.add_argument("--relation-audit", type=Path)
    args = parser.parse_args()
    package, repo, qa = args.package.resolve(), args.repository.resolve(), args.qa_dir.resolve()
    qa.mkdir(parents=True, exist_ok=True)
    source = (package / "notes/source_zh.md").read_text(encoding="utf-8")
    assert digest(package / "notes/source_zh.md") == digest(repo / args.source)
    abstract = (package / "sections/abstract.tex").read_text(encoding="utf-8")
    abstract = re.sub(r"\\(?:begin|end)\{abstract\}", "", abstract)
    word_count = len(abstract.split())
    assert 100 <= word_count <= 150, word_count
    source_rows = {}
    for row in source.splitlines():
        if not row.startswith("| "):
            continue
        cells = [s.strip() for s in row.strip("|").split("|")]
        if len(cells) != 11:
            continue
        try:
            source_rows[cells[1]] = [float(s.replace("*", "").replace("−", "-")) for s in cells[2:]]
        except ValueError:
            continue
    checked = []
    table = (package / "tables/main_results.tex").read_text(encoding="utf-8")
    for line in table.splitlines():
        if "&" not in line or not line.endswith(r"\\"):
            continue
        label = line.split("&")[0].strip()
        key = "已评测完整系统" if label == "Audited full pipeline" else label
        if label.startswith(r"$\Delta$"):
            key = "相较此前逐项最佳（百分点）"
        if key in source_rows:
            assert tex_numbers(line) == source_rows[key], (key, tex_numbers(line), source_rows[key])
            checked.append(key)
    assert len(checked) == 12, checked

    files = ["results/e002/rbee_vs_official.json", "results/e002/setpo_vs_rbee.json",
             "results/e002/reltwin_setpo_vs_rbee.json", "results/e002/sota_decision.json",
             "results/controlled/sft_seed0/public_summary.json",
             "results/controlled/sft_seed0/train_summary.json",
             "results/rbee_seed_0/train_summary.json",
             "results/nova_boundary_utility_v1/spotsound_full400_report.json",
             "results/nova_boundary_utility_v1/clotho_full6649_report.json",
             "results/official_protocol_rerun_20260907/summary.json"]
    rbee, step, relation, decision, sft = [read_json(repo / f) for f in files[:5]]
    sft_train, rbee_train = [read_json(repo / f) for f in files[5:7]]
    for key in ("steps", "learning_rate", "seed", "relation_pairs", "rehearsal_examples", "trainable_parameters"):
        assert sft_train[key] == rbee_train[key], key
    assert sft_train["mode"] == "sft" and rbee_train["mode"] == "rbee"
    metrics = rbee["metrics_fraction"]["mIoU"]
    rbee_seeds = [v * 100 for v in metrics["method_per_seed"]]
    setpo_seeds = [v + d * 100 for v, d in zip(rbee_seeds, step["delta_target_minus_source"]["mIoU"]["per_seed_delta"])]
    stages_tex = (package / "tables/training_results.tex").read_text(encoding="utf-8")
    for values in (rbee_seeds, setpo_seeds):
        token = f"{statistics.mean(values):.3f}\\pm{statistics.stdev(values):.3f}"
        assert token in stages_tex, token
    assert f'{sft["mIoU"]:.3f}' in stages_tex
    assert f'{metrics["baseline"]*100:.3f}' in stages_tex
    for key in ("rbee_three_seed_mean", "setpo_three_seed_mean"):
        for metric in ("R1@0.3_percent", "R1@0.5_percent"):
            assert f'{decision[key][metric]:.3f}' in stages_tex
    relation_tex = (package / "tables/relation_results.tex").read_text(encoding="utf-8")
    relation_values = {}
    relation_ci_scope = "Historical fixed-seed stratified query-row/pair bootstrap, NOT audio-cluster bootstrap."
    if args.relation_audit:
        audit_path = repo / args.relation_audit
        audit = read_json(audit_path)
        assert digest(audit_path) == digest(package / "notes/relation_audit.json")
        files.append(str(args.relation_audit))
        assert audit["audio_groups"] == 80 and audit["source_connected_groups"] == 39
        assert not audit["train_development_source_recording_overlap"]
        keys = ("mIoU", "PairAcc@0.5", "SwapError", "JointPairAcc@0.5")
        def vector(name):
            return [audit["models"][name]["metrics_percent"][k] for k in keys]
        def mean_stage(stage):
            return [statistics.mean(values) for values in zip(*(vector(f"cached_{stage}_seed{s}") for s in range(3)))]
        sft0, rbee0 = vector("cached_sft_seed0"), vector("cached_rbee_seed0")
        rmean, smean = mean_stage("rbee"), mean_stage("setpo")
        expected = {"SFT, seed 0": sft0, "RBEE, seed 0": rbee0,
                    r"$\Delta_0$": [b-a for a, b in zip(sft0, rbee0)],
                    "RBEE, 3 seeds": rmean, r"RBEE $\to$ SetPO": smean,
                    r"$\Delta$": [b-a for a, b in zip(rmean, smean)]}
        verified = set()
        for line in relation_tex.splitlines():
            label = line.split("&")[0].strip()
            if label in expected:
                # Decimal half-up is the displayed convention at exact .005 ties.
                rounded = [float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) for v in expected[label]]
                assert tex_numbers(line) == rounded, (label, tex_numbers(line), rounded)
                verified.add(label)
        assert verified == set(expected)
        relation_values = expected
        relation_ci_scope = audit["seed_resampling"] + " Audio/source-component paired bootstrap; post-hoc development data."
        discussion = (package / "sections/discussion.tex").read_text(encoding="utf-8")
        for name in ("cached_RBEE_minus_SFT_seed0", "cached_SetPO_minus_RBEE"):
            for grouping in ("audio_cluster", "source_connected_cluster"):
                ci = audit["comparisons"][name]["metrics"]["mIoU"][grouping]["ci95_points"]
                assert f"[{ci[0]:.2f},{ci[1]:.2f}]" in discussion, (name, grouping, ci)
    else:
        for key in ("query_mIoU", "pair_acc_0.5", "swap_error_rate"):
            values = relation["aggregate"][key]
            relation_values[key] = {k: values[k]*100 for k in ("source_mean", "target_mean", "mean_delta")}
            for field, value in relation_values[key].items():
                assert f"{value:.2f}" in relation_tex, (key, field, value)
    for key, filename in (("spotsound", files[7]), ("clotho", files[8])):
        full = read_json(repo / filename)
        assert f'{full["nova_mbr_mIoU_percent"]:.2f}' in table, key

    evidence = {
        "source_hashes": {name: digest(repo / name) for name in files},
        "methods_hashes": {str(p.relative_to(repo)): digest(p) for p in
                           [repo / "scripts" / f for f in ("train_reltwin_micro.py", "train_reltwin_setpo.py",
                            "setpo_objective.py", "setpo_candidates.py", "build_esc50_reltwin.py", "interval_metrics.py")]},
        "rbee_miou_seeds_percent": rbee_seeds, "setpo_miou_seeds_percent": setpo_seeds,
        "rbee_minus_official_points": metrics["mean_delta"] * 100,
        "sft_seed0_miou_percent": sft["mIoU"], "matched_seed0_delta_points": rbee_seeds[0]-sft["mIoU"],
        "relation_values_percent": relation_values,
        "relation_ci_scope": relation_ci_scope,
        "new_experiments_in_reported_results": False,
        "post_hoc_cached_relation_reanalysis": bool(args.relation_audit),
        "training_controls_status": "pending; excluded from manuscript results" if args.relation_audit else "not launched in this revision",
    }

    doc = pymupdf.open(package / "main.pdf")
    assert len(doc) == 5, f"Expected four technical pages plus references; got {len(doc)}"
    pages = []
    fonts = {}
    for i, page in enumerate(doc, 1):
        text = page.get_text()
        spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
                 for line in b["lines"] for s in line["spans"] if s["text"].strip()]
        bbox = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
                max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
        assert abs(page.rect.width - 612) < 1 and abs(page.rect.height - 792) < 1
        assert bbox[0] >= 52 and bbox[2] <= 562 and bbox[1] >= 65 and bbox[3] <= 727, (i, bbox)
        for font in page.get_fonts():
            name, ext, kind, data = doc.extract_font(font[0])
            assert data and kind != "Type3", (name, kind)
            fonts[name] = {"type": kind, "embedded": bool(data)}
        pages.append({"page": i, "text_bbox": bbox, "words": len(text.split()),
                      "text_start": text[:120], "text_end": text[-120:]})
        page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(qa / f"page-{i}.png")
    assert "REFERENCES" in doc[4].get_text()
    assert "ACKNOWLEDGMENTS" in doc[3].get_text()
    assert not any(label in doc[4].get_text() for label in ("6. ACKNOWLEDGMENTS", "5. CONCLUSION"))
    log = (package / "main.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull \\hbox" not in log
    assert "undefined references" not in log and "undefined on input line" not in log
    styles = {f: digest(package / f) for f in ("spconf.sty", "IEEEbib.bst")}
    for filename, value in styles.items():
        assert value == digest(repo / "paper/overleaf_icassp2027" / filename)
    report = {"abstract_words": word_count, "page_count": len(doc), "technical_pages": 4,
              "main_table_rows_verified": checked, "training_and_relation_tables_verified": True,
              "style_hashes": styles, "fonts": fonts, "pages": pages,
              "pdf_sha256": digest(package / "main.pdf")}
    for filename, data in (("evidence_summary.json", evidence), ("build_validation.json", report)):
        (package / "notes" / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("abstract_words", "page_count", "technical_pages", "pdf_sha256")}, indent=2))


if __name__ == "__main__":
    main()
