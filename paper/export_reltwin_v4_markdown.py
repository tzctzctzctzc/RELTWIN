"""Small explicit LaTeX-to-Markdown export for this manuscript; preserves math."""
import argparse
from pathlib import Path
import re


LINKS = {
    "xu2021grounding": ("Text-to-Audio Grounding", "https://arxiv.org/abs/2102.11474"),
    "munakata2025amr": ("Language-based Audio Moment Retrieval", "https://arxiv.org/abs/2409.15672"),
    "sun2026spotsound": ("SpotSound, Table 3", "https://arxiv.org/html/2604.13023v2"),
    "yuan2024tclap": ("T-CLAP", "https://arxiv.org/html/2404.17806v1"),
    "ghosh2024compa": ("CompA", "https://arxiv.org/html/2310.08753v3"),
    "ren2026costala": ("CoSTALA", "https://arxiv.org/html/2608.24374v1"),
    "cheng2024shine": ("SHINE", "https://arxiv.org/abs/2407.05118"),
    "piczak2015esc": ("ESC-50", "https://github.com/karolpiczak/ESC-50"),
    "thrush2022winoground": ("Winoground", "https://arxiv.org/abs/2204.03162"),
}
REFS = {"fig:pair": "1", "tab:main": "1", "tab:training": "2", "tab:controls": "3",
        "eq:labels": "1", "eq:candidate": "3", "eq:loss": "4"}


def prose(text):
    text = re.sub(r"\\cite\{([^}]+)\}", lambda m: ", ".join(f"[{LINKS[k][0]}]({LINKS[k][1]})" for k in m[1].split(",")), text)
    text = re.sub(r"\\ref\{([^}]+)\}", lambda m: REFS[m[1]], text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\section\{([^}]+)\}", r"## \1\n", text)
    text = re.sub(r"\\subsection\{([^}]+)\}", r"### \1\n", text)
    for macro, mark in (("textbf", "**"), ("emph", "*"), ("textit", "*")):
        text = re.sub(r"\\" + macro + r"\{([^{}]+)\}", lambda m: mark + m[1] + mark, text)
    text = text.replace(r"\setiou", r"\operatorname{IoU}_{\mathrm{set}}")
    text = text.replace(r"\JS", r"\operatorname{JS}")
    for env in ("equation", "align"):
        text = text.replace(r"\begin{" + env + "}", "\n$$\n").replace(r"\end{" + env + "}", "\n$$\n")
    text = text.replace(r"\begin{abstract}", "").replace(r"\end{abstract}", "")
    text = text.replace(r"\%", "%").replace(r"\ ", " ").replace(r"\,", " ")
    text = text.replace("~", " ").replace("``", '“').replace("''", '”')
    text = text.replace("SFT--Cand", "SFT–Cand").replace("query--window", "query–window").replace("audio--caption", "audio–caption")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def table(path, kind):
    body = path.read_text(encoding="utf-8")
    if kind == "main":
        headers = ["Method", "SpotSound mIoU", "R1@.3", "R1@.5", "Clotho mIoU", "R1@.3", "R1@.5"]
        start = "WTATG"
    elif kind == "training":
        headers, start = ["Method", "SpotSound mIoU", "Relation mIoU", "JointPairAcc"], "Official"
    else:
        headers, start = ["Contrast", "Metric", "Delta (points)", "95% source-group CI"], "RBEE"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    active = False
    for line in body.splitlines():
        if line.startswith(start):
            active = True
        if active and "&" in line and line.endswith(r"\\"):
            cells = [prose(x.strip()) for x in line[:-2].split("&")]
            assert len(cells) == len(headers)
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("package", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    package = args.package
    def section(name):
        body = (package / f"sections/{name}.tex").read_text(encoding="utf-8")
        body = body.replace(r"\input{sections/bridge_results}", (package / "sections/bridge_results.tex").read_text(encoding="utf-8"))
        return prose(body)
    lines = ["# RelTwin: Learning Query-Specific Windows from Co-Occurring Inverse Relations", "",
        "Zhicheng Tang; Yuehan Zhang — Huazhong University of Science and Technology, Wuhan, China.", "",
        "ICASSP 2027 author-review draft, 2026-09-13. Exported from the English manuscript; tables preserve the same evidence roles. The schematic is described in the methods rather than rasterized here.", "", "## Abstract", "", section("abstract"), "",
        "Keywords: audio temporal grounding; compositional reasoning; hard answer negatives; paired supervision.", "", section("introduction"), "", section("method"), "", section("experiments"), "",
        "### Table 1. Public grounding results (%)", "",
        "Literature values are from SpotSound Table 3, not its ablations. RelTwin rows are three-seed current-harness means. The archived full system is a distinct dataset-specific pipeline; Clotho is official plus refinement, not RelTwin transfer. The final row compares with each metric's prior main-table best. Bold denotes the highest listed point estimate, not significance.", "", table(package / "tables/main_results.tex", "main"), "",
        "### Table 2. Unified controlled evaluation (%)", "", "Current SFT and Cand seed 0 share the training runtime. Lower rows use all three observed seeds; ± is sample standard deviation, not CI. Historical models retain the training-runtime caveat described in the setup.", "", table(package / "tables/training_results.tex", "training"), "",
        "### Table 3. Extension contrasts", "", "Intervals use original-source connected groups. Neither contrast establishes an independent extension benefit.", "", table(package / "tables/controls.tex", "controls"), "", section("results"), "", section("conclusion"), "",
        "## AI assistance disclosure", "", "OpenAI Codex assisted with initial drafting and translation of all sections, the schematic, LaTeX preparation, and analysis code.", "", "## References", ""]
    lines += [f"- [{title}]({url})" for title, url in LINKS.values()]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
