"""Render the manuscript's controlled-results table from audited numerical evidence."""
import argparse
import json
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


def fmt2(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text(encoding="utf-8"))
    if not report["bridge_complete"]:
        raise ValueError("Do not publish a pending runtime bridge")
    lines = [r"\begin{table}[t]", r"\centering",
        r"\caption{Unified evaluation (\%). Upper rows include the matched-runtime SFT--Cand seed~0 comparison. Lower rows report all three fixed seeds; $\pm$ is sample SD for mIoU. Joint denotes JointPairAcc. Historical SFT is not the primary matched-runtime control.}",
        r"\label{tab:training}", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r" & SpotSound & \multicolumn{2}{c}{Relation development}\\",
        r"Method & mIoU & mIoU & Joint\\", r"\midrule"]
    for name, label in [("official", "Official"), ("sft_seed0", "Historical SFT, s0"),
                        ("sft_current_seed0", "Current SFT, s0"), ("no_exchange_seed0", "Cand, s0")]:
        m = report["models"][name]
        lines.append(f"{label} & {fmt2(m['public']['mIoU'])} & {fmt2(m['relation']['mIoU'])} & {fmt2(m['relation']['JointPairAcc@0.5'])}" + r"\\")
    lines.append(r"\midrule")
    for name, label in [("no_exchange", "Cand, 3 seeds"), ("rbee", "RBEE, 3 seeds"),
                        ("continue_rbee", "RBEE +64 updates"), ("setpo", "SetPO +64 updates")]:
        m = report["stage_means"][name]
        a, b, joint = m["public"]["mIoU"], m["relation"]["mIoU"], m["relation"]["JointPairAcc@0.5"]["mean"]
        lines.append(f"{label} & ${fmt2(a['mean'])}\\pm{fmt2(a['sample_sd'])}$ & ${fmt2(b['mean'])}\\pm{fmt2(b['sample_sd'])}$ & {fmt2(joint)}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = args.package / "tables/training_results.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
