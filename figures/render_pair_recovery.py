from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


TOTAL = 160
COUNTS = {
    "correct_to_correct": 84,
    "correct_to_incorrect": 1,
    "incorrect_to_correct": 51,
    "incorrect_to_incorrect": 24,
}

COLORS = {
    "correct": "#2B7A78",
    "incorrect": "#A8B0BA",
    "recovered": "#31A36D",
    "regressed": "#D95F59",
    "ink": "#252A31",
    "muted": "#5F6873",
    "guide": "#D8DDE3",
}


def ribbon(ax, x0, x1, left, right, color, alpha=0.78, zorder=1):
    """Draw a cubic alluvial ribbon between two vertical intervals."""
    l0, l1 = left
    r0, r1 = right
    bend = 0.42 * (x1 - x0)
    verts = [
        (x0, l0),
        (x0 + bend, l0),
        (x1 - bend, r0),
        (x1, r0),
        (x1, r1),
        (x1 - bend, r1),
        (x0 + bend, l1),
        (x0, l1),
        (x0, l0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
    )


def main():
    assert sum(COUNTS.values()) == TOTAL
    assert COUNTS["correct_to_correct"] + COUNTS["correct_to_incorrect"] == 85
    assert COUNTS["correct_to_correct"] + COUNTS["incorrect_to_correct"] == 135

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
            "font.size": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.42))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_left, x_right, node_w = 0.17, 0.83, 0.055
    y_bottom, y_top, gap = 0.19, 0.91, 0.035
    usable = y_top - y_bottom - gap
    scale = usable / TOTAL

    sft_correct = COUNTS["correct_to_correct"] + COUNTS["correct_to_incorrect"]
    rel_correct = COUNTS["correct_to_correct"] + COUNTS["incorrect_to_correct"]
    left_correct = (y_top - sft_correct * scale, y_top)
    left_incorrect = (y_bottom, y_bottom + (TOTAL - sft_correct) * scale)
    right_correct = (y_top - rel_correct * scale, y_top)
    right_incorrect = (y_bottom, y_bottom + (TOTAL - rel_correct) * scale)

    # Sub-intervals preserve the exact 2x2 paired transition counts.
    l_cc = (left_correct[1] - 84 * scale, left_correct[1])
    l_ci = (left_correct[0], left_correct[0] + 1 * scale)
    l_ic = (left_incorrect[1] - 51 * scale, left_incorrect[1])
    l_ii = (left_incorrect[0], left_incorrect[0] + 24 * scale)

    r_cc = (right_correct[1] - 84 * scale, right_correct[1])
    r_ic = (right_correct[0], right_correct[0] + 51 * scale)
    r_ci = (right_incorrect[1] - 1 * scale, right_incorrect[1])
    r_ii = (right_incorrect[0], right_incorrect[0] + 24 * scale)

    ribbon(ax, x_left + node_w, x_right, l_cc, r_cc, COLORS["correct"], 0.60, 1)
    ribbon(ax, x_left + node_w, x_right, l_ii, r_ii, COLORS["incorrect"], 0.72, 1)
    ribbon(ax, x_left + node_w, x_right, l_ic, r_ic, COLORS["recovered"], 0.86, 2)
    ribbon(ax, x_left + node_w, x_right, l_ci, r_ci, COLORS["regressed"], 0.95, 3)

    for x, interval, color in [
        (x_left, left_correct, COLORS["correct"]),
        (x_left, left_incorrect, COLORS["incorrect"]),
        (x_right, right_correct, COLORS["correct"]),
        (x_right, right_incorrect, COLORS["incorrect"]),
    ]:
        ax.add_patch(
            Rectangle(
                (x, interval[0]),
                node_w,
                interval[1] - interval[0],
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
            )
        )

    ax.text(x_left + node_w / 2, 0.965, "SFT", ha="center", va="center", fontsize=9, weight="bold")
    ax.text(x_right + node_w / 2, 0.965, "RelTwin", ha="center", va="center", fontsize=9, weight="bold")

    label_x_left = x_left - 0.018
    label_x_right = x_right + node_w + 0.018
    ax.text(label_x_left, sum(left_correct) / 2, "Correct\n85 (53.1%)", ha="right", va="center", color=COLORS["ink"])
    ax.text(label_x_left, sum(left_incorrect) / 2, "Incorrect\n75 (46.9%)", ha="right", va="center", color=COLORS["ink"])
    ax.text(label_x_right, sum(right_correct) / 2, "Correct\n135 (84.4%)", ha="left", va="center", color=COLORS["ink"])
    ax.text(label_x_right, sum(right_incorrect) / 2, "Incorrect\n25 (15.6%)", ha="left", va="center", color=COLORS["ink"])

    ax.text(0.50, 0.455, "51 recovered", ha="center", va="center", color="white", fontsize=8.2, weight="bold", zorder=5)
    ax.text(0.50, 0.805, "84 remain correct", ha="center", va="center", color=COLORS["ink"], fontsize=7.4, zorder=5)
    ax.text(0.50, 0.255, "24 remain incorrect", ha="center", va="center", color=COLORS["ink"], fontsize=7.4, zorder=5)
    ax.annotate(
        "1 regressed",
        xy=(0.62, 0.395),
        xytext=(0.62, 0.585),
        ha="center",
        va="center",
        fontsize=7.2,
        color=COLORS["regressed"],
        arrowprops={"arrowstyle": "-", "color": COLORS["regressed"], "lw": 0.8},
        zorder=6,
    )

    ax.plot([0.06, 0.94], [0.13, 0.13], color=COLORS["guide"], lw=0.7)
    ax.text(
        0.50,
        0.075,
        r"Same-window collapse for inverse queries:  $37 \;\rightarrow\; 1$",
        ha="center",
        va="center",
        fontsize=7.7,
        color=COLORS["ink"],
    )

    out_dir = Path(__file__).resolve().parent
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 450} if suffix == "png" else {}
        fig.savefig(out_dir / f"pair_recovery.{suffix}", bbox_inches="tight", pad_inches=0.015, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
