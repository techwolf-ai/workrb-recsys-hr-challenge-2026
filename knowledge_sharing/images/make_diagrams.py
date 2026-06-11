"""matplotlib xkcd-mode diagrams for the get_started notebook.
re-run with `uv run python knowledge_sharing/images/make_diagrams.py`."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pathlib import Path

OUT = Path(__file__).parent

CREAM = "#fefae0"
SAGE  = "#cdebc5"
PEACH = "#fbe7c6"
BLUSH = "#f0d4d4"
PAPER = "#f6f4ec"
INK   = "#3a2e2c"
DRIED = "#7a4a3b"


def box(ax, x, y, w, h, text, fc=CREAM, ec=INK, fontsize=11):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=2.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def diamond(ax, cx, cy, w, h, text, fc=PEACH, ec=INK, fontsize=10):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy),
           (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=2.0,
                         edgecolor=ec, facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, label=None, label_offset=0.18,
          label_side="above", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=INK,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if label_side == "above":
            ax.text(mx, my + label_offset, label, ha="center", va="bottom",
                    fontsize=8.5, style="italic")
        elif label_side == "below":
            ax.text(mx, my - label_offset, label, ha="center", va="top",
                    fontsize=8.5, style="italic")
        elif label_side == "left":
            ax.text(mx - label_offset, my, label, ha="right", va="center",
                    fontsize=8.5, style="italic")
        else:
            ax.text(mx + label_offset, my, label, ha="left", va="center",
                    fontsize=8.5, style="italic")


# ============================================================================
# 1. bi-encoder pipeline (compact)
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    box(ax, 0.2, 3.4, 2.0, 0.9, "sentence", fontsize=10)
    box(ax, 0.2, 0.5, 2.0, 0.9, "skill name", fontsize=10)
    box(ax, 3.2, 2.0, 2.2, 1.2, "shared\nencoder", fc=SAGE, fontsize=10)
    box(ax, 6.2, 3.4, 1.4, 0.9, "vec_s", fontsize=10)
    box(ax, 6.2, 0.5, 1.4, 0.9, "vec_k", fontsize=10)
    box(ax, 8.4, 2.0, 2.3, 1.2, "cosine sim\n(B x B)", fc=PEACH, fontsize=10)

    arrow(ax, 2.2, 3.85, 3.2, 2.9)
    arrow(ax, 2.2, 0.95, 3.2, 2.3)
    arrow(ax, 5.4, 2.9, 6.2, 3.85)
    arrow(ax, 5.4, 2.3, 6.2, 0.95)
    arrow(ax, 7.6, 3.85, 8.4, 2.9)
    arrow(ax, 7.6, 0.95, 8.4, 2.3)

    ax.text(5.45, 4.7, "tied weights: same encoder runs twice",
            ha="center", fontsize=9, color=DRIED, style="italic")
    ax.text(9.55, 0.55,
            "diagonal = positives,\noff-diag = free negatives",
            ha="center", fontsize=8, color=DRIED)

    plt.tight_layout()
    plt.savefig(OUT / "biencoder_pipeline.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 2. lexical shortcut (compact)
#    sentence: "the engineer will refactor and debug the Rust services
#               this quarter"
#    word "rust" matches the real ESCO skill 'remove rust from motor
#    vehicles'; gold is the real ESCO skill 'debug software'.
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")

    ax.text(6, 6.2,
            "the model chases shared words, not meaning",
            ha="center", fontsize=11, weight="bold")

    # the sentence (centered top)
    box(ax, 1.4, 4.6, 9.2, 1.0,
        '"the engineer will refactor and debug\nthe Rust services this quarter"',
        fc=PAPER, fontsize=9.5)

    # left branch: model's top pick (wrong)
    box(ax, 0.2, 1.8, 5.0, 1.4,
        "model's top pick:\n   remove rust from motor vehicles   \n(real ESCO car-bodywork skill)",
        fc=BLUSH, fontsize=9.5)
    arrow(ax, 4.2, 4.6, 2.7, 3.2,
          label="overlap on 'rust'",
          label_side="left", label_offset=0.1)

    # right branch: actual gold
    box(ax, 6.8, 1.8, 5.0, 1.4,
        "what we actually want:\n   debug software   \n(real ESCO software skill)",
        fc=SAGE, fontsize=9.5)
    arrow(ax, 7.8, 4.6, 9.3, 3.2,
          label="needs context", label_side="right", label_offset=0.1)

    ax.text(2.7, 0.8, "the wrong answer wins\non word match alone.",
            ha="center", fontsize=9, color=DRIED, style="italic")
    ax.text(9.3, 0.8, "the right answer needs\nsemantic understanding.",
            ha="center", fontsize=9, color=DRIED, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "lexical_shortcut.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 3. cascade judge (compact)
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 14); ax.axis("off")

    Q_FC = PEACH
    cx = 4.5
    leaf_x = 7.6
    yes_back_x = 1.0

    ax.text(5.5, 13.6,
            "cascade judge: how a (query, predicted skill) pair gets a label",
            ha="center", fontsize=10.5, weight="bold")

    # input
    box(ax, cx - 1.6, 12.4, 3.2, 0.7,
        "input:  query x skill", fc=PAPER, fontsize=9)

    y_q1 = 11.2
    y_q2 = 9.6
    y_q3 = 8.0
    y_q4 = 6.4
    y_q5 = 4.8
    y_leaf = 2.8

    diamond_w, diamond_h = 2.8, 1.2

    diamond(ax, cx, y_q1, diamond_w, diamond_h, "human\nlabelled?", fc=Q_FC, fontsize=9)
    diamond(ax, cx, y_q2, diamond_w, diamond_h, "domain\ncorrect?", fc=Q_FC, fontsize=9)
    diamond(ax, cx, y_q3, diamond_w, diamond_h, "activity\ncorrect?", fc=Q_FC, fontsize=9)
    diamond(ax, cx, y_q4, diamond_w, diamond_h, "granularity\ncorrect?", fc=Q_FC, fontsize=9)
    diamond(ax, cx, y_q5, diamond_w, diamond_h,
            "direct\nreplacement\nof a positive?", fc=Q_FC, fontsize=8)

    arrow(ax, cx, 12.4, cx, y_q1 + diamond_h / 2)
    arrow(ax, cx, y_q1 - diamond_h / 2, cx, y_q2 + diamond_h / 2,
          label="no", label_side="right", label_offset=0.12)
    arrow(ax, cx, y_q2 - diamond_h / 2, cx, y_q3 + diamond_h / 2,
          label="yes", label_side="right", label_offset=0.12)
    arrow(ax, cx, y_q3 - diamond_h / 2, cx, y_q4 + diamond_h / 2,
          label="yes", label_side="right", label_offset=0.12)
    arrow(ax, cx, y_q4 - diamond_h / 2, cx, y_q5 + diamond_h / 2,
          label="yes", label_side="right", label_offset=0.12)

    box(ax, leaf_x, y_q5 - 0.4, 3.2, 0.8,
        "3:  LLM correct", fc=SAGE, fontsize=9)
    arrow(ax, cx + diamond_w / 2, y_q5, leaf_x, y_q5,
          label="no", label_side="above", label_offset=0.15)

    box(ax, leaf_x, y_q4 - 0.4, 3.2, 0.8,
        "2:  recommended but\n     not core to query", fc=BLUSH, fontsize=8.5)
    arrow(ax, cx + diamond_w / 2, y_q4, leaf_x, y_q4,
          label="no", label_side="above", label_offset=0.15)

    box(ax, leaf_x, y_q3 - 0.4, 3.2, 0.8,
        "1:  plausible but\n     not mentioned", fc=BLUSH, fontsize=8.5)
    arrow(ax, cx + diamond_w / 2, y_q3, leaf_x, y_q3,
          label="no", label_side="above", label_offset=0.15)

    box(ax, leaf_x, y_q2 - 0.4, 3.2, 0.8,
        "0:  nonsense", fc=BLUSH, fontsize=9)
    arrow(ax, cx + diamond_w / 2, y_q2, leaf_x, y_q2,
          label="no", label_side="above", label_offset=0.15)

    box(ax, cx - 1.6, y_leaf - 0.4, 3.2, 0.8,
        "4:  human correct", fc=SAGE, fontsize=9)

    arrow(ax, cx, y_q5 - diamond_h / 2, cx, y_leaf + 0.4,
          label="yes", label_side="right", label_offset=0.12)

    arrow(ax, cx - diamond_w / 2, y_q1, yes_back_x, y_q1,
          label="yes", label_side="above", label_offset=0.15)
    a = FancyArrowPatch(
        (yes_back_x, y_q1), (yes_back_x, y_leaf),
        arrowstyle="-", linewidth=1.4, color=INK,
    )
    ax.add_patch(a)
    arrow(ax, yes_back_x, y_leaf, cx - 1.6, y_leaf)

    plt.tight_layout()
    plt.savefig(OUT / "cascade_judge.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 4. metrics walkthroughs (compact)
# ============================================================================
def draw_ranks(ax, n=8, gold=None, fontsize=8, y=2.0, h=0.9, x0=0.6, w=1.05,
               gap=0.08):
    if gold is None:
        gold = set()
    for i in range(n):
        x = x0 + i * (w + gap)
        is_rel = (i + 1) in gold
        fc = SAGE if is_rel else PAPER
        box(ax, x, y, w, h, f"rank\n{i+1}", fc=fc, fontsize=fontsize)


with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.0))

    GOLD = {1, 4}
    x0, step = 0.6, 1.13

    ax = axes[0]
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(5.5, 4.0, "MAP: precision at every gold hit, then averaged",
            ha="center", fontsize=10, weight="bold")
    draw_ranks(ax, gold=GOLD, y=2.0, h=0.9)
    ax.annotate("P@1 = 1/1", xy=(x0 + 0 * step + 0.5, 2.0),
                xytext=(x0 + 0 * step + 0.5, 1.0),
                ha="center", fontsize=8.5, color=DRIED,
                arrowprops=dict(arrowstyle="->", color=DRIED, lw=1.0))
    ax.annotate("P@4 = 2/4", xy=(x0 + 3 * step + 0.5, 2.0),
                xytext=(x0 + 3 * step + 0.5, 1.0),
                ha="center", fontsize=8.5, color=DRIED,
                arrowprops=dict(arrowstyle="->", color=DRIED, lw=1.0))
    ax.text(5.5, 0.2, "AP  =  (1.0 + 0.5) / 2  =  0.75",
            ha="center", fontsize=9)

    ax = axes[1]
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(5.5, 4.0, "MRR: only the first gold hit counts",
            ha="center", fontsize=10, weight="bold")
    draw_ranks(ax, gold=GOLD, y=2.0, h=0.9)
    ax.annotate("first hit", xy=(x0 + 0 * step + 0.5, 2.0),
                xytext=(x0 + 0 * step + 0.5, 1.0),
                ha="center", fontsize=8.5, color=DRIED,
                arrowprops=dict(arrowstyle="->", color=DRIED, lw=1.0))
    ax.text(x0 + 3 * step + 0.5, 1.2, "(ignored)",
            ha="center", fontsize=8.5, color="#999")
    ax.text(5.5, 0.2, "MRR  =  1 / 1  =  1.0",
            ha="center", fontsize=9)

    ax = axes[2]
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")
    ax.text(5.5, 4.0, "RP@K: count gold hits in the top K",
            ha="center", fontsize=10, weight="bold")
    draw_ranks(ax, gold=GOLD, y=2.0, h=0.9)

    bracket_x_l = x0
    bracket_x_r = x0 + 4 * step - 0.05
    ax.plot([bracket_x_l, bracket_x_l, bracket_x_r, bracket_x_r],
            [3.1, 3.25, 3.25, 3.1], color=DRIED, linewidth=1.2)
    ax.text((bracket_x_l + bracket_x_r) / 2, 3.4, "top-4",
            ha="center", fontsize=8.5, color=DRIED, style="italic")

    ax.text(5.5, 1.2,
            "2 hits in top-4,  R = 2  ->  RP@4  =  2 / min(4, 2)  =  1.0",
            ha="center", fontsize=9)
    ax.text(5.5, 0.2,
            "(WorkRB uses K = 10 by default)",
            ha="center", fontsize=8.5, color=DRIED, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "metrics_walkthrough.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


print("wrote:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.name)
