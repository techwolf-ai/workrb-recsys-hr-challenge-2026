"""matplotlib xkcd-mode diagrams for the baselines notebook.
re-run with `uv run python knowledge_sharing/images/make_baselines_diagrams.py`."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
# 1. four-baselines overview: who pushes on which lever
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    ax.text(6, 6.5, "four baselines, four ways of getting from text to a similarity",
            ha="center", fontsize=11, weight="bold")

    headers = ["text -> vector", "score = ?", "trained on skills?"]
    rows = [
        ("TF-IDF",          "sparse bag of words",   "cosine",                 "no, just counts"),
        ("MPNet (off the shelf)", "dense, mean-pooled", "cosine",            "no, generic web text"),
        ("ConTeXT-Match",   "dense, per token",      "attention-weighted cos", "yes, ESCO synthetic"),
        ("CurriculumMatch", "dense, mean-pooled",    "cosine",                 "yes, definitions then ESCO synthetic"),
    ]

    col_x = [0.4, 3.1, 6.0, 8.6]
    col_w = [2.6, 2.8, 2.5, 3.3]

    # header row
    y_h = 5.4
    box(ax, col_x[0], y_h, col_w[0], 0.7, "model", fc=PEACH, fontsize=9.5)
    for i, h in enumerate(headers):
        box(ax, col_x[i+1], y_h, col_w[i+1], 0.7, h, fc=PEACH, fontsize=9.5)

    # data rows
    fcs = [PAPER, PAPER, SAGE, SAGE]
    for ri, row in enumerate(rows):
        y = 4.5 - ri * 1.05
        for ci, cell in enumerate(row):
            box(ax, col_x[ci], y, col_w[ci], 0.85, cell, fc=fcs[ri], fontsize=8.5)

    ax.text(6, 0.05,
            "the bottom two are SOTA. they each push on a different lever.",
            ha="center", fontsize=9, color=DRIED, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "baselines_overview.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 2. TF-IDF: shared-vocabulary bag of words, cosine.
#    Both vectors live in the SAME |V|-dim space, with a 0 in every dimension
#    where the string didn't contain that token. The dot product is then just
#    the sum over dimensions where both are non-zero -- here, only "debug".
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.0); ax.axis("off")

    ax.text(6.75, 6.55,
            "TF-IDF: same vocabulary axis for both, cosine over shared columns",
            ha="center", fontsize=11, weight="bold")

    # vocabulary header (shared dimensions)
    vocab = ["refactor", "debug", "rust", "services", "software"]
    # one column per vocabulary entry, two rows: query and skill
    col_x0 = 3.4
    col_w = 1.6
    col_gap = 0.12
    row_h = 0.85

    # row labels
    box(ax, 0.4, 4.1, 2.6, row_h, '"refactor and debug\n the Rust services"',
        fc=PAPER, fontsize=8.5)
    box(ax, 0.4, 2.5, 2.6, row_h, '"debug software"',
        fc=PAPER, fontsize=9)

    # vocab header row
    for i, tok in enumerate(vocab):
        x = col_x0 + i * (col_w + col_gap)
        box(ax, x, 5.2, col_w, row_h, tok, fc=PEACH, fontsize=8.5)

    # query weights: refactor 0.4, debug 0.3, rust 0.5, services 0.4, software 0
    qw = [0.40, 0.30, 0.50, 0.40, 0.00]
    sw = [0.00, 0.60, 0.00, 0.00, 0.80]

    def cell(ax, x, y, w, h, val, hi=False):
        fc = SAGE if hi else (CREAM if val > 0 else PAPER)
        box(ax, x, y, w, h, f"{val:.2f}" if val > 0 else "0",
            fc=fc, fontsize=8.5)

    # query row
    for i, v in enumerate(qw):
        x = col_x0 + i * (col_w + col_gap)
        hi = (vocab[i] == "debug")
        cell(ax, x, 4.1, col_w, row_h, v, hi=hi)

    # skill row
    for i, v in enumerate(sw):
        x = col_x0 + i * (col_w + col_gap)
        hi = (vocab[i] == "debug")
        cell(ax, x, 2.5, col_w, row_h, v, hi=hi)

    # multiply column underneath: dim-by-dim products
    ax.text(2.0, 1.55, "elementwise\nproduct:", ha="center", fontsize=8.5,
            color=DRIED, style="italic")
    for i in range(len(vocab)):
        x = col_x0 + i * (col_w + col_gap)
        prod = qw[i] * sw[i]
        cell(ax, x, 1.2, col_w, row_h, prod, hi=(prod > 0))

    # sum / cosine label
    last_x = col_x0 + (len(vocab) - 1) * (col_w + col_gap) + col_w
    ax.text(last_x + 0.4, 1.6,
            "sum,\nnormalise\n= cosine",
            ha="left", fontsize=9, color=DRIED, style="italic")

    ax.text(6.75, 0.45,
            "every other dimension multiplies to 0. only 'debug' contributes.",
            ha="center", fontsize=9, color=DRIED, style="italic")
    ax.text(6.75, 0.0,
            "no training, no semantics. lowercased, vocabulary fitted on the candidate skills.",
            ha="center", fontsize=8.5, color=DRIED)

    plt.tight_layout()
    plt.savefig(OUT / "tfidf_diagram.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 3. MPNet off-the-shelf: pretrained encoder, mean pool, cosine.
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.0); ax.axis("off")

    ax.text(6, 5.65,
            "MPNet, off the shelf: pretrained, then never touched",
            ha="center", fontsize=11, weight="bold")

    box(ax, 0.3, 3.8, 3.6, 0.9, "sentence", fc=PAPER, fontsize=10)
    box(ax, 0.3, 1.0, 3.6, 0.9, "skill name", fc=PAPER, fontsize=10)

    box(ax, 4.4, 2.1, 3.0, 1.6,
        "all-mpnet-\nbase-v2\n(pretrained)", fc=BLUSH, fontsize=9.5)

    box(ax, 7.9, 3.8, 1.6, 0.9, "mean\nvec", fc=CREAM, fontsize=9.5)
    box(ax, 7.9, 1.0, 1.6, 0.9, "mean\nvec", fc=CREAM, fontsize=9.5)
    box(ax, 10.0, 2.1, 1.7, 1.6, "cosine\nsim", fc=PEACH, fontsize=9.5)

    arrow(ax, 3.9, 4.25, 4.4, 3.4)
    arrow(ax, 3.9, 1.45, 4.4, 2.4)
    arrow(ax, 7.4, 3.4, 7.9, 4.25)
    arrow(ax, 7.4, 2.4, 7.9, 1.45)
    arrow(ax, 9.5, 4.25, 10.0, 3.4)
    arrow(ax, 9.5, 1.45, 10.0, 2.4)

    ax.text(5.9, 0.4,
            "weights frozen at 'pretrained on web text'. no skill data ever shown.",
            ha="center", fontsize=9, color=DRIED, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "mpnet_pretrain.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 4. ConTeXT-Match: per-token attention against the skill vector.
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.5); ax.axis("off")

    ax.text(6, 7.1,
            "ConTeXT-Match: skill picks which tokens to look at",
            ha="center", fontsize=11, weight="bold")

    # left column: per-token sentence embeddings
    tokens = ["refactor", "and", "debug", "the", "Rust", "services"]
    weights = [0.05, 0.02, 0.55, 0.02, 0.30, 0.06]  # illustrative attention
    y0 = 1.4
    h = 0.65
    gap = 0.12
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        y = y0 + i * (h + gap)
        # weight bar (right of token)
        box(ax, 0.3, y, 1.7, h, tok, fc=PAPER, fontsize=9)
        # opacity-mapped fill: heavier = more saturated
        bar_w = 1.4 * w / max(weights)
        box(ax, 2.2, y + 0.05, max(0.18, bar_w), h - 0.1,
            f"{w:.2f}", fc=PEACH if w > 0.1 else CREAM, fontsize=8)

    ax.text(2.0, y0 - 0.55,
            "per-token\n vectors    weight",
            ha="center", fontsize=8.5, color=DRIED, style="italic")

    # right side: skill mean vector
    box(ax, 8.4, 4.5, 3.0, 1.0,
        '"debug software"\n  -> mean vector', fc=PAPER, fontsize=9.5)

    # the attention computation node
    box(ax, 5.0, 4.0, 2.6, 1.6,
        "attention\nweight\n(skill . token)", fc=PEACH, fontsize=9)

    # arrows from tokens to attention node (only for the heaviest)
    arrow(ax, 4.0, y0 + 2 * (h + gap) + h / 2, 5.0, 4.6,
          label="big dot", label_side="above", label_offset=0.1)
    arrow(ax, 8.4, 5.0, 7.6, 4.8)

    # weighted-cosine node and final score
    box(ax, 5.0, 1.8, 2.6, 1.4,
        "weighted\ncosine sum", fc=SAGE, fontsize=9.5)
    arrow(ax, 6.3, 4.0, 6.3, 3.2, label="weights", label_side="right",
          label_offset=0.1)
    box(ax, 8.4, 2.0, 3.0, 1.0, "match score", fc=CREAM, fontsize=9.5)
    arrow(ax, 7.6, 2.5, 8.4, 2.5)

    ax.text(6, 0.3,
            "trained with InfoNCE on (sentence, skill) plus (skill, definition).",
            ha="center", fontsize=9, color=DRIED, style="italic")

    plt.tight_layout()
    plt.savefig(OUT / "contextmatch_score.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# 5. CurriculumMatch: definitions first, then synthetic sentences.
# ============================================================================
with plt.xkcd(scale=0.9, length=80, randomness=2):
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")

    ax.text(6.5, 6.1,
            "CurriculumMatch: easy first, hard second",
            ha="center", fontsize=11, weight="bold")

    # phase 1
    box(ax, 0.3, 3.0, 4.0, 1.6,
        "phase 1: pretrain\non (skill, definition)\n~112K ESCO entries",
        fc=SAGE, fontsize=9)
    # phase 2
    box(ax, 5.2, 3.0, 4.0, 1.6,
        "phase 2: fine-tune\non (synthetic sentence, skill)\n~138K + augmentation",
        fc=PEACH, fontsize=9)
    # output
    box(ax, 10.0, 3.0, 2.7, 1.6,
        "skill retriever\n(mean-pool +\ncosine)",
        fc=CREAM, fontsize=9.5)

    arrow(ax, 4.3, 3.8, 5.2, 3.8,
          label="warm start", label_side="above", label_offset=0.15)
    arrow(ax, 9.2, 3.8, 10.0, 3.8)

    # captions
    ax.text(2.3, 2.4,
            "easy: skill name <-> its definition.\ntwo strings that mean the same thing.",
            ha="center", fontsize=8.5, color=DRIED, style="italic")
    ax.text(7.2, 2.4,
            "harder: same skill, but expressed in\na noisy job-ad sentence.",
            ha="center", fontsize=8.5, color=DRIED, style="italic")

    ax.text(6.5, 0.4,
            "same MPNet backbone, same InfoNCE loss. the curriculum is the contribution.",
            ha="center", fontsize=9, color=DRIED)

    plt.tight_layout()
    plt.savefig(OUT / "curriculum_phases.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close()


print("wrote:")
for f in ["baselines_overview", "tfidf_diagram", "mpnet_pretrain",
          "contextmatch_score", "curriculum_phases"]:
    print(f"  images/{f}.png")
