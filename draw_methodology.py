"""
Professional methodology diagram for the Diabetic Retinopathy Detection project.
Clean, research-presentation quality. Dark-header + light body style.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Canvas ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

if True:  # single-indent block to keep diff minimal

    # ── Design constants ─────────────────────────────────────────────────────
    BOX_W    = 3.6
    BOX_H    = 1.85
    CORNER_R = 0.14

    FILL_DARK  = "white"     # row-1 boxes — white fill
    FILL_MID   = "#f0f0f0"   # light grey — row-2 model boxes
    FILL_INFER = "#e0e0e0"   # slightly darker — inference boxes
    TEXT_LIGHT = "#1a1a2e"  # dark text on white row-1
    TEXT_DARK  = "#1a1a2e"
    ARROW_C    = "#1a1a2e"
    LW         = 2.2

    # ── Helpers ──────────────────────────────────────────────────────────────
    def draw_box(cx, cy, title, subtitle, fill=FILL_MID, title_color=TEXT_DARK):
        x, y = cx - BOX_W / 2, cy - BOX_H / 2
        rect = mpatches.FancyBboxPatch(
            (x, y), BOX_W, BOX_H,
            boxstyle=f"round,pad=0,rounding_size={CORNER_R}",
            linewidth=LW, edgecolor=TEXT_DARK,
            facecolor=fill, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.28, title,
                ha="center", va="center",
                fontsize=12.0, fontweight="bold", color=title_color,
                fontfamily="DejaVu Sans", zorder=4, multialignment="center")
        sub_col = "#444444"
        ax.text(cx, cy - 0.32, subtitle,
                ha="center", va="center",
                fontsize=9.5, color=sub_col,
                fontfamily="DejaVu Sans", zorder=4,
                multialignment="center", linespacing=1.5)

    def h_arrow(x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                    lw=3.2, mutation_scale=52), zorder=2)

    def v_arrow(x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                    lw=3.2, mutation_scale=52), zorder=2)

    def elbow(ox, oy, corner_x, corner_y, tx, ty):
        """Draw lines ox,oy → corner → tx,ty with arrowhead at end."""
        ax.plot([ox, corner_x], [oy, corner_y], color=ARROW_C, lw=3.2, zorder=2)
        ax.annotate("", xy=(tx, ty), xytext=(corner_x, corner_y),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_C,
                                    lw=3.2, mutation_scale=52), zorder=2)

    def step_badge(cx, cy, n, fill):
        bx = cx - BOX_W / 2 + 0.26
        by = cy + BOX_H / 2 - 0.26
        sq = mpatches.Rectangle((bx - 0.22, by - 0.22), 0.44, 0.44,
                                  linewidth=1.2, edgecolor="#1a1a2e",
                                  facecolor=fill, zorder=5)
        ax.add_patch(sq)
        ax.text(bx, by, str(n), ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=6)

    # ════════════════════════════════════════════════════════════════════════
    # TITLE
    # ════════════════════════════════════════════════════════════════════════
    ax.text(10, 12.35, "Methodology Diagram",
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            color=TEXT_DARK, fontfamily="DejaVu Sans")

    # ════════════════════════════════════════════════════════════════════════
    # ROW 1 — Data Preparation  (left → right)
    # ════════════════════════════════════════════════════════════════════════
    ROW1_Y = 10.1
    R1_XS  = [2.0, 5.8, 9.6, 13.4]
    ROW1_DATA = [
        ("Data\nInitialization",  "Mount Drive &\nSet Paths"),
        ("Data Loading",          "Load OCT & Fundus\nImages (0/1 Labels)"),
        ("Preprocessing",         "Merge OCT +\nFundus Data"),
        ("Dataset Extraction",    "Extract Deep\nFeatures"),
    ]
    for cx, (t, s) in zip(R1_XS, ROW1_DATA):
        draw_box(cx, ROW1_Y, t, s, fill=FILL_DARK, title_color=TEXT_LIGHT)
    for i in range(len(R1_XS) - 1):
        h_arrow(R1_XS[i] + BOX_W/2, R1_XS[i+1] - BOX_W/2, ROW1_Y)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 2 — Model Building  (right → left, snake)
    # ════════════════════════════════════════════════════════════════════════
    ROW2_Y = 7.0
    R2_XS  = [13.4, 9.6, 5.8, 2.0]   # drawn right-to-left in flow
    ROW2_DATA = [
        ("Model Saving",             "Save as\nbest_dr_model.pkl"),
        ("Evaluation",               "Accuracy, Confusion\nMatrix, Report"),
        ("Train-Test Split",         "80 / 20\n(Stratified)"),
        ("Flatten & Scale\nFeatures","Pixel Flatten +\nStandardScaler"),
    ]
    for cx, (t, s) in zip(R2_XS, ROW2_DATA):
        draw_box(cx, ROW2_Y, t, s, fill=FILL_MID)
    for i in range(len(R2_XS) - 1):
        h_arrow(R2_XS[i] - BOX_W/2, R2_XS[i+1] + BOX_W/2, ROW2_Y)

    # Connector: right edge of Row 1 → right edge of Row 2
    RX = R1_XS[-1] + BOX_W/2 + 0.30
    ax.plot([R1_XS[-1] + BOX_W/2, RX], [ROW1_Y, ROW1_Y], color=ARROW_C, lw=3.2, zorder=2)
    elbow(RX, ROW1_Y, RX, ROW2_Y, R2_XS[0] + BOX_W/2, ROW2_Y)

    # ════════════════════════════════════════════════════════════════════════
    # ROW 3 — Inference Pipeline  (centred, left → right)
    # ════════════════════════════════════════════════════════════════════════
    ROW3_Y = 3.85
    R3_XS  = [4.6, 9.2, 13.8]
    ROW3_DATA = [
        ("User Upload",       "Upload Retinal\nImage (Web UI)"),
        ("Preprocessing",     "Deep Feature\nGeneration"),
        ("Output Generation", "Display Prediction\n& Confidence Score"),
    ]
    for cx, (t, s) in zip(R3_XS, ROW3_DATA):
        draw_box(cx, ROW3_Y, t, s, fill=FILL_INFER)
    for i in range(len(R3_XS) - 1):
        h_arrow(R3_XS[i] + BOX_W/2, R3_XS[i+1] - BOX_W/2, ROW3_Y)

    # Connector: left edge of Row 2 → left edge of Row 3
    LX = R2_XS[-1] - BOX_W/2 - 0.30
    ax.plot([R2_XS[-1] - BOX_W/2, LX], [ROW2_Y, ROW2_Y], color=ARROW_C, lw=3.2, zorder=2)
    elbow(LX, ROW2_Y, LX, ROW3_Y, R3_XS[0] - BOX_W/2, ROW3_Y)

    # ════════════════════════════════════════════════════════════════════════
    # Section labels — horizontal, above each row group
    # ════════════════════════════════════════════════════════════════════════
    # "Training Pipeline" sits between the main title and Row 1
    ax.text((R1_XS[0] + R1_XS[-1]) / 2, ROW1_Y + BOX_H / 2 + 0.55,
            "— Training Pipeline —",
            ha="center", va="center",
            fontsize=12, color="#333333",
            fontfamily="DejaVu Sans", fontweight="bold")

    # "Inference Pipeline" sits between Row 2 and Row 3
    ax.text((R3_XS[0] + R3_XS[-1]) / 2, ROW3_Y + BOX_H / 2 + 0.55,
            "— Inference Pipeline —",
            ha="center", va="center",
            fontsize=12, color="#333333",
            fontfamily="DejaVu Sans", fontweight="bold")

    # ════════════════════════════════════════════════════════════════════════
    # Save
    # ════════════════════════════════════════════════════════════════════════
    plt.tight_layout(pad=0.3)
    out = "/Users/srivardhan/Vaishu Major Project/methodology_diagram.png"
    fig.savefig(out, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved -> {out}")
