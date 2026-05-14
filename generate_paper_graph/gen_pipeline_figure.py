"""
Generate pipeline flowchart for the paper (Figure 1).
Outputs: ../tex_workspace/figures/fig_pipeline.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT = "../tex_workspace/figures/fig_pipeline.pdf"

C_INPUT = "#4A90D9"
C_SEG   = "#E67E22"
C_FLOW  = "#27AE60"
C_FUSE  = "#8E44AD"
C_INP   = "#C0392B"
C_OUT   = "#2C3E50"
C_VGGT  = "#16A085"
C_SAM3  = "#D35400"
C_BG    = "#F8F9FA"

BW, BH = 1.45, 0.46   # box width / height
GAP    = 0.18          # gap between box edge and arrow tip

def box(ax, cx, cy, label, sub="", color=C_SEG, fs=7.8):
    rect = FancyBboxPatch((cx - BW/2, cy - BH/2), BW, BH,
                          boxstyle="round,pad=0.025",
                          facecolor=color, edgecolor="white",
                          linewidth=1.4, alpha=0.93, zorder=3)
    ax.add_patch(rect)
    dy = 0.07 if sub else 0
    ax.text(cx, cy + dy, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="white", zorder=4)
    if sub:
        ax.text(cx, cy - 0.10, sub, ha="center", va="center",
                fontsize=6.2, color="white", alpha=0.88, zorder=4)

def arrow(ax, x0, y, x1, color="#666666"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.3, mutation_scale=11), zorder=2)

def varrow(ax, x, y0, y1, color="#666666", dashed=False):
    ls = "dashed" if dashed else "solid"
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                linestyle=ls, mutation_scale=10), zorder=2)

def dot(ax, x, y, color):
    ax.plot(x, y, "o", ms=6, color=color, zorder=5)

fig, ax = plt.subplots(figsize=(13, 5.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.8)
ax.axis("off")
ax.set_facecolor(C_BG); fig.patch.set_facecolor(C_BG)

# ── Row y-positions ──────────────────────────────────────────────────────────
YA, YB, YE, YF = 4.6, 3.3, 2.1, 0.85
YMID = (YA + YF) / 2   # 2.725

# ── Input Video (vertically centred, left side) ──────────────────────────────
IX = 0.85
box(ax, IX, YMID, "Input\nVideo", "", C_INPUT, fs=8)

# vertical spine from Input Video to each row
SPINE_X = IX + BW/2 + 0.05
for y in (YA, YB, YE, YF):
    ax.plot([SPINE_X, SPINE_X], [YMID, y], color="#AAAAAA", lw=1.2,
            linestyle="dotted", zorder=1)
    arrow(ax, SPINE_X, y, SPINE_X + GAP + 0.05)

# ── Route labels (circle bullets) ────────────────────────────────────────────
LABEL_X = SPINE_X + GAP + 0.18
for lbl, y, col in [("A", YA, C_SEG), ("B", YB, C_SEG),
                     ("E", YE, C_SAM3), ("F", YF, C_VGGT)]:
    ax.text(LABEL_X, y, lbl, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.13", fc=col, ec="white", lw=1.1))

# ── Box x-positions (shared grid) ────────────────────────────────────────────
X1 = 2.55   # first box centre
STEP = BW + 0.38

def xs(n): return X1 + n * STEP   # nth box centre

# ── Route A ──────────────────────────────────────────────────────────────────
box(ax, xs(0), YA, "Mask R-CNN",    "instance seg.",      C_SEG)
arrow(ax, xs(0)+BW/2+GAP, YA, xs(1)-BW/2-GAP)
box(ax, xs(1), YA, "Optical Flow",  "LK + threshold τ",   C_FLOW)
arrow(ax, xs(1)+BW/2+GAP, YA, xs(2)-BW/2-GAP)
box(ax, xs(2), YA, "cv2.inpaint",   "Telea + temp. borrow", C_INP)
arrow(ax, xs(2)+BW/2+GAP, YA, xs(3)-BW/2-GAP)
box(ax, xs(3), YA, "Output A",      "MS: 0.7923",         C_OUT)

# ── Route B ──────────────────────────────────────────────────────────────────
box(ax, xs(0), YB, "Track Anything","SAM + XMem bidir.",   C_SEG)
arrow(ax, xs(0)+BW/2+GAP, YB, xs(1)-BW/2-GAP)
box(ax, xs(1), YB, "ProPainter",    "dual-domain prop.",   C_INP)
arrow(ax, xs(1)+BW/2+GAP, YB, xs(2)-BW/2-GAP)
box(ax, xs(2), YB, "Output B",      "MS: 0.8091",         C_OUT)

# ── Route E (branch from Track Anything output) ───────────────────────────────
# dashed vertical from Track Anything box bottom → E row
varrow(ax, xs(0), YB - BH/2 - GAP, YE + BH/2 + GAP, color=C_SAM3, dashed=True)
dot(ax, xs(0), YB - BH/2 - GAP, C_SAM3)

box(ax, xs(0), YE, "SAM 3 Refine",  "point prompts bidir.", C_SAM3)
arrow(ax, xs(0)+BW/2+GAP, YE, xs(1)-BW/2-GAP)
box(ax, xs(1), YE, "ProPainter",    "dual-domain prop.",   C_INP)
arrow(ax, xs(1)+BW/2+GAP, YE, xs(2)-BW/2-GAP)
box(ax, xs(2), YE, "Output E",      "MS: 0.8254",         C_OUT)

# ── Route F / Phase 6 (4 boxes, shifted right by half step) ──────────────────
FX = [xs(0), xs(1), xs(2), xs(3)]
box(ax, FX[0], YF, "VGGT4D Prior",    "Gram sim. attn.",     C_VGGT)
arrow(ax, FX[0]+BW/2+GAP, YF, FX[1]-BW/2-GAP)
box(ax, FX[1], YF, "Weighted Fusion", "α·motion+(1-α)·sem.", C_FUSE)
arrow(ax, FX[1]+BW/2+GAP, YF, FX[2]-BW/2-GAP)
box(ax, FX[2], YF, "SAM 3 Refine",   "Phase 6 only",        C_SAM3)
arrow(ax, FX[2]+BW/2+GAP, YF, FX[3]-BW/2-GAP)
box(ax, FX[3], YF, "ProPainter",      "dual-domain prop.",   C_INP)
arrow(ax, FX[3]+BW/2+GAP, YF, xs(4)-BW/2-GAP)
box(ax, xs(4), YF, "Output F/P6",    "MS: 0.8561",          C_OUT)

# ── Route G failure note ──────────────────────────────────────────────────────
ax.text(xs(4), YB, "Route G\n(Diffusion)\n✗ flickering",
        ha="center", va="center", fontsize=7, color="#C0392B",
        bbox=dict(boxstyle="round,pad=0.22", fc="#FADBD8", ec="#C0392B", lw=1))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_SEG,  label="Segmentation / Tracking"),
    mpatches.Patch(color=C_FLOW, label="Optical Flow"),
    mpatches.Patch(color=C_VGGT, label="VGGT4D Prior"),
    mpatches.Patch(color=C_SAM3, label="SAM 3 Refinement"),
    mpatches.Patch(color=C_FUSE, label="Mask Fusion"),
    mpatches.Patch(color=C_INP,  label="Video Inpainting"),
    mpatches.Patch(color=C_OUT,  label="Output"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=7,
          framealpha=0.88, ncol=2, handlelength=1.1,
          bbox_to_anchor=(0.995, 0.97))

ax.set_title("Video Object Removal Pipeline: Routes A / B / E / F / Phase 6",
             fontsize=10, fontweight="bold", pad=6)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
