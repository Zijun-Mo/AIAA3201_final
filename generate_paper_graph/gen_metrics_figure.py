"""
Generate metrics bar chart for the paper (Figure 2).
Outputs: ../tex_workspace/figures/fig_metrics.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "../tex_workspace/figures/fig_metrics.pdf"

methods = ["A-best", "B-best", "B+E", "B+F", "Phase 6"]
gt_cov  = [0.9222, 0.9208, 0.9531, 0.9722, 0.9741]
jm      = [0.6059, 0.6387, 0.6328, 0.7074, 0.7075]
jr      = [0.7188, 0.7562, 0.7625, 0.7688, 0.7688]
mscore  = [0.7923, 0.8091, 0.8254, 0.8552, 0.8561]
fastvqa = [0.0835, 0.1060, 0.1258, 0.1201, 0.1234]

x = np.arange(len(methods))
w = 0.16

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor("#F8F9FA")

# ── Left: mask quality ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#F8F9FA")
colors = ["#4A90D9", "#27AE60", "#E67E22", "#8E44AD"]
bars = [gt_cov, jm, jr, mscore]
labels = ["GT Coverage", "JM", "JR", "MaskScore"]
for i, (vals, lbl, col) in enumerate(zip(bars, labels, colors)):
    rects = ax.bar(x + i * w, vals, w, label=lbl, color=col, alpha=0.88,
                   edgecolor="white", linewidth=0.8)
    # annotate MaskScore only
    if lbl == "MaskScore":
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=col,
                    fontweight="bold")

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylim(0.55, 1.02)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("Mask Quality Metrics", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# ── Right: video quality ──────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#F8F9FA")
tcf = [0.0608, 0.0640, 0.0647, 0.0698, 0.0697]
bar1 = ax2.bar(x - w/2, fastvqa, w*1.5, label="FAST-VQA ↑", color="#C0392B",
               alpha=0.88, edgecolor="white")
ax2_r = ax2.twinx()
bar2 = ax2_r.bar(x + w, tcf, w*1.5, label="TCF ↓", color="#2980B9",
                 alpha=0.6, edgecolor="white")

ax2.set_xticks(x + w/4)
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel("FAST-VQA", fontsize=9, color="#C0392B")
ax2_r.set_ylabel("TCF (lower=better)", fontsize=9, color="#2980B9")
ax2.set_title("Video Quality Metrics", fontsize=10, fontweight="bold")
ax2.spines[["top"]].set_visible(False)
ax2_r.spines[["top"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

lines = [bar1, bar2]
labels2 = ["FAST-VQA ↑", "TCF ↓"]
ax2.legend(lines, labels2, fontsize=8, loc="upper left")

plt.suptitle("Quantitative Comparison Across Methods", fontsize=11,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
