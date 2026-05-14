"""
Generate per-dataset breakdown + incremental gain chart (Figure 3).
Outputs: ../tex_workspace/figures/fig_ablation.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "../tex_workspace/figures/fig_ablation.pdf"

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor("#F8F9FA")

# ── Left: per-dataset MaskScore (Phase 6) ────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#F8F9FA")
datasets = ["bmx-trees", "tennis"]
gt_cov_d = [0.9548, 0.9935]
jm_d     = [0.4938, 0.9212]
jr_d     = [0.5375, 1.0000]
ms_d     = [0.7352, 0.9770]

x = np.arange(len(datasets))
w = 0.18
colors = ["#4A90D9", "#27AE60", "#E67E22", "#8E44AD"]
for i, (vals, lbl, col) in enumerate(zip(
        [gt_cov_d, jm_d, jr_d, ms_d],
        ["GT Coverage", "JM", "JR", "MaskScore"], colors)):
    ax.bar(x + i * w, vals, w, label=lbl, color=col, alpha=0.88,
           edgecolor="white", linewidth=0.8)

ax.set_xticks(x + 1.5 * w)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score", fontsize=9)
ax.set_title("Phase 6: Per-Dataset Breakdown", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.text(0, 0.74, "0.7352", ha="center", va="bottom", fontsize=8,
        color="#8E44AD", fontweight="bold")
ax.text(1, 0.98, "0.9770", ha="center", va="bottom", fontsize=8,
        color="#8E44AD", fontweight="bold")

# ── Right: incremental MaskScore gain ────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#F8F9FA")
stages = ["A-best\n(baseline)", "B-best\n(+SOTA)", "B+E\n(+SAM3)", "B+F\n(+VGGT4D)", "Phase 6\n(+both)"]
scores = [0.7923, 0.8091, 0.8254, 0.8552, 0.8561]
deltas = [0, 0.0168, 0.0163, 0.0460, 0.0009]
bar_colors = ["#95A5A6", "#4A90D9", "#D35400", "#16A085", "#2C3E50"]

bars = ax2.bar(range(len(stages)), scores, color=bar_colors, alpha=0.88,
               edgecolor="white", linewidth=0.8, width=0.6)
for i, (rect, s, d) in enumerate(zip(bars, scores, deltas)):
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.003,
             f"{s:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
             color=bar_colors[i])
    if d > 0:
        ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() - 0.018,
                 f"+{d:.4f}", ha="center", va="top", fontsize=7.5,
                 color="white", fontweight="bold")

ax2.set_xticks(range(len(stages)))
ax2.set_xticklabels(stages, fontsize=8)
ax2.set_ylim(0.75, 0.88)
ax2.set_ylabel("MaskScore", fontsize=9)
ax2.set_title("Incremental MaskScore Improvement", fontsize=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT}")
