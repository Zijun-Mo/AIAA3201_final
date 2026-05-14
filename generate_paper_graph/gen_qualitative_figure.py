"""
Qualitative comparison figure: 2 datasets × 4 methods, showing restored frames only.
Layout: rows = datasets (bmx-trees, tennis), cols = methods (A-best, B-best, B+E, Phase6)
Output: ../tex_workspace/figures/fig_qualitative.pdf
"""
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).parent.parent
VIDEOS = ROOT / "outputs/videos"
OUT = ROOT / "tex_workspace/figures/fig_qualitative.pdf"
TMP = Path("/tmp/paper_frames")
TMP.mkdir(exist_ok=True)

METHODS = {
    "A-best":  "phase1_maskscore_fastvqa_20260510_023457_pl220",
    "B-best":  "phase2_maskscore_fastvqa_20260510_023457_pl220",
    "B+E":     "phase3_maskscore_fastvqa_20260510_023457_pl220",
    "Phase6":  "phase6_core_maskscore_fastvqa_20260511_235610_pl220",
}
DATASETS = ["bmx-trees", "tennis"]
FRAME_SEC = {"bmx-trees": 1, "tennis": 1}


def extract_frame(mp4: Path, out_png: Path, sec: int):
    if out_png.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(sec), "-i", str(mp4),
         "-frames:v", "1", str(out_png)],
        check=True, capture_output=True
    )


def load(p: Path):
    img = mpimg.imread(str(p))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


n_rows = len(DATASETS)
n_cols = len(METHODS)

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 2.8, n_rows * 1.8))
fig.subplots_adjust(hspace=0.08, wspace=0.04,
                    left=0.08, right=0.99, top=0.92, bottom=0.02)

for r, ds in enumerate(DATASETS):
    for c, (method, phase_dir) in enumerate(METHODS.items()):
        mp4 = VIDEOS / phase_dir / ds / "restored_h264.mp4"
        png = TMP / f"qual_{phase_dir}_{ds}.png"
        extract_frame(mp4, png, FRAME_SEC[ds])
        ax = axes[r][c]
        ax.imshow(load(png))
        ax.axis("off")
        if r == 0:
            ax.set_title(method, fontsize=9, pad=3)
        if c == 0:
            ax.set_ylabel(ds, fontsize=8, rotation=0,
                          labelpad=48, va="center")

fig.savefig(str(OUT), bbox_inches="tight", dpi=150)
print(f"Saved: {OUT}")
