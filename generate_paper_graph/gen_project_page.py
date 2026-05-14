"""
Generate docs/index.html and copy required videos for GitHub Pages project page.
"""
import shutil
from pathlib import Path

ROOT   = Path(__file__).parent.parent
VIDEOS = ROOT / "outputs/videos"
DOCS   = ROOT / "docs"
DVID   = DOCS / "videos"
DIMG   = DOCS / "images"
DOCS.mkdir(exist_ok=True)
DVID.mkdir(exist_ok=True)
DIMG.mkdir(exist_ok=True)

# Copy pipeline image
shutil.copy(ROOT / "tex_workspace/ChatGPTimage.png", DIMG / "pipeline.png")

# Videos to show: (label, phase_dir, dataset, video_type)
PHASE_DIRS = {
    "A-best":  "phase1_maskscore_fastvqa_20260510_023457_pl220",
    "B-best":  "phase2_maskscore_fastvqa_20260510_023457_pl220",
    "Phase 6": "phase6_core_maskscore_fastvqa_20260511_235610_pl220",
}
DATASETS   = ["bmx-trees", "tennis", "wild"]
VTYPES     = ["restored_h264.mp4", "mask_overlay_h264.mp4"]

# Copy videos and build relative paths
def vid_rel(method, ds, vtype):
    phase_dir = PHASE_DIRS[method]
    src = VIDEOS / phase_dir / ds / vtype
    if not src.exists():
        return None
    dst_dir = DVID / phase_dir / ds
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / vtype
    if not dst.exists():
        shutil.copy(src, dst)
    return f"videos/{phase_dir}/{ds}/{vtype}"

# ── HTML ─────────────────────────────────────────────────────────────────────
def video_cell(path, label=""):
    if path is None:
        return "<td><em>N/A</em></td>"
    lbl = f'<div class="vid-label">{label}</div>' if label else ""
    return f"""<td>{lbl}<video autoplay muted loop playsinline>
  <source src="{path}" type="video/mp4">
</video></td>"""

def results_section():
    rows = ""
    for ds in DATASETS:
        cells = "".join(
            f"<td><video autoplay muted loop playsinline>"
            f'<source src="{vid_rel(m, ds, "restored_h264.mp4")}" type="video/mp4">'
            f"</video></td>"
            for m in PHASE_DIRS
        )
        rows += f"<tr><th>{ds}</th>{cells}</tr>\n"
    return rows

def failure_section():
    phase5 = "phase5_maskscore_fastvqa_20260510_023457_pl220"
    cells = ""
    for ds in ["bmx-trees", "tennis"]:
        src = VIDEOS / phase5 / ds / "restored_h264.mp4"
        if src.exists():
            dst_dir = DVID / phase5 / ds
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / "restored_h264.mp4"
            if not dst.exists():
                shutil.copy(src, dst)
            rel = f"videos/{phase5}/{ds}/restored_h264.mp4"
            cells += f"<td><div class='vid-label'>{ds}</div><video autoplay muted loop playsinline><source src='{rel}' type='video/mp4'></video></td>"
    return cells

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynamic Object Removal &amp; Background Restoration</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Helvetica Neue", Arial, sans-serif; background: #fff; color: #222; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }}
  h1 {{ font-size: 1.9rem; text-align: center; margin-bottom: .5rem; line-height: 1.3; }}
  .authors {{ text-align: center; color: #555; margin-bottom: .4rem; font-size: 1rem; }}
  .affil {{ text-align: center; color: #777; font-size: .88rem; margin-bottom: 1rem; }}
  .badges {{ display: flex; justify-content: center; gap: .7rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .badge {{ display: inline-block; padding: .35rem .9rem; border-radius: 4px;
            font-size: .85rem; font-weight: 600; text-decoration: none; color: #fff; }}
  .badge-gh  {{ background: #24292e; }}
  .badge-pdf {{ background: #c0392b; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 2rem 0; }}
  h2 {{ font-size: 1.3rem; margin-bottom: 1rem; border-left: 4px solid #4A90D9;
        padding-left: .6rem; }}
  p  {{ line-height: 1.7; color: #333; margin-bottom: .8rem; }}
  .pipeline-img {{ width: 100%; border-radius: 6px; box-shadow: 0 2px 12px rgba(0,0,0,.1); }}
  table.results {{ width: 100%; border-collapse: collapse; }}
  table.results th, table.results td {{ padding: .4rem .3rem; text-align: center;
    vertical-align: middle; border: 1px solid #eee; }}
  table.results th {{ background: #f5f5f5; font-size: .85rem; }}
  table.results td:first-child {{ font-weight: 600; font-size: .85rem; width: 80px; }}
  video {{ width: 100%; border-radius: 4px; display: block; }}
  .vid-label {{ font-size: .75rem; color: #666; margin-bottom: .2rem; }}
  .failure-row {{ display: flex; gap: 1rem; }}
  .failure-row td {{ flex: 1; }}
  pre {{ background: #f6f8fa; padding: 1rem; border-radius: 6px;
         font-size: .82rem; overflow-x: auto; line-height: 1.5; }}
  .note {{ font-size: .82rem; color: #888; margin-top: .4rem; }}
  @media(max-width:600px) {{ h1 {{ font-size: 1.4rem; }} }}
</style>
</head>
<body>
<div class="container">

<h1>Dynamic Object Removal and Background Restoration in Videos:<br>
A Progressive Pipeline from Classical to Geometry-Guided Segmentation</h1>

<p class="authors">Zijun Mo &nbsp;&nbsp; Xiaolong Qiao</p>
<p class="affil">HKUST (Guangzhou) &nbsp;·&nbsp; AIAA 3201 Spring 2026</p>

<div class="badges">
  <a class="badge badge-gh" href="https://github.com/Zijun-Mo/AIAA3201_final" target="_blank">GitHub</a>
  <a class="badge badge-pdf" href="#" target="_blank">Paper (PDF)</a>
</div>

<hr>

<h2>Abstract</h2>
<p>
We present a progressive pipeline for automatic dynamic object removal and background
restoration in videos. Starting from a classical baseline combining Mask R-CNN instance
segmentation, sparse optical flow filtering, and <code>cv2.inpaint</code>, we advance to a
state-of-the-art pipeline built on Track Anything and ProPainter. We then introduce two
complementary improvements: SAM&nbsp;3 mask refinement (Route&nbsp;E) and geometry-guided
segmentation via VGGT4D attention priors (Route&nbsp;F). Our best configuration (Phase&nbsp;6)
achieves a MaskScore of <strong>0.8561</strong> on the mandatory datasets (bmx-trees, tennis),
representing a <strong>+6.9%</strong> gain over the classical baseline. We further analyze a
diffusion-based inpainting route (Route&nbsp;G) and document its failure modes.
</p>

<hr>

<h2>Method Overview</h2>
<img src="images/pipeline.png" alt="Pipeline Overview" class="pipeline-img">

<hr>

<h2>Quantitative Results</h2>
<table class="results" style="margin-bottom:.5rem">
  <thead>
    <tr>
      <th>Method</th><th>GT-Cov</th><th>JM</th><th>JR</th>
      <th>MaskScore</th><th>TCF↓</th><th>FAST-VQA↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>A-best</td><td>0.9222</td><td>0.6059</td><td>0.7188</td><td>0.7923</td><td>0.0608</td><td>0.0835</td></tr>
    <tr><td>B-best</td><td>0.9208</td><td>0.6387</td><td>0.7562</td><td>0.8091</td><td>0.0640</td><td>0.1060</td></tr>
    <tr><td>B+E</td><td>0.9531</td><td>0.6328</td><td>0.7625</td><td>0.8254</td><td>0.0647</td><td>0.1258</td></tr>
    <tr><td>B+F</td><td>0.9722</td><td>0.7074</td><td>0.7688</td><td>0.8552</td><td>0.0698</td><td>0.1201</td></tr>
    <tr style="font-weight:700;background:#f0f7ff">
      <td>Phase 6</td><td>0.9741</td><td>0.7075</td><td>0.7688</td><td>0.8561</td><td>0.0697</td><td>0.1234</td>
    </tr>
  </tbody>
</table>
<p class="note">MaskScore = 0.5·GT-Cov + 0.25·JM + 0.25·JR. Evaluated on bmx-trees and tennis.</p>

<hr>

<h2>Video Results</h2>
<p>Each cell shows the <em>restored</em> output video. Columns: A-best (classical baseline),
B-best (Track Anything + ProPainter), Phase&nbsp;6 (best configuration).</p>
<table class="results">
  <thead>
    <tr><th>Dataset</th>{"".join(f"<th>{m}</th>" for m in PHASE_DIRS)}</tr>
  </thead>
  <tbody>
    {results_section()}
  </tbody>
</table>

<hr>

<h2>Failure Analysis: Route G (Diffusion Inpainting)</h2>
<p>Stable Diffusion inpainting (Route&nbsp;G) produces severe temporal flickering due to
independent per-frame latent noise sampling. Below are the best G-hybrid variant results
compared to B-best — the flickering and boundary hallucination are clearly visible.</p>
<table class="results">
  <thead><tr><th>Dataset</th><th>G-hybrid (Route G)</th></tr></thead>
  <tbody><tr>{failure_section()}</tr></tbody>
</table>

<hr>

<h2>Citation</h2>
<pre>@article{{mo2026videoremoval,
  title   = {{Dynamic Object Removal and Background Restoration in Videos}},
  author  = {{Mo, Zijun and Qiao, Xiaolong}},
  journal = {{AIAA 3201 Course Project}},
  year    = {{2026}}
}}</pre>

</div>
</body>
</html>
"""

(DOCS / "index.html").write_text(html, encoding="utf-8")
print(f"Generated: {DOCS / 'index.html'}")
print(f"Images:    {DIMG}")
print(f"Videos:    {DVID}")
