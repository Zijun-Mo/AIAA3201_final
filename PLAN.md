# Project 3 执行计划（优化版）

> 项目：Video Object Removal & Inpainting  
> 目标：在保证可交付的前提下，先完成稳定主线，再做可控探索。

---

## 0. 总体策略

### 0.1 交付优先级（必须遵守）
1. **必须先保证能交作业**：A + B 路线、3 个 mandatory 数据集结果、核心指标与可视化。
2. **再做加分探索**：E/F/Phase 6 按 `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR` 的收益逐步推进；G 仅保留定性失败分析。
3. **每个阶段必须可复现**：配置、指标、可视化、日志四件套必须齐全。

### 0.2 成功标准（Definition of Done）
- 代码可复现跑通（含 README 指令）。
- mandatory 数据集（Wild / bmx-trees / tennis）都有输出视频。
- 指标完整：`GT_Coverage`、JM、JR、彩色 TCF、FAST-VQA。
- 有可用于论文的表格和图（含 failure cases）。
- 有一条明确的结论链：`A-best -> B-best -> (B+E / B+F / Phase6-best)`；G 作为 diffusion 定性失败案例。

---

## 1. 阶段化执行（带验收门槛）

## Phase 0：统一实验设置（0.5-1 天）

### 目标
建立可比实验基线，避免后续结果不可比。

### 任务
- 数据集整理：Wild、bmx-trees、tennis、可选 DAVIS。
- 统一预处理：分辨率、fps、帧命名、mask 格式。  
- 统一评价协议：`MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR` + 彩色 TCF + FAST-VQA、可视化样式。
- 建立实验记录模板。

### 输出
- 标准目录结构与命名规范。
- 统一评估脚本接口设计。
- 实验记录模板（参数、耗时、显存、结果、现象）。

### 验收门槛（Go/No-Go）
- 任意方法输出都能进入同一评估脚本。
- 同一视频可稳定复现实验结果。

### 验收结果
- **状态**：PASS（2026-04-30）
- 3 个 mandatory 数据集（wild / bmx-trees / tennis）预处理完成，帧命名与 mask 格式统一。
- 统一评估脚本接口验证通过，任意方法输出均可进入同一评估流程。

### 风险与回退
- 风险：数据格式不一致导致评估失败。
- 回退：先做 frame-level 标准化脚本，只允许标准输入进入后续阶段。

---

## Phase 1：路线 A（经典 baseline，1.5-2 天）

`YOLOv8-seg / Mask R-CNN -> 光流动态筛选 -> cv2.inpaint`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| A1 | 检测/分割对比 | YOLOv8-seg vs Mask R-CNN | 0.5 天 | 找到更稳 mask 来源 |
| A2 | 动态筛选 | 光流阈值 | 0.5 天 | 减少静态误删 |
| A3 | mask 边界处理 | dilation kernel | 0.25 天 | 减少残边 |
| A4 | 修复算法对比 | Telea vs NS | 0.25 天 | 确定 A-best |
| A5 | 简易时序传播 | 前后帧借背景 | 0.25-0.5 天 | 减轻闪烁 |

### 输出
- `A-best`（作为最低基线）。
- A 路线失败案例包（拖影、纹理缺失、边界残留）。

### 验收门槛
- 3 个 mandatory 数据集至少跑通 A-best。
- A-best 指标与可视化可复现。

### 验收结果
- **状态**：PASS（最新复跑：2026-05-10）
- Exp ID：`phase1_maskscore_fastvqa_20260510_023457_pl220`（`check_phase1`: PASS）
- A-best：`A5 / temporal_2`，即 Mask R-CNN + 光流筛选（threshold=0.8）+ dilation=3 + Telea + temporal borrowing（window=2）。
- Aggregate（A-best）：`GT_Coverage=0.9222`，`JM=0.6059`，`JR=0.7188`，`MaskScore=0.7923`，`TCF(color)=0.0608`，`FAST_VQA=0.0835`。
- 3 个 mandatory 数据集均跑通 A-best，输出视频与 mask 视频齐全；`wild` 无 GT，不参与 mask 聚合。

---

## Phase 2：路线 B（主线方法，2.5-3.5 天）

`SAM2 / Track Anything + bidirectional no-wrap propagation -> ProPainter`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| B1 | 动态 mask 方案对比 | SAM2 vs Track Anything | 0.5-1 天 | 选主分割方案 |
| B2 | ProPainter 接入 | 固定 mask 输入 | 0.5 天 | 跑通主链路 |
| B3 | 上游质量影响 | 粗 mask vs 精 mask | 0.5 天 | 证明 mask 重要性 |
| B4 | 推理参数实验 | clip length / neighbor frames | 0.5 天 | 找稳定配置 |
| B5 | mandatory 全量评估 | Wild     / bmx / tennis | 0.5-1 天 | 形成主结果表 |

### 输出
- `B-best`（主方法）。
- `A-best vs B-best` 定量+定性对比。
- B-best 后处理协议需记录：全局单一 mask backend、prompt anchors、`bidirectional_no_wrap` 传播策略、是否触发 fallback。

### 验收门槛
- B-best 在多数场景优于 A-best（至少在视觉质量和关键指标上）。
- 3 个 mandatory 数据集完整视频输出齐全。

### 验收结果
- **状态**：PASS（最新复跑：2026-05-10）
- Exp ID：`phase2_maskscore_fastvqa_20260510_023457_pl220`（`check_phase2 --strict-dual-run false`: PASS）
- B-best：`B5 / b_best_finalize`，backend=`trackanything`，mask variant=`coarse`。
- Aggregate（B-best）：`GT_Coverage=0.9208`，`JM=0.6387`，`JR=0.7562`，`MaskScore=0.8091`，`TCF(color)=0.0640`，`FAST_VQA=0.1060`。
- A→B delta：`delta_GT_Coverage=-0.0015`，`delta_JM=+0.0328`，`delta_JR=+0.0375`，`delta_MaskScore=+0.0168`，`delta_TCF=+0.0031`，`delta_FAST_VQA=+0.0225`。
- 3 个 mandatory 数据集完整视频输出齐全；B-best 在新 `MaskScore` 上优于 A-best。

### 风险与回退
- 风险：SOTA 环境复杂，短期无法稳定。
- 回退：固定最稳分割 + 降低 ProPainter 参数规模，先保交付完整性。

---     

## Phase 3：路线 E（低风险高收益优化，1.5-2 天）

`B + mask enhancement`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| E1 | mask 后处理 | opening/closing/dilation/smoothing | 0.25-0.5 天 | 降低边界毛刺 |
| E2 | temporal smoothing | 融合窗口大小 | 0.5 天 | 减少 flicker |
| E3 | 更强 refiner | SAM3/其他 | 0.5-1 天 | 提升边界精度 |
| E4 | 增益验证 | GT_Coverage/JM/JR/彩色 TCF | 0.25 天 | 给出量化提升 |

### 输出
- `B vs B+E` 对比图与增益表。
- 边界区域可视化专题图。

### 验收门槛
- 至少一个 mandatory 数据集上出现可解释的提升。

### 验收结果
- **状态**：PASS（最新复跑：2026-05-10）
- Exp ID：`phase3_maskscore_fastvqa_20260510_023457_pl220`（`check_phase3 --strict-sam3-permission true`: PASS）
- E-best：`E4 / b_plus_e_finalize`，SAM3 multi-anchor refiner 与 B-best 提示词策略统一。
- Aggregate（B+E）：`GT_Coverage=0.9531`，`JM=0.6328`，`JR=0.7625`，`MaskScore=0.8254`，`TCF(color)=0.0647`，`FAST_VQA=0.1258`。
- B→E delta：`delta_GT_Coverage=+0.0323`，`delta_JM=-0.0058`，`delta_JR=+0.0063`，`delta_MaskScore=+0.0163`，`delta_TCF=+0.0007`，`delta_FAST_VQA=+0.0198`。
- bmx-trees 上覆盖提升明显；tennis 维持较高 `MaskScore`；`wild` 无 GT mask 故不参与 mask 聚合。

---

## Phase 4：路线 F（研究探索：VGGT4D prior + backend 对照，2-3 天）

**状态（2026-05-10）：按 MaskScore 协议与 alternate backend 口径完成复跑验收。**
- 最新实验：`phase4_maskscore_fastvqa_altbackend_20260510_pl220`（`check_phase4`: PASS）
- 修正规则：Phase4 final 从 F-stage 全部成功候选中按 `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR` 选择；F3 固定测试 `VGGT4D prior + B-best 未选择的后端`。
- 关联 Phase2：`phase2_maskscore_fastvqa_20260510_023457_pl220`

`VGGT4D raw mask -> B-best mask backend + bidirectional no-wrap propagation = VGGT4D prior -> ProPainter`

### 口径约束
- Phase4 不再做 YOLO-only / VGGT4D+YOLO / mask prior fusion 对照。
- F3 专门测试 `VGGT4D prior + Phase2 B-best 未选择的后端`。若 B-best 是 Track Anything，则 F3 用 SAM2；若 B-best 是 SAM2，则 F3 用 Track Anything。
- 路线 F 最终导出的视频、指标与报告主结果必须按全局 `MaskScore` 选择，不再强制来自 `VGGT4D prior`。
- `VGGT4D prior` 不是直接使用 VGGT4D 原始 mask；它指 VGGT4D 输出先转为 prompt anchors，再经过 B-best 实际使用的全局 mask backend 与 `bidirectional_no_wrap` 传播策略后的结果。
- 首尾不连续视频按普通非循环视频处理；SAM2/Track Anything 禁止把尾帧记忆直接接到首帧。

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| F1 | VGGT4D prior 主结果 | chunk_size / frame chunk strategy / B-best mask backend | 0.5 天 | 生成路线 F 最终视频与指标 |
| F2 | B-best 基线复现实验 | B-best 固定配置 | 0.25 天 | 作为 F 路线对照基线 |
| F3 | alternate backend 对照 | VGGT4D prior + 非 B-best 后端 | 0.5 天 | 判断 VGGT4D prior 对 backend 的敏感性 |
| F4 | flow/trajectory refinement | bidirectional consistency / weighted motion filtering | 0.5-1 天 | 仅作为消融，分析是否带来增益或副作用 |
| F5 | 难例对比 | 遮挡/多目标/静态人车 | 0.25 天 | 解释 VGGT4D prior 的适用边界 |

### 输出
- `F-final` 的视频、指标与报告主结果，选择标准为全局 `MaskScore`。
- `VGGT4D raw vs VGGT4D prior+B-best backend vs VGGT4D prior+alternate backend` 对比结果。
- `B-best vs F-final` 对比结果（视频级）。
- 难场景 case study（重点讲“该不该删”的判断改进）。

### 验收门槛
- 3 个 mandatory 数据集均有 `VGGT4D prior` 版本视频与指标。
- `phase4_ablation.csv` 中 `is_final_best=1` 的候选必须达到全表最高 `MaskScore`。
- 若 `VGGT4D prior` 生成失败，路线 F 标记为失败并记录原因，禁止改用 YOLO 或融合结果冒充路线 F 最终产物。

### 验收结果
- **状态**：PASS（最新复跑：2026-05-10）
- Exp ID：`phase4_maskscore_fastvqa_altbackend_20260510_pl220`（`check_phase4`: PASS）
- F-best：`F3 / vggt4d_prior_sam2`。Phase2 B-best 使用 Track Anything，因此 alternate backend 实验使用 SAM2。
- Aggregate（F-best）：`GT_Coverage=0.9722`，`JM=0.7074`，`JR=0.7688`，`MaskScore=0.8552`，`TCF(color)=0.0698`，`FAST_VQA=0.1201`。
- B→F delta：`delta_GT_Coverage=+0.0515`，`delta_JM=+0.0687`，`delta_JR=+0.0125`，`delta_MaskScore=+0.0460`，`delta_TCF=+0.0058`，`delta_FAST_VQA=+0.0140`。
- Backend prior audit：`phase4_backend_priors.csv` 记录 VGGT4D raw、VGGT4D prior + Track Anything（B-best backend）、VGGT4D prior + SAM2（alternate backend）的 mask ratio。
- 结论：删去 mask-prior 对照后，`VGGT4D prior + SAM2` 显著优于 B-best 的综合 mask 分数；主要收益来自更高 GT 覆盖和 JM，但彩色 TCF 较 B-best 略差，说明仍有时序稳定性代价。

---

## Phase 5：路线 G（扩散生成修复，2-3 天）

`fixed mask -> ProPainter stable base / diffusion inpainting -> keyframe propagation`

实现入口：`src/part3/run_diffusion.py`（已从 legacy `src/part4/run_diffusion.py` 迁移）。

### 口径约束
- 路线 G 不作为 ProPainter 的全局替代，而是验证 diffusion 在“背景从未出现 / 不可借像素”区域的生成式补强价值。
- 与 ProPainter 对比时必须固定同一套 `B-best` mask，避免 mask 差异污染 inpainting 结论。
- 默认提示词采用场景无关模板，不为每个视频手动设计纹理：正向强调“高质量、真实纹理、光照一致、空背景”，负向排除原动态目标、水印、文字、风格化与畸变。
- 每个 diffusion 实验必须记录 seed、steps、guidance/cfg、denoise strength、mask dilation、keyframe interval 与是否使用 ProPainter base。

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| G1 | 固定 mask 输入对齐 | `B-best` mask，same-mask ProPainter vs diffusion | 0.25-0.5 天 | 隔离 inpainting 差异 |
| G2 | 重绘幅度消融 | low/mid/high：mask dilation + denoise strength | 0.5-1 天 | 评估重绘幅度对真实性与稳定性的影响 |
| G3 | 通用提示词策略 | generic prompt / negative prompt / fixed seed | 0.25 天 | 减少人工场景设计，提高可复现性 |
| G4 | 时序策略对比 | framewise / every 4 frames / every 8 frames / propagation | 0.5-1 天 | 量化 diffusion 闪烁与传播收益 |
| G5 | Hybrid refinement | ProPainter base + diffusion 局部重绘 | 0.5 天 | 保留时序稳定性，同时增强不可见背景纹理 |
| G6 | 背景未出现专题 | borrowable vs unobserved background | 0.25-0.5 天 | 展示生成式优势与适用边界 |
| G7 | 失败分析 | 风格漂移/闪烁/结构幻觉/边界接缝 | 0.25 天 | 诚实分析局限 |

### 推荐重绘幅度网格
| Variant | Mask Dilation | Denoise Strength | 定位 |
| --- | ---: | ---: | --- |
| G-low | 0-4 px | 0.30-0.40 | 保守重绘，优先时序稳定 |
| G-mid | 8-12 px | 0.50-0.60 | 主推荐配置，平衡真实纹理与稳定性 |
| G-high | 16-24 px | 0.70-0.80 | 强重绘，观察纹理真实性上限与闪烁代价 |
| G-extreme | 32 px | 0.85+ | 仅作失败案例，不作为最终结果 |

### 默认提示词
- Positive prompt：`high quality, realistic texture, natural lighting, clean empty background, temporally consistent video frame`
- Negative prompt：`person, human, cyclist, bicycle, tennis player, racket, car, vehicle, text, logo, watermark, cartoon, painting, blurry, distorted geometry, inconsistent lighting`
- 默认不做逐视频纹理描述；若通用提示词明显失败，只允许加入一个粗粒度场景词（如 `forest trail` / `tennis court`），并在日志中记录。

### 输出
- `same-mask ProPainter vs G-low/G-mid/G-high` 定性可视化对比，不作为定量主结果。
- `framewise diffusion vs keyframe diffusion vs hybrid refinement` 连续帧定性对比，重点观察闪烁、边界接缝与风格漂移。
- “背景从未出现”专题案例与连续帧 strip。
- G 路线失败案例包（闪烁、风格漂移、结构幻觉、边界接缝）。

### 验收门槛
- Phase 5 仅作为定性探索与失败分析，不纳入定量主表，也不作为最终视频修复主方法。
- 必须保留连续帧可视化，明确报告 diffusion 对闪烁、边界接缝、风格漂移和结构幻觉的影响。

### 验收结果
- **状态**：PASS；完成定性探索，结论为失败/不作为主方法（最新复跑：2026-05-10）
- Exp ID：`phase5_maskscore_fastvqa_20260510_023457_pl220`（`check_phase5`: PASS）
- 选择设置：`G-hybrid`（ProPainter base + diffusion local redraw），按 `MaskScore` 与其他 G variants 并列最高。
- Aggregate（G-hybrid）：`GT_Coverage=0.9208`，`JM=0.6387`，`JR=0.7562`，`MaskScore=0.8091`，`TCF(color)=0.0490`，`FAST_VQA=0.0991`。
- Model：`stable-diffusion-v1-5/stable-diffusion-inpainting`，device=cuda（RTX 4080 SUPER）
- 验收结论：Phase 5 会严重加剧时序闪烁和边界问题；局部纹理虽可能更“生成式”，但连续帧稳定性、接缝一致性和结构可信度不足。
- 因上述问题，G 路线只进入定性失败案例与“背景未出现”讨论，不进入主方法定量排名，也不选择任何 G-best。
- 3 个 mandatory 数据集均有可用于定性检查的视频输出。

---

## Phase 6：E/F 核心修改点叠加验证（目标确认是否超过 E-best 与 F-best，0.5 天）

`B-best / E-best / F-best references -> F-best prior + SAM3 refinement -> ProPainter`

### 已确认起点
- B-best：`phase2_maskscore_fastvqa_20260510_023457_pl220`，`GT_Coverage=0.9208`，`JM=0.6387`，`JR=0.7562`，`MaskScore=0.8091`。
- E-best：`phase3_maskscore_fastvqa_20260510_023457_pl220 / E4 b_plus_e_finalize`，`GT_Coverage=0.9531`，`JM=0.6328`，`JR=0.7625`，`MaskScore=0.8254`。
- F-best：`phase4_maskscore_fastvqa_altbackend_20260510_pl220 / F3 vggt4d_prior_sam2`，`GT_Coverage=0.9722`，`JM=0.7074`，`JR=0.7688`，`MaskScore=0.8552`。

### 目标
- Phase 6 不是广泛对比实验；它只确认 E-best 的 SAM3 refine 核心修改点与 F-best 的 VGGT4D prior + SAM2 核心修改点叠加后，是否能同时高于 E-best 与 F-best。
- Phase6-best 和各阶段 mask-best 的选择标准为 `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR`。
- 彩色 TCF 和 FAST-VQA 作为视频质量副指标报告；不使用 ROS/BES。
- Phase 6 不再包含 dataset selector、oracle upper bound、guided fusion 或 union-safe 等横向搜索项；这些旧实验只作为历史参考，不进入最终文档口径。

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| H0 | 起点复核 | `b_best_ref` / `e_best_ref` / `f_best_ref` | 0.1 天 | 在同一 Phase 6 管线中复现 B-best、E-best、F-best |
| H1 | 核心叠加 | `f_best_then_sam3` | 0.25 天 | 验证 `VGGT4D prior + SAM2 -> SAM3` 是否同时超过 E-best 与 F-best |

### 推荐候选顺序
1. `H0_b_best_ref`：复现 Phase2 B-best。
2. `H0_e_best_ref`：复现 Phase3 E-best（`morph_light + SAM3 refine`）。
3. `H0_f_best_ref`：复现 Phase4 F-best（`VGGT4D prior + SAM2`）。
4. `H1_f_best_then_sam3`：唯一新实验，直接测试 F-best 是否能被 SAM3 继续提升。

### 验收门槛
- 最低门槛：Phase6-best 必须达到 B-best 的综合 mask 水平，即 `MaskScore >= 0.8091`，且 `JM/JR` 不出现不可解释的大幅退化。
- 强门槛：Phase6-best 的 `MaskScore` 高于 E-best 与 F-best（报告口径），即超过 `max(0.8254, 0.8552)=0.8552`。
- 理想门槛：`H1_f_best_then_sam3` 同时达到或超过 F-best 的 `GT_Coverage=0.9722`、`JM=0.7074`、`JR=0.7688`，并且超过 E-best 的 `JR=0.7625`。

### 输出
- `Phase6-best` 视频、mask 视频、`summary.json`、`per_dataset.csv`、`phase6_ablation.csv`、`phase6_selection.json`。
- `B/E/F/Phase6` 的 `GT_Coverage/JM/JR/TCF/FAST_VQA` 主表与 per-dataset 对比表。
- `B vs E vs F vs Phase6` 边界专题图，重点展示 bmx-trees 和 tennis 的提升/退化区域。
- 若 Phase 6 未同时超过 E-best 与 F-best，保留失败原因：过度扩张、漏检、VGGT4D prior 与 SAM3 边界冲突、per-frame 不稳定。

- **状态**：PASS（2026-05-12；`check_phase6 --strict-sam3-permission true`: PASS）
- 旧 Exp ID：`phase6_20260509_123005_pl220` 是灰度 TCF + ROS/BES + JM/JR-only 排序口径，仅保留历史参考，不再作为最终结论。
- 历史 Exp ID：`phase6_maskscore_fastvqa_20260511_172500_pl220`、`phase6_v2_maskscore_fastvqa_20260511_183000_pl220` 仅保留历史参考，不再作为最终 Phase6 结论。
- Exp ID：`phase6_core_maskscore_fastvqa_20260511_235610_pl220`。
- Phase6-best：`H1 / f_best_then_sam3`，即 `VGGT4D prior + SAM2 -> SAM3 refine`，再用 ProPainter 修复。
- 全局非 oracle 选择集：有 GT 的 mandatory 数据集；`wild` 因 GT mask 缺失被自动排除出 `GT_Coverage/JM/JR/MaskScore` 选择，但仍生成最终视频和 mask 视频。
- Aggregate（Phase6-best）：`GT_Coverage=0.9741`，`JM=0.7075`，`JR=0.7688`，`MaskScore=0.8561`，`TCF(color)=0.0697`，`FAST_VQA=0.1234`。
- B→Phase6 delta：`delta_GT_Coverage=+0.0534`，`delta_JM=+0.0688`，`delta_JR=+0.0125`，`delta_MaskScore=+0.0470`，`delta_TCF=+0.0058`，`delta_FAST_VQA=+0.0173`。
- F→Phase6 delta：`delta_GT_Coverage=+0.0019`，`delta_JM=+0.0001`，`delta_JR=+0.0000`，`delta_MaskScore=+0.0010`，`delta_TCF=-0.0000`，`delta_FAST_VQA=+0.0033`（相对 Phase4 final summary）。
- 结论：H1 叠加后 `MaskScore=0.8561`，略高于 E-best `0.8254` 和 F-best `0.8552`；收益很小，主要来自 `GT_Coverage` 的轻微上升。

### Phase 6 候选结果
| Candidate | GT_Coverage↑ | JM↑ | JR↑ | MaskScore↑ | TCF(color)↓ | FAST_VQA↑ | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `H0/b_best_ref` | 0.9196 | 0.6378 | 0.7500 | 0.8067 | 0.0649 | 0.1190 | B-best 在 Phase 6 管线中的复核 |
| `H0/e_best_ref` | 0.9531 | 0.6328 | 0.7625 | 0.8254 | 0.0647 | 0.1195 | 复现 E-best 核心路径 |
| `H0/f_best_ref` | 0.9722 | 0.7074 | 0.7688 | 0.8552 | 0.0698 | 0.1244 | 复现 F-best 核心路径 |
| `H1/f_best_then_sam3` | 0.9741 | 0.7075 | 0.7688 | 0.8561 | 0.0697 | 0.1234 | 最终 Phase6-best；小幅超过 E-best 与 F-best |

---

## 2. 最小必做实验集 vs 加分实验集

## 2.1 最小必做（保证可交付）
- A 路线完整跑通并给出 A-best。
- B 路线完整跑通并给出 B-best。
- 3 个 mandatory 数据集均有输出视频。
- 表 1（总体性能）+ 关键可视化 + failure cases。

## 2.2 加分建议（时间充足）
- E 路线完整增益验证。
- F 路线完整增益验证。
- Phase 6 作为 E/F 核心修改点叠加验证优先推进。
- G 路线只保留定性失败分析，不进入定量主线。
- 在 DAVIS 上补充泛化实验。

---

## 3. 12-14 天排期（优化后）

| 天数 | 任务 |
| --- | --- |
| Day 1 | Phase 0（统一数据、预处理、评估、日志模板） |
| Day 2-3 | Phase 1（A 路线全部 + A-best） |
| Day 4-6 | Phase 2（B 路线全部 + B-best + mandatory 完整结果） |
| Day 7-8 | Phase 3（E 路线） |
| Day 9-10 | Phase 4（F 路线） |
| Day 11 | Phase 6（E/F 核心修改点叠加验证） |
| Day 12 | Phase 5（G 路线定性失败分析） |
| Day 13 | 指标汇总、可视化整理、failure cases 归档 |
| Day 14 | 写实验与分析章节，补齐图表与结论链 |

> 若时间不足：优先保证 Day 1-6 + Day 13-14 完整；E/F/Phase 6 中保留 `GT_Coverage` 优先规则下最强路线，G 只保留失败案例。

---

## 4. 结果表与图（建议最终最少产出）

### 状态
- **状态**：DONE（2026-05-12，已用最新 Phase1-6 accepted runs 重新生成）
- 生成入口：`bash scripts/build_results_artifacts.sh`
- 表格目录：`outputs/metrics/final_results/`
- 图像目录：`outputs/figures/final_results/`
- Manifest：`outputs/metrics/final_results/final_results_manifest.json`

### 已生成表格
- 表 1 总体性能主表：`table1_main_performance.{csv,md,tex}`
- 表 2 A 路线消融：`table2_a_ablation.{csv,md,tex}`
- 表 3 B/E/F/Phase6 消融：`table3_bef_phase6_ablation.{csv,md,tex}`
- 表 4 Phase 5/G 定性失败分析：`table4_phase5_qualitative.{csv,md,tex}`
- 汇总说明：`final_results_summary.md`

### 表 1 当前核心数值
| Method | GT_Coverage↑ | JM↑ | JR↑ | MaskScore↑ | TCF(color)↓ | FAST_VQA↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-best | 0.9222 | 0.6059 | 0.7188 | 0.7923 | 0.0608 | 0.0835 |
| B-best | 0.9208 | 0.6387 | 0.7562 | 0.8091 | 0.0640 | 0.1060 |
| B+E | 0.9531 | 0.6328 | 0.7625 | 0.8254 | 0.0647 | 0.1258 |
| B+F | 0.9722 | 0.7074 | 0.7688 | 0.8552 | 0.0698 | 0.1201 |
| Phase6-best | 0.9741 | 0.7075 | 0.7688 | 0.8561 | 0.0697 | 0.1234 |

> Phase 5 / G 路线只做定性失败分析，不进入总体性能主表的定量排名。

### 指标解释：TCF 与 FAST-VQA
- 当前 TCF 是彩色帧时序一致性指标，数值越低只说明预测 mask ROI 内的帧间差异越小；它不能单独代表补全真实度或视觉质量。
- A 路线和 G 路线的 TCF 较低并不等价于修复更好。主要原因是模糊、低频、错误但稳定的补全会天然降低帧间差异；相反，更真实的纹理、边缘和结构细节会让 optical-flow 对齐误差更明显，从而可能抬高 TCF。
- 因此 TCF 只作为“时序平滑程度”的辅助指标，不作为 best 选择依据，也不能用来否定 `MaskScore` 和视觉对比中的提升。
- FAST-VQA 的排序更接近人工观感：A-best `0.0835` 与 G-hybrid `0.0991` 较低，B+E `0.1258`、Phase6-best `0.1234` 较高；这与定性观察中 A 的模糊补全、G 的闪烁/边界问题、E/Phase6 的生成质量更可靠基本一致。
- 最终论文口径应使用 `MaskScore` 评价 mask 质量，用 FAST-VQA 与可视化互相印证生成质量；TCF 只报告其局限性与时序一致性 trade-off。

### 已生成图
- `figure1_overall_metrics.png`：总体指标柱状图。
- `figure2_a_vs_b_global.png`：A-best vs B-best 全局对比图。
- `figure3_b_e_f_phase6_visual.png`：B/E/F/Phase6 主视觉对比图。
- `figure4_b_e_f_phase6_boundary.png`：B vs E vs F vs Phase6 边界细节图。
- `figure5_b_f_phase6_hard_scene.png`：B vs F(VGGT4D prior) vs Phase6 难场景图。
- `figure6_jmjr_main_methods.png`：主方法 JM/JR 散点图。
- `figure7_phase6_pareto.png`：Phase 6 H0/H1 核心候选分布图（文件名沿用 pareto，但不代表广泛搜索）。
- `figure8_phase5_failure_strip.png`：B vs G 连续帧定性失败 strip。

---

## 5. 统一命名与落盘规范（强烈建议）

### 5.1 命名模板
- 实验 ID：`<route>-<dataset>-<variant>-<date>`
- 指标文件：`metrics_<exp_id>.json`
- 视频文件：`video_<exp_id>.mp4`
- 可视化：`viz_<exp_id>_<frame_id>.png`

### 5.2 每次实验必须落盘的最小集合
- 配置：`config_<exp_id>.yaml/json`
- 指标：`metrics_<exp_id>.json`
- 可视化：至少 3 张关键帧
- 日志：`log_<exp_id>.txt`

---

## 6. 重点观测现象（用于论文分析段）

| 观察项 | 问题定义 | 建议解释方向 |
| --- | --- | --- |
| 边界残留 | 人/车轮廓是否留下“幽灵边” | mask 边界质量与 inpaint 接缝问题 |
| 时序闪烁 | 相邻帧背景是否跳动 | mask flicker 或传播不稳定 |
| 纹理模糊 | 墙砖/地面/树叶是否发糊 | 传播失真或生成保真度不足 |
| 大遮挡失败 | 大面积遮挡后能否恢复 | 上下文不足、传播链断裂 |
| 未出现背景 | 从未出现区域能否修复 | 传播法上限 vs 生成法优势 |
| 静态物体误删 | 停着的人/车会否被删 | “是否动态”判别错误 |
