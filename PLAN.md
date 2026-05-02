# Project 3 执行计划（优化版）

> 项目：Video Object Removal & Inpainting  
> 目标：在保证可交付的前提下，先完成稳定主线，再做可控探索。

---

## 0. 总体策略

### 0.1 交付优先级（必须遵守）
1. **必须先保证能交作业**：A + B 路线、3 个 mandatory 数据集结果、核心指标与可视化。
2. **再做加分探索**：E/F/G 按收益和风险逐步推进。
3. **每个阶段必须可复现**：配置、指标、可视化、日志四件套必须齐全。

### 0.2 成功标准（Definition of Done）
- 代码可复现跑通（含 README 指令）。
- mandatory 数据集（Wild / bmx-trees / tennis）都有输出视频。
- 指标完整：JM、JR、ROS、TCF、BES。
- 有可用于论文的表格和图（含 failure cases）。
- 有一条明确的结论链：`A-best -> B-best -> (B+E / B+F / G)`。

---

## 1. 阶段化执行（带验收门槛）

## Phase 0：统一实验设置（0.5-1 天）

### 目标
建立可比实验基线，避免后续结果不可比。

### 任务
- 数据集整理：Wild、bmx-trees、tennis、可选 DAVIS。
- 统一预处理：分辨率、fps、帧命名、mask 格式。  
- 统一评价协议：JM/JR + ROS/TCF/BES、可视化样式。
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
- **状态**：PASS（2026-04-30）
- Exp ID：`phase1_20260430_*`（`check_phase1`: PASS）
- 3 个 mandatory 数据集均跑通 A-best（YOLO+光流筛选+cv2.inpaint）。
- A-best 指标可复现，输出视频与可视化齐全。

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
- **状态**：PASS（2026-04-30）
- Exp ID：`phase2_20260430_175248`（`check_phase2 --strict-dual-run false`: PASS）
- B-best backend：`sam2`，传播策略：`bidirectional_no_wrap`
- Aggregate：`JM=0.6647`，`JR=0.7688`，`TCF=0.0600`
- 3 个 mandatory 数据集完整视频输出齐全，B-best 在视觉质量和关键指标上优于 A-best。

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
| E4 | 增益验证 | JM/JR/TCF | 0.25 天 | 给出量化提升 |

### 输出
- `B vs B+E` 对比图与增益表。
- 边界区域可视化专题图。

### 验收门槛
- 至少一个 mandatory 数据集上出现可解释的提升。

### 验收结果
- **状态**：PASS（2026-05-01）
- Exp ID：`phase3_sam3_multianchor_20260501_012933`（`check_phase3 --strict-sam3-permission true`: PASS）
- E3 SAM3 multi-anchor refiner 与 SAM2 提示词策略完全统一（multi-anchor、gap-spaced、bidirectional_no_wrap）。
- Aggregate（B+E+SAM3）：`JM=0.6782`，`JR=0.7812`，`TCF=0.0609`
- B→E+SAM3 delta：`delta_JM=+0.0135`，`delta_JR=+0.0124`，`delta_TCF=+0.0009`
- bmx-trees 和 tennis 上 JM/JR 均有可解释提升；wild 无 GT mask 故 JM/JR 为 None。

---

## Phase 4：路线 F（研究探索：VGGT4D prior 对照，2-3 天）

**状态（2026-04-30）：已完成开发与全量验收。**
- 全量实验：`phase4_bidir_full_20260430_162056`
- Gate：`scripts/check_phase4.sh` 通过
- 最终导出：`F1 / bbest_vggt4d_replace_yolo`（`phase4_final_policy=force_vggt4d_prior`）
- 关联 Phase2：`phase2_bidir_full_20260430_150340`（全局 B-best backend=`sam2`）

`VGGT4D raw mask -> B-best mask backend + bidirectional no-wrap propagation = VGGT4D prior -> ProPainter`

### 口径约束
- YOLO-only、VGGT4D raw、VGGT4D+YOLO 与相关先验融合都只是探索/消融实验，只需要诚实产生对比结果，不作为路线 F 的最终产物。
- 路线 F 最终导出的视频、指标与报告主结果必须来自 `VGGT4D prior`。
- `VGGT4D prior` 不是直接使用 VGGT4D 原始 mask；它指 VGGT4D 输出先转为 prompt anchors，再经过 B-best 实际使用的全局 mask backend 与 `bidirectional_no_wrap` 传播策略后的结果。
- 首尾不连续视频按普通非循环视频处理；SAM2/Track Anything 禁止把尾帧记忆直接接到首帧。

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| F1 | VGGT4D prior 主结果 | chunk_size / frame chunk strategy / B-best mask backend | 0.5 天 | 生成路线 F 最终视频与指标 |
| F2 | B-best 基线复现实验 | B-best 固定配置 | 0.25 天 | 作为 F 路线对照基线 |
| F3 | mask prior 对照 | YOLO-only / VGGT4D raw / VGGT4D prior / VGGT4D+YOLO | 0.5 天 | 诚实比较先验质量与失败模式 |
| F4 | 先验融合探索 | vggt4d_guided / weighted / intersection / union / motion filtering | 0.5-1 天 | 仅作为消融，分析是否带来增益或副作用 |
| F5 | 难例对比 | 遮挡/多目标/静态人车 | 0.25 天 | 解释 VGGT4D prior 的适用边界 |

### 输出
- `F-final = B+F(VGGT4D prior)` 的视频、指标与报告主结果。
- `YOLO-only vs VGGT4D raw vs VGGT4D prior vs VGGT4D+YOLO/fusion` 对比结果（mask 级/消融级）。
- `B-best vs F-final(VGGT4D prior)` 对比结果（视频级）。
- 难场景 case study（重点讲“该不该删”的判断改进）。

### 验收门槛
- 3 个 mandatory 数据集均有 `VGGT4D prior` 版本视频与指标。
- YOLO、VGGT4D raw、融合结果只进入消融表和失败分析；若没有优于 B-best，也按真实结果报告，不替换 F-final。
- 若 `VGGT4D prior` 生成失败，路线 F 标记为失败并记录原因，禁止改用 YOLO 或融合结果冒充路线 F 最终产物。

### 验收结果
- **状态**：PASS（2026-04-30）
- Exp ID：`phase4_20260430_183904`（`check_phase4`: PASS）
- F-best：`F1 / VGGT4D prior`，关联 Phase2：`phase2_20260430_175248`（B-best backend=`sam2`）
- Aggregate（VGGT4D prior）：`JM=0.7074`，`JR=0.7688`，`TCF=0.0639`
- B→F delta：`delta_JM=+0.0427`，`delta_JR=0.0`，`delta_TCF=+0.0039`
- 3 个 mandatory 数据集均有 VGGT4D prior 版本视频与指标；JM 显著提升，TCF 轻微上升属正常范围。

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
- `same-mask ProPainter vs G-low/G-mid/G-high` 指标与可视化对比。
- `framewise diffusion vs keyframe diffusion vs hybrid refinement` 时序对比。
- “背景从未出现”专题案例与连续帧 strip。
- G 路线失败案例包（闪烁、风格漂移、结构幻觉、边界接缝）。

### 验收门槛
- 至少在一个”不可借像素”场景上相对 ProPainter 有明显视觉收益。
- 同时必须报告 TCF/BES 或连续帧可视化中的时序代价；若 TCF 上升但纹理真实性提升，也按真实 trade-off 写入报告。

### 验收结果
- **状态**：PASS（2026-05-01）
- Exp ID：`phase5_20260501_153130`（`check_phase5`: PASS）
- G-best variant：`G-high`（mask_dilation=20, denoise_strength=0.75, keyframe_interval=1）
- Model：`stable-diffusion-v1-5/stable-diffusion-inpainting`，device=cuda（RTX 4080 SUPER）
- Aggregate（G-high）：`JM=0.6782`，`JR=0.7813`，`TCF=0.0722`
- B→G delta：`delta_JM=+0.0135`，`delta_JR=+0.0125`，`delta_TCF=+0.0122`
- 消融结果：G-low TCF=0.0610，G-mid=0.0526，G-high=0.0722，G-hybrid=0.0511
- 修复 bug（diffusion 输入改为 ProPainter 补全帧而非原始帧）后，TCF 相对 B-best 轻微上升（+0.0122）；G-high 在大 mask dilation 下纹理生成质量最优。
- 3 个 mandatory 数据集均有完整视频输出与指标。

---

## 2. 最小必做实验集 vs 加分实验集

## 2.1 最小必做（保证可交付）
- A 路线完整跑通并给出 A-best。
- B 路线完整跑通并给出 B-best。
- 3 个 mandatory 数据集均有输出视频。
- 表 1（总体性能）+ 关键可视化 + failure cases。

## 2.2 加分建议（时间充足）
- E 路线完整增益验证。
- F 或 G 至少完成一条并形成清晰研究结论。
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
| Day 11-12 | Phase 5（G 路线） |
| Day 13 | 指标汇总、可视化整理、failure cases 归档 |
| Day 14 | 写实验与分析章节，补齐图表与结论链 |

> 若时间不足：优先保证 Day 1-6 + Day 13-14 完整，E/F/G 只保留 1 条最有把握路线。

---

## 4. 结果表与图（建议最终最少产出）

## 表 1：总体性能主表（必须）
| Method | Mask Source | Inpainting | JM↑ | JR↑ | TCF↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A-best | YOLO+Flow | OpenCV |  |  |  |
| B-best | SAM2/TA | ProPainter |  |  |  |
| B+E | refined mask | ProPainter |  |  |  |
| B+F | VGGT4D prior | ProPainter |  |  |  |
| G | refined mask | Diffusion |  |  |  |

## 表 2：A 路线消融（建议）
| Setting | Dynamic Filter | Dilation | Inpaint Algo | ROS | TCF | BES |
| --- | --- | --- | --- | ---: | ---: | ---: |

## 表 3：B/E/F 消融（建议）
| Setting | Temporal Smoothing | Motion Prior | Refinement | JM | JR | TCF |
| --- | --- | --- | --- | ---: | ---: | ---: |

## 表 4：成本分析（建议）
| Method | Time/frame | GPU Mem | Resolution |
| --- | ---: | ---: | --- |

## 图（至少 4 组）
- A-best vs B-best 全局对比图。
- B vs B+E 边界细节图。
- B vs B+F(VGGT4D prior) 难场景图（遮挡/多目标）。
- B vs G “背景未出现”专题图。

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
