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
- 指标完整：JM、JR、PSNR、SSIM（有 GT 的实验）。
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
- 统一评价协议：JM/JR、PSNR/SSIM、可视化样式。
- 建立实验记录模板。

### 输出
- 标准目录结构与命名规范。
- 统一评估脚本接口设计。
- 实验记录模板（参数、耗时、显存、结果、现象）。

### 验收门槛（Go/No-Go）
- 任意方法输出都能进入同一评估脚本。
- 同一视频可稳定复现实验结果。

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

---

## Phase 2：路线 B（主线方法，2.5-3.5 天）

`SAM2 / Track Anything -> ProPainter`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| B1 | 动态 mask 方案对比 | SAM2 vs Track Anything | 0.5-1 天 | 选主分割方案 |
| B2 | ProPainter 接入 | 固定 mask 输入 | 0.5 天 | 跑通主链路 |
| B3 | 上游质量影响 | 粗 mask vs 精 mask | 0.5 天 | 证明 mask 重要性 |
| B4 | 推理参数实验 | clip length / neighbor frames | 0.5 天 | 找稳定配置 |
| B5 | mandatory 全量评估 | Wild / bmx / tennis | 0.5-1 天 | 形成主结果表 |

### 输出
- `B-best`（主方法）。
- `A-best vs B-best` 定量+定性对比。

### 验收门槛
- B-best 在多数场景优于 A-best（至少在视觉质量和关键指标上）。
- 3 个 mandatory 数据集完整视频输出齐全。

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
| E4 | 增益验证 | IoU/PSNR/SSIM | 0.25 天 | 给出量化提升 |

### 输出
- `B vs B+E` 对比图与增益表。
- 边界区域可视化专题图。

### 验收门槛
- 至少一个 mandatory 数据集上出现可解释的提升。

---

## Phase 4：路线 F（研究探索：运动/几何增强，2-3 天）

`motion/geometric cue + semantic mask`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| F1 | motion prior 生成 | 稀疏/稠密光流 | 0.5 天 | 获得动态先验 |
| F2 | 与 SAM 融合 | 交集/并集/加权 | 0.5 天 | 提升动态判别可信度 |
| F3 | 轨迹一致性过滤 | threshold/track length | 0.5 天 | 减少静态误删 |
| F4 | 可选高级线索 | VGGT4D cue | 0.5-1 天 | 增强复杂场景稳定性 |
| F5 | 难例对比 | 遮挡/多目标场景 | 0.25 天 | 给出 F 的优势证据 |

### 输出
- `B vs B+F` 对比结果。
- 难场景 case study（重点讲“该不该删”的判断改进）。

### 验收门槛
- 至少在静态误删或复杂运动场景上优于 B。

---

## Phase 5：路线 G（扩散生成修复，2-3 天）

`mask -> diffusion inpainting / ControlNet -> 时序传播`

### 子实验
| 编号 | 内容 | 变量 | 预计耗时 | 目标 |
| --- | --- | --- | ---: | --- |
| G1 | 单帧扩散修复 | prompt/steps/mask size | 0.5-1 天 | 生成合理背景 |
| G2 | 关键帧策略 | 每 N 帧修复 | 0.5 天 | 降低算力成本 |
| G3 | 关键帧传播 | 光流/插值/传播 | 0.5-1 天 | 提升时序一致性 |
| G4 | 与 ProPainter 对比 | 背景未出现场景 | 0.5 天 | 展示生成式优势 |
| G5 | 失败分析 | 风格漂移/闪烁 | 0.25 天 | 诚实分析局限 |

### 输出
- `B vs G` 对比结果。
- “背景从未出现”专题案例。

### 验收门槛
- 至少在一个“不可借像素”场景上有明显视觉收益。

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
| Method | Mask Source | Inpainting | JM↑ | JR↑ | PSNR↑ | SSIM↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A-best | YOLO+Flow | OpenCV |  |  |  |  |
| B-best | SAM2/TA | ProPainter |  |  |  |  |
| B+E | refined mask | ProPainter |  |  |  |  |
| B+F | mask+motion cue | ProPainter |  |  |  |  |
| G | refined mask | Diffusion |  |  |  |  |

## 表 2：A 路线消融（建议）
| Setting | Dynamic Filter | Dilation | Inpaint Algo | PSNR | SSIM |
| --- | --- | --- | --- | ---: | ---: |

## 表 3：B/E/F 消融（建议）
| Setting | Temporal Smoothing | Motion Prior | Refinement | JM | JR | PSNR |
| --- | --- | --- | --- | ---: | ---: | ---: |

## 表 4：成本分析（建议）
| Method | Time/frame | GPU Mem | Resolution |
| --- | ---: | ---: | --- |

## 图（至少 4 组）
- A-best vs B-best 全局对比图。
- B vs B+E 边界细节图。
- B vs B+F 难场景图（遮挡/多目标）。
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