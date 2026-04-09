# AGENTS.md

本文件定义本仓库（AIAA3201 Project 3: Video Object Removal & Inpainting）的统一开发规范。  
所有人类开发者与 AI Agent 在本仓库内协作时，都应遵循本规范。

## 1. 项目目标与交付边界

### 1.1 核心任务
- 输入：包含动态目标（行人、车辆等）的视频。
- 输出：自动识别并移除动态目标，并利用时序信息恢复干净背景。

### 1.2 必交付项（硬约束）
- 公共 GitHub 仓库代码与可运行 README。
- CVPR 模板论文 PDF（正文 6-8 页，参考文献不计入）。
- arXiv 上传版本（camera-ready）。
- `videos.zip`，且必须包含 3 个 mandatory 数据集的处理结果：
  - Wild video
  - `bmx-trees`
  - `tennis`

### 1.3 指标（硬约束）
- Mask 质量：JM（IoU mean）、JR（IoU recall）。
- 视频移除质量：ROS、TCF、BES、Q_REMOVE。
- 定性结果：多方法可视化对比图与失败案例。

## 2. 技术路线优先级

以 [PLAN.md](/home/jun/AIAA3201_final/PLAN.md) 为主计划，默认执行顺序：
1. Phase 0：统一实验设置。
2. Phase 1：A 路线（经典 baseline）。
3. Phase 2：B 路线（SAM2/TrackAnything + ProPainter）作为主线结果。
4. Phase 3-5：E/F/G 路线（优化与扩展）。

规则：
- 先保证 A 与 B 跑通并可复现，再做 E/F/G。
- 任何探索路线失败也必须保留失败案例与原因分析，禁止“静默丢弃”。

## 3. 仓库目录规范

推荐目录（如不存在可按此创建）：

```text
.
├── AGENTS.md
├── PLAN.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── gt/
├── src/
│   ├── common/
│   ├── part1/
│   ├── part2/
│   └── part3/
├── outputs/
│   ├── masks/
│   ├── videos/
│   ├── figures/
│   ├── metrics/
│   └── logs/
└── docs/
```

约束：
- 原始数据只放 `data/raw/`，不得覆盖。
- 所有中间与最终结果仅写入 `outputs/`。
- 代码只放 `src/`，禁止把临时代码散落在根目录。

## 4. 代码开发规范

### 4.1 基础要求
- 默认语言：Python。
- 所有可执行脚本必须支持 `--help`。
- 路径必须参数化，禁止硬编码绝对路径。
- 随机过程必须显式设置 seed，并在日志中记录。

### 4.2 可复现要求
- 每次实验至少保存：
  - 配置（yaml/json/命令行参数）
  - 指标结果（json/csv）
  - 代表性可视化（帧图或短视频）
- 命名中包含：`method + dataset + timestamp`。

### 4.3 质量要求
- 新增核心逻辑时，至少补一个最小验证脚本或单元测试。
- 不允许提交无法运行的占位主流程（除明确标记为 WIP 且不在主分支交付范围内）。

## 5. 实验协议规范

### 5.1 输入统一
- 对比实验必须统一分辨率、帧率、mask 编码格式与命名规则。
- 若某方法需要额外预处理，必须在日志中写清楚，不得隐式处理。

### 5.2 评估统一
- JM/JR 计算逻辑在所有路线中保持一致。
- 质量评估统一使用 ROS/TCF/BES，并由 Q_REMOVE 汇总。
- 表格中的数值必须可追溯到 `outputs/metrics/` 原始文件。

### 5.3 可视化统一
- 每条路线至少包含：
  - 输入帧
  - mask 可视化
  - 修复结果
  - 与基线或主线对比图
- 失败案例单独归档，并附简短解释。

## 6. AI Agent 执行规范

AI Agent 在执行任务时必须：
1. 先阅读本文件与 [PLAN.md](/home/jun/AIAA3201_final/PLAN.md)。
2. 默认优先推进当前 phase 的阻塞项，不跳过 mandatory 数据集。
3. 修改代码前先说明将改哪些文件与目的。
4. 修改后给出：
   - 改动文件列表
   - 运行命令
   - 结果摘要
   - 未完成项与风险
5. 不得删除或覆盖他人结果文件；清理动作仅限本次任务生成的临时文件。

## 7. 提交与记录规范

### 7.1 提交信息
- 建议格式：`<type>(<scope>): <summary>`
- `type` 建议值：`feat` `fix` `exp` `docs` `refactor` `test`。

示例：
- `exp(part2): compare sam2 and track-anything masks on tennis`
- `fix(eval): align JM/JR threshold handling`

### 7.2 实验记录
- 每次关键实验在 `docs/` 或固定实验日志中记录：
  - 目标
  - 变量与配置
  - 核心结果
  - 现象与结论

## 8. 合并前检查清单（DoD）

在声明“完成”前，必须满足：
- 代码可运行，命令可复现。
- mandatory 数据集结果完整可追溯。
- 指标文件、可视化与文字结论一致。
- README/AGENT等文档更新到位。
- 若是阶段性结果，明确下一步与风险。

## 9. 冲突处理原则

当本文件与临时任务要求冲突时，优先级如下：
1. 课程硬性要求（项目 PDF）
2. 本文件（AGENTS.md）
3. 临时任务偏好

如有不确定，先保守执行：优先可复现、可解释、可交付。

## 10. 环境基线规范

- 默认 conda 环境名称：`aiaa3201`。
- 默认 Python 版本：`3.10.x`。
- 环境入口文件：`environment.yml`。
- GPU 依赖策略：PyTorch 使用官方 `cu121` wheel 索引安装，不在 conda 中一次性求解重型 CUDA 栈。

### 10.1 分阶段安装要求（同一环境）

- Stage A（必须）：
  - 执行 `pip install -r requirements.txt`。
  - `requirements.txt` 中核心依赖采用强锁定版本。
- Stage B（按需，SOTA 主线）：
  - 安装 `torch/torchvision/torchaudio`（cu121）。
  - 安装视频修复主线所需扩展依赖（推荐版本区间）。
- Stage C（按需，Route G）：
  - 仅在做 diffusion 相关实验时安装 `diffusers/transformers/accelerate/xformers`。

## 11. 新增依赖流程

新增依赖时必须按以下顺序执行：
1. 在实验记录中写明用途（对应路线/模块）与版本策略（强锁定或推荐区间）。
2. 更新 `requirements.txt` 或 README 的 Stage B/Stage C 依赖段。
3. 补充验证命令（`import` 或最小运行命令）。
4. 在提交说明中标注依赖变更影响范围。

## 12. 复现实验前检查清单（环境）

- `conda env list | grep aiaa3201` 输出存在。
- `python -V` 为 `3.10.x`。
- Stage A 校验：
  - `python -c "import numpy, cv2, yaml, skimage, matplotlib; print('core ok')"`
- Stage B 校验（若启用）：
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- 记录当前依赖快照（建议）：
  - `pip freeze > outputs/logs/pip_freeze_<date>.txt`
