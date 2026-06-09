# BJTU NetContSecurity Project LHCGC 2026

本仓库用于课程项目“基于多模态社交网络的内容风控与异常检测”。目标是结合文本内容、图像/OCR 信息和用户行为特征，识别垃圾广告、诈骗信息、异常账号或其他有害内容。

当前仓库是 public。提交前必须确认没有真实数据、密钥、cookie、账号信息、未脱敏样本或大体积模型文件。

工程分工和主流程见 [Architecture.mmd](Architecture.mmd)。

## Quick Start

项目使用 `uv` 管理 Python 环境，默认 Python 版本为 3.11。

```powershell
uv sync
uv run python -c "import torch, transformers, pandas, sklearn; print('env ok')"
```

常用依赖已经写入 `pyproject.toml` 并锁定在 `uv.lock` 中，包括 PyTorch、Transformers、Datasets、Scikit-learn、Pandas、OpenCV、Matplotlib、Seaborn 和 Streamlit。

## Repository Guide

```text
.
├── Architecture.mmd
├── document/
│   ├── 网络内容安全任务概述.md
│   ├── [Google-Gemini]网络安全研究分工参考.md
│   ├── 人员分工参考Team-Responsibility-Reference.md
│   ├── 项目入手指南.md
│   ├── 开发组角色选择说明.md             ← 新增：A/B/C怎么选角色
│   ├── 开发组协作指南.md                 ← 新增：A/B/C如何配合
│   ├── 开发组协作流程.mmd                ← VS Code预览：甘特图
│   ├── 开发组协作流程-流程图.mmd          ← VS Code预览：流程图
│   ├── 开发组协作流程.md                 ← 完整版：4幅图+说明
│   ├── 成员A数据与文本模型指南Member-A-Data-Text-Guide.md
│   ├── 成员B图像与多模态融合指南Member-B-Image-Fusion-Guide.md
│   ├── 成员C系统集成与评测指南Member-C-Integration-Evaluation-Guide.md
│   ├── 成员D报告撰写指南Member-D-Report-Writing-Guide.md
│   └── 成员EPPT与答辩指南Member-E-Slides-Defense-Guide.md
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

后续开发建议新增但不要提交本地内容的目录：

```text
data/raw/          # 原始公开数据或爬虫输出，本地保存
data/processed/    # 清洗后的训练数据，本地保存
models/            # 本地模型权重
outputs/           # 指标、图表、预测结果
src/               # 数据处理、训练、推理、评估代码
```

这些数据、模型和输出目录已经在 `.gitignore` 中排除。

## Team Workflow

推荐使用“3 人开发组 + 2 人报告答辩组”的流水线：

- 成员 A：数据与文本模型。
- 成员 B：图像模型与多模态融合。
- 成员 C：系统集成与评测。
- 成员 D：报告总编与排版。
- 成员 E：PPT、讲稿和答辩。

详细分工见 [document/人员分工参考Team-Responsibility-Reference.md](document/人员分工参考Team-Responsibility-Reference.md)。

开发组三人协作（接口约定、依赖链、每天谁等谁）见 **[document/开发组协作指南.md](document/开发组协作指南.md)**，协作流程图见 **[document/开发组协作流程.mmd](document/开发组协作流程.mmd)**。

详细技术路线见 [document/项目入手指南.md](document/项目入手指南.md)。

个人落地指南：

- [成员 A 数据与文本模型指南](document/成员A数据与文本模型指南Member-A-Data-Text-Guide.md)
- [成员 B 图像与多模态融合指南](document/成员B图像与多模态融合指南Member-B-Image-Fusion-Guide.md)
- [成员 C 系统集成与评测指南](document/成员C系统集成与评测指南Member-C-Integration-Evaluation-Guide.md)
- [成员 D 报告撰写指南](document/成员D报告撰写指南Member-D-Report-Writing-Guide.md)
- [成员 E PPT 与答辩指南](document/成员EPPT与答辩指南Member-E-Slides-Defense-Guide.md)

## Public Repository Rules

不要提交：

- `.env`、API token、cookie、平台账号、私有 URL。
- 未脱敏原始数据、爬虫原始 HTML、二维码、手机号、邮箱、头像、主页链接。
- 训练 checkpoint、模型权重、日志缓存、大体积数据集。
- 老师、同学或平台用户的个人隐私信息。

可以提交：

- 代码、配置模板、脱敏样例和字段说明。
- 小体积实验图表、报告草稿、PPT 素材。
- `.env.example` 这类只含占位符的示例文件。

## Development Notes

建议每个功能分支只做一类事情，例如：

```text
feature/data-cleaning
feature/text-baseline
feature/multimodal-fusion
docs/report-outline
docs/defense-slides
```

每次提交前检查：

```powershell
git status --short
uv run python -c "import torch, transformers, pandas, sklearn; print('env ok')"
```

如果后续增加代码，建议把“数据处理、训练、推理、评估”拆成独立脚本，并把复现实验命令写入 `document/项目入手指南.md` 或报告附录。
