# BJTU NetContSecurity Project LHCGC 2026

本仓库用于课程项目“基于多模态社交网络的内容风控与异常检测”。目标是结合文本内容、图像/OCR 信息和用户行为特征，识别垃圾广告、诈骗信息、异常账号或其他有害内容。

当前仓库是 public。提交前必须确认没有真实数据、密钥、cookie、账号信息、未脱敏样本或大体积模型文件。

工程分工和主流程见 [Architecture.mmd](Architecture.mmd)。

报告和 PPT 组优先阅读 [document/DE交接区/README.md](document/DE交接区/README.md)。这个目录是 public-safe 的最终交接区，成员 D/E 不读完整仓库也能理解项目原理、流程、图表和写作主线。

## Quick Start

项目使用 `uv` 管理 Python 环境，默认 Python 版本为 3.11。

```powershell
uv sync
uv run python -c "import torch, transformers, pandas, sklearn; print('env ok')"
```

常用依赖已经写入 `pyproject.toml` 并锁定在 `uv.lock` 中，包括 PyTorch、Transformers、Datasets、Scikit-learn、Pandas、OpenCV、Matplotlib、Seaborn 和 Streamlit。

## Reproduce Locally

本仓库提供的是“可复现流程”，不是“零数据直接复现最终指标”。完整数据、图片、模型权重、模型缓存和实验输出不进入 public 仓库。别人 clone 后需要先下载同源公开数据集，或使用课程组私下分发的 processed 数据，才能复现最终指标。

默认推荐数据集是 EANN-KDD18 Weibo 多模态谣言检测数据集：

- 论文：Wang et al., *EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection*, KDD 2018
- 数据入口：[https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)

下载或解压后，把 Weibo 数据放到本地 ignored 目录，例如：

```text
tmp/weibo/
├── tweets/
│   ├── train_rumor.txt
│   ├── train_nonrumor.txt
│   ├── test_rumor.txt
│   └── test_nonrumor.txt
├── train_id.pickle
├── validate_id.pickle
├── test_id.pickle
├── rumor_images/
└── nonrumor_images/
```

`tmp/` 已被 `.gitignore` 排除。确认目录结构后，生成项目需要的 processed CSV：

```powershell
uv run python src/prepare_data.py --weibo-root tmp/weibo --output-dir data/processed --handoff-dir outputs/handoff
```

生成后应至少得到：

```text
data/processed/dataset_v1.csv
data/processed/text_samples.csv
data/processed/behavior_features.csv
data/processed/all_images.csv
data/processed/dataset_stats.json
```

然后按顺序运行最终链路。NVIDIA 显卡机器优先使用 CUDA：

```powershell
uv run python src/train_text_baseline.py
uv run python src/extract_text_features.py --input data/processed/text_samples.csv --model bert-base-chinese --batch-size 32 --max-length 128 --device cuda
uv run python src/extract_resnet_features.py --input data/processed/dataset_v1.csv --image-root tmp --device cuda
uv run python src/extract_behavior_features.py
uv run python src/train_fusion.py --dataset data/processed/dataset_v1.csv --mode embeddings
uv run python src/run_ablation.py --dataset data/processed/dataset_v1.csv --mode embeddings
uv run python src/evaluate.py
uv run python verify_all.py
```

如果复现机器没有 NVIDIA CUDA，把 `--device cuda` 改成 `--device auto` 或 `--device cpu`。如果 BERT 或 ResNet 权重本地没有缓存，首次运行会从 Hugging Face / PyTorch 下载；缓存和权重不要提交 Git。

本项目本地最终指标基于 7723 条 processed 样本。其他人如果使用不同下载版本、不同清洗方式或其他数据集，指标可能不同，但只要转换成同样的 CSV 合同，就可以复现本仓库的数据处理、训练、融合、消融和评估流程。数据下载、字段格式和小样例见 [data/README.md](data/README.md)。

已经人工确认不含敏感样本内容的 Weibo 示例结果放在 [outputs/demo/](outputs/demo/README.md)。这里包含汇总指标和图表，可用于向老师和同学展示项目 demo；完整数据、预测明细和错误案例仍不提交。

## Repository Guide

```text
.
├── Architecture.mmd
├── data/
│   ├── README.md
│   ├── processed/README.md
│   └── sample/                            ← 脱敏字段样例
├── outputs/
│   └── demo/                              ← public-safe 示例指标和图表
├── document/
│   ├── DE交接区/                            ← 报告/PPT组优先阅读的保姆级交接区
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

后续开发建议使用但不要提交完整本地内容的目录：

```text
data/raw/          # 原始公开数据或爬虫输出，本地保存
data/processed/    # 清洗后的训练数据，本地保存
models/            # 本地模型权重
outputs/           # 本地指标、图表、预测结果；仅 demo 子目录可公开
src/               # 数据处理、训练、推理、评估代码
tmp/               # 本地下载的公开数据集或临时数据
```

这些数据、模型和输出目录已经在 `.gitignore` 中排除；只有 README、字段说明、`data/sample/` 脱敏样例和 `outputs/demo/` public-safe 示例结果可提交。

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
- 完整 `tmp/`、完整 `data/processed/*.csv`、完整 `outputs/` 本地结果。
- 样本级预测 CSV、错误案例 CSV、真实微博 ID 或原文片段。

可以提交：

- 代码、配置模板、脱敏样例和字段说明。
- 小体积实验图表、报告草稿、PPT 素材。
- `.env.example` 这类只含占位符的示例文件。
- `data/sample/` 中的小体积脱敏样例。
- `outputs/demo/` 中已人工确认不含敏感样本内容的汇总指标和图表。

## Security Notes

GitHub 可能提示 PyTorch 的 `torch.jit.script` memory corruption 安全告警。当前项目保留 `torch` / `torchvision` 是为了复现 BERT 和 ResNet 特征提取流程，但项目代码没有调用 `torch.jit.script` 或 TorchScript 加载逻辑。该告警目前显示没有 patched version，因此不建议盲目改依赖版本；当前缓解方式是只使用官方预训练模型、不加载不可信权重、不提交模型缓存，并等待 PyTorch 上游修复后再升级。

详细说明见 [SECURITY.md](SECURITY.md)。

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
