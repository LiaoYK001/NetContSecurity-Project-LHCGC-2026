# Day7 交接包说明

本文是 public-safe 版本，说明开发组如何把本地结果交给 D/E。完整预测文件、模型权重、真实数据、错误案例和本地交接材料保存在本地 `outputs/` 和 `data/processed/`，默认不提交 GitHub。经人工确认不含敏感样本内容的汇总指标和图表可以放到 `outputs/demo/` 公开展示。

## 1. 当前本地数据口径

最终结论使用 `data/processed/` 中的 processed 数据：

```text
dataset_v1.csv
text_samples.csv
behavior_features.csv
```

当前本地最终口径为 7723 条样本，`dataset_v1.csv`、`text_samples.csv`、`behavior_features.csv` 的 `sample_id` 均唯一且可一一对齐。如果这些文件还在 `tmp/processed/processed/`，需要先复制到 `data/processed/`，再运行训练和评估脚本。

`all_images.csv` 是多图展开表，同一个 `sample_id` 可以出现多行，不能作为融合主表。融合与评估必须使用 `dataset_v1.csv`。

## 2. 给 D/E 的本地交接包

本地交接包建议放在：

```text
outputs/handoff/
```

至少包含：

```text
README_Day7交接包.md
复现命令.md
数据说明.md
模型说明.md
最终指标表.md
图表清单.md
错误案例说明.md
ABC技术解释.md
```

这些文件默认被 `.gitignore` 排除，避免把真实数据、错误案例、预测明细或本地交接材料误提交到 public 仓库。

如果 D/E 需要的是“可以直接写报告和做 PPT 的 public-safe 交接材料”，优先阅读：

```text
document/DE交接区/README.md
```

`outputs/handoff/` 是本地开发交接区，适合 A/B/C 和本机协作；`document/DE交接区/` 是公开展示和报告/PPT 写作入口，适合 D/E、老师和评分同学。

## 3. 可公开 Demo 结果

经人工确认不含原始文本、真实样本 ID、用户链接、cookie、token、预测明细或完整 processed 数据的结果，可以放在：

```text
outputs/demo/
```

当前 public-safe demo 包含：

```text
ablation_metrics.csv
ablation_summary.csv
ablation_f1_bar.png
roc_all_compare.png
cm_text_only.png
cm_image_only.png
cm_behavior_only.png
cm_text_image.png
cm_text_image_behavior.png
README.md
```

这些文件可用于向老师和同学展示 Weibo 示例实验结果。它们是展示用静态结果，不代表别人 clone 仓库后无需下载数据即可复现最终指标。

## 4. 当前最终结果摘要

最终评估只使用 `split=test` 的 1465 条样本。五组消融结果如下：

| 模型组 | F1 | ROC AUC | 说明 |
| --- | ---: | ---: | --- |
| `behavior_only` | 0.9729 | 0.9900 | 指标最高，但要说明可能存在数据分布偏差或潜在泄漏风险 |
| `text_image_behavior` | 0.9640 | 0.9923 | 推荐作为最终完整三模态工程方案 |
| `text_image` | 0.8742 | 0.9416 | 比纯文本略有提升 |
| `text_only` | 0.8724 | 0.9366 | 文本语义本身已有稳定识别能力 |
| `image_only` | 0.6501 | 0.6666 | 图片单独较弱，适合作辅助信号 |

建议 D/E 在报告和 PPT 中把 `text_image_behavior` 作为最终方案展示，把 `behavior_only` 的高指标作为消融实验发现和局限性讨论。

## 5. D/E 应引用哪些材料

可以直接引用：

1. 数据字段和隐私合规说明。
2. TF-IDF、BERT、ResNet、行为特征和融合模型说明。
3. 消融实验设计。
4. 错误案例分析框架。
5. A/B/C 技术解释。

本地可引用但不要直接提交 GitHub 的材料：

1. 最终指标表。
2. F1 柱状图。
3. ROC 曲线。
4. 混淆矩阵。
5. 真实但已脱敏的错误案例。

其中，前 4 类如果只包含汇总指标和图表，可以同步到 `outputs/demo/` 公开展示；错误案例必须先人工脱敏或改写成合成案例。

## 6. 最终复现命令

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

验收通过标准：

1. `verify_all.py` 退出码为 0。
2. `outputs/metrics/ablation_metrics.csv` 有 5 个模型组。
3. `outputs/figures/` 有 F1 柱状图、ROC 图和 5 张混淆矩阵。
4. `git status --short` 不出现真实数据、模型权重、预测明细或错误案例。

如果复现机器没有 NVIDIA CUDA，把 `--device cuda` 改成 `--device auto` 或 `--device cpu`；模型权重允许从 Hugging Face / PyTorch 官方源下载，但缓存不进入 Git。
