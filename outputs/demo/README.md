# Weibo Demo Results

本目录保存一组 public-safe 的 Weibo 示例实验结果，用于给老师、同学或评审快速查看项目效果。

These are public-safe demo results from the Weibo experiment for quick review by instructors, classmates, or evaluators.

报告/PPT 写作说明见 [../../document/DE交接区/README.md](../../document/DE交接区/README.md)。DE 交接区会说明每张图放在哪里、怎么讲、如何解释指标。

## 1. 结果来源

这些结果来自默认 EANN-KDD18 Weibo 流程：

```text
公开 Weibo 数据集 -> prepare_data.py -> 文本 / 图像 / 行为特征 -> 融合与消融 -> evaluate.py
```

本次本地运行口径：

```text
processed 样本数：7723
test 样本数：1465
文本特征：冻结 bert-base-chinese，768 维
图像特征：冻结 ResNet18，512 维
行为特征：13 维结构化特征
```

## 2. 文件清单

| 文件 | 用途 | 是否含敏感样本内容 |
| --- | --- | --- |
| `ablation_metrics.csv` | 五组消融最终指标 | 否 |
| `ablation_summary.csv` | 消融实验维度、指标和输出路径摘要 | 否 |
| `ablation_f1_bar.png` | 五组消融 F1 柱状图 | 否 |
| `roc_all_compare.png` | 五组模型 ROC 曲线 | 否 |
| `cm_text_only.png` | 文本单模态混淆矩阵 | 否 |
| `cm_image_only.png` | 图像单模态混淆矩阵 | 否 |
| `cm_behavior_only.png` | 行为单模态混淆矩阵 | 否 |
| `cm_text_image.png` | 文本+图像混淆矩阵 | 否 |
| `cm_text_image_behavior.png` | 完整三模态混淆矩阵 | 否 |

## 3. 主要结论

| 模型组 | F1 | ROC AUC | 说明 |
| --- | ---: | ---: | --- |
| `behavior_only` | 0.9729 | 0.9900 | 行为特征很强，但要谨慎解释数据分布偏差 |
| `text_image_behavior` | 0.9640 | 0.9923 | 推荐作为最终三模态工程方案 |
| `text_image` | 0.8742 | 0.9416 | 比纯文本略有提升 |
| `text_only` | 0.8724 | 0.9366 | 文本语义已有稳定识别能力 |
| `image_only` | 0.6501 | 0.6666 | 图片单独较弱，适合作辅助信号 |

## 4. 隐私边界

本目录可以公开，因为只包含汇总指标和图表，不包含：

```text
原始文本
真实样本 ID
用户主页链接
原图
预测明细 CSV
cookie / token / 账号密码
模型权重或缓存
完整 processed 数据
```

如果需要复现这些 demo 结果，请按根目录 `README.md` 和 `data/README.md` 下载公开数据集并重新运行流程。不同下载版本或清洗方式可能导致指标略有差异。
