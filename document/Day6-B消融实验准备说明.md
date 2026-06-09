# Day6 B 消融实验准备说明

适用角色：成员 B  
日期：2026-06-09  
目标：在 Day5 融合闭环基础上，产出 Day6 消融实验预测和指标汇总，交给成员 C 画图出表。

## 1. 当前基础

Day5 已经跑通保底融合：

```text
text_tfidf_pred.csv 的 risk_prob
+ image_embeddings.csv 的 512 维 ResNet 图像向量
+ behavior_embeddings.csv 的 13 维行为向量
-> fusion_pred.csv
```

当前可用输入：

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `outputs/predictions/text_tfidf_pred.csv` | 已生成 | 文本 TF-IDF 风险概率，作为文本保底特征 |
| `outputs/predictions/image_embeddings.csv` | 已生成 | 图像 ResNet18 512 维向量 |
| `outputs/predictions/behavior_embeddings.csv` | 已生成 | 行为 13 维标准化向量 |
| `outputs/predictions/fusion_pred.csv` | 已生成 | Day5 三模态融合预测 |

注意：`image_resnet_pred.csv` 只是 Day4 接口占位，`risk_prob=0.5`，不作为正式图像消融指标。Day6 的图像消融使用 `image_embeddings.csv` 重新训练分类器。

## 2. 新增脚本

本次新增：

```text
src/run_ablation.py
```

同时扩展：

```text
src/train_fusion.py
```

新增参数：

```text
--feature-groups text,image,behavior
--model-name fusion_v1
```

默认行为仍是三模态融合，不影响 Day5 命令。

## 3. 消融组合

`run_ablation.py` 会生成 5 组预测：

| 组合 | 使用特征 | 输出文件 |
| --- | --- | --- |
| `text_only` | 文本概率或文本向量 | `outputs/predictions/ablation/text_only_pred.csv` |
| `image_only` | 图像向量 | `outputs/predictions/ablation/image_only_pred.csv` |
| `behavior_only` | 行为向量 | `outputs/predictions/ablation/behavior_only_pred.csv` |
| `text_image` | 文本 + 图像 | `outputs/predictions/ablation/text_image_pred.csv` |
| `text_image_behavior` | 文本 + 图像 + 行为 | `outputs/predictions/ablation/text_image_behavior_pred.csv` |

每个预测 CSV 都保持 Day1 统一 5 列：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
```

## 4. 运行命令

当前本地保底消融：

> 注意：`tmp\2026年6月9日\` 是 czl 本机临时目录，其他同学默认没有。下面命令用于记录本机验收过程；团队复现时请把 `$base` 改成自己的 A 数据目录，或使用 `data/processed/dataset_v1.csv`。

```powershell
$base = Resolve-Path 'tmp\2026年6月9日'

python src\run_ablation.py `
  --dataset "$base\processed\dataset_v1.csv" `
  --mode scores
```

如果已经按项目规范准备了 `data/processed/`，可使用：

```powershell
python src\run_ablation.py `
  --dataset data\processed\dataset_v1.csv `
  --mode scores
```

如果 A 后续提供 `text_embeddings.csv`，可改为：

```powershell
python src\run_ablation.py `
  --dataset data\processed\dataset_v1.csv `
  --mode embeddings
```

## 5. 交给 C 的文件

Day6 B 交付给 C：

| 文件 | 用途 |
| --- | --- |
| `outputs/predictions/ablation/*_pred.csv` | 五组消融预测 |
| `outputs/metrics/ablation_summary.csv` | 消融指标汇总表 |
| `outputs/metrics/ablation_summary.json` | 消融指标机器可读版本 |

以上文件均在 `outputs/`，默认不提交 Git。C 可以用这些文件画：

- 消融指标柱状图
- 混淆矩阵
- 文本/图像/行为/融合对比表

## 6. 验收标准

- 5 个消融预测 CSV 均为 7723 行、统一 5 列。
- `ablation_summary.csv` 至少有 5 行。
- 每组指标都基于同一批 `split=test`。
- 输出文件不进入 Git，只提交脚本和说明文档。

## 7. 本次本地验收结果

当前已用 `--mode scores` 跑通 5 组保底消融。文本侧使用 `text_tfidf_pred.csv` 的 `risk_prob`，图像侧使用 512 维 ResNet 向量，行为侧使用 13 维行为向量。

| 组合 | 特征维度 | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| text_only | 1 | 0.7509 | 0.8811 | 0.5979 | 0.7124 |
| image_only | 512 | 0.6253 | 0.6279 | 0.6720 | 0.6492 |
| behavior_only | 13 | 0.9727 | 0.9958 | 0.9511 | 0.9729 |
| text_image | 513 | 0.7918 | 0.8282 | 0.7526 | 0.7886 |
| text_image_behavior | 526 | 0.9481 | 0.9658 | 0.9325 | 0.9489 |

解读给 C/D/E 时需要注意：

- `image_only` 是基于图像向量重新训练分类器，不是 Day4 的 `risk_prob=0.5` 占位预测。
- 当前 `behavior_only` 指标很高，需要在报告里谨慎说明：行为特征可能非常强，也可能包含数据集分布特征；Day6/C 评估时应重点检查是否存在潜在数据泄漏或 split 分布偏差。
- `text_image_behavior` 不是最佳单项指标，但它代表完整三模态流程，仍应作为最终融合模型保留。
