# Day5 B 多模态融合任务说明

适用角色：成员 B  
日期：2026-06-09  
目标：在 Day4 图像向量和行为向量基础上，补齐 Day5 多模态融合入口，输出给成员 C 评估的统一预测 CSV。

## 1. 当前进度

Day4 已完成，B 侧已有：

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `outputs/predictions/image_embeddings.csv` | 已生成 | ResNet18 图像向量，512 维 |
| `outputs/predictions/image_resnet_pred.csv` | 已生成 | 图像分支接口占位预测，`risk_prob=0.5` |
| `outputs/predictions/behavior_embeddings.csv` | 已生成 | 标准化行为向量，13 维 |
| `outputs/predictions/behavior_feature_meta.json` | 已生成 | 行为特征名与 scaler 统计 |

Day5 新增融合入口：

```text
src/train_fusion.py
```

默认输出：

```text
outputs/predictions/fusion_pred.csv
outputs/metrics/fusion_metrics.json
```

以上输出仍位于 `outputs/`，默认不提交 Git。

## 2. 融合策略

标准路线优先使用文本向量：

```text
text_embeddings.csv + image_embeddings.csv + behavior_embeddings.csv
-> 按 sample_id 合并
-> 只在 split=train 上 fit scaler 和训练 LogisticRegression
-> 输出 fusion_pred.csv
```

如果 A 的 `text_embeddings.csv` 暂时没有交付，脚本会使用保底路线：

```text
text_tfidf_pred.csv 的 risk_prob + image_embeddings.csv + behavior_embeddings.csv
-> 按 sample_id 合并
-> 训练融合分类器
-> 输出 fusion_pred.csv
```

保底路线不是最终深度文本融合，但能保证 Day5 闭环先跑通。

## 3. 运行命令

标准路线：

```powershell
uv run python src\train_fusion.py `
  --dataset data\processed\dataset_v1.csv `
  --mode embeddings
```

当前本地临时数据路线：

> 注意：`tmp\2026年6月9日\` 只是 czl 本机临时目录，其他同学默认没有。下面命令用于记录本机验收过程；团队成员复现时应把 `$base` 改成自己的 A 数据目录，或先把 A 的交付文件放到 `data/processed/`。

```powershell
$base = Resolve-Path 'tmp\2026年6月9日'

uv run python src\train_text_baseline.py `
  --input "$base\processed\text_samples.csv" `
  --pred-output outputs\predictions\text_tfidf_pred.csv `
  --metrics-output outputs\metrics\text_tfidf_metrics.json `
  --errors-output outputs\handoff\text_tfidf_error_cases.csv

uv run python src\train_fusion.py `
  --dataset "$base\processed\dataset_v1.csv" `
  --mode auto
```

说明：`--mode auto` 会优先查找 `text_embeddings.csv`；如果不存在，则使用 `text_tfidf_pred.csv` 的 `risk_prob`。

如果已经按项目规范准备了 `data/processed/`，可使用：

```powershell
uv run python src\train_text_baseline.py `
  --input data\processed\text_samples.csv `
  --pred-output outputs\predictions\text_tfidf_pred.csv `
  --metrics-output outputs\metrics\text_tfidf_metrics.json `
  --errors-output outputs\handoff\text_tfidf_error_cases.csv

uv run python src\train_fusion.py `
  --dataset data\processed\dataset_v1.csv `
  --mode auto
```

## 4. 输出接口

`fusion_pred.csv` 固定为 Day1 约定的 5 列：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
```

其中：

- `model_name` 固定为 `fusion_v1`
- `true_label` / `pred_label` 只能是 `normal` 或 `risk`
- `risk_prob` 是融合模型预测为 `risk` 的概率

`fusion_metrics.json` 会记录：

- 文本侧来源：`text_embeddings` 或 `text_tfidf_score`
- 总样本数
- 融合特征维度
- 文本、图像、行为各自维度
- train/val/test 各 split 的 accuracy、precision、recall、F1

## 5. 本次本地验收结果

当前已使用保底路线跑通 Day5：

```text
text_tfidf_pred.csv + image_embeddings.csv + behavior_embeddings.csv
```

输出文件：

| 文件 | 形状 | 说明 |
| --- | --- | --- |
| `outputs/predictions/fusion_pred.csv` | 7723 行，5 列 | 统一预测 CSV，`model_name=fusion_v1` |
| `outputs/metrics/fusion_metrics.json` | JSON | 融合指标与输入来源 |

融合特征维度：

| 模态 | 维度 | 来源 |
| --- | ---: | --- |
| 文本 | 1 | `text_tfidf_pred.csv` 的 `risk_prob` |
| 图像 | 512 | `image_embeddings.csv` 的 `img_emb_*` |
| 行为 | 13 | `behavior_embeddings.csv` 的 `beh_emb_*` |
| 合计 | 526 | 拼接后输入逻辑回归 |

本地 test split 指标：

| Accuracy | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 0.9481 | 0.9658 | 0.9325 | 0.9489 |

注意：这是当前可复现的 Day5 保底闭环。若 A 后续补齐 `text_embeddings.csv`，应再用 `--mode embeddings` 跑一次标准融合，并把两组结果都交给 C 做对比。

## 6. Day6 衔接

Day5 完成后，Day6 可以进入消融与对比：

| 对比项 | 输入 |
| --- | --- |
| 文本单模态 | `text_tfidf_pred.csv` 或 `text_bert_pred.csv` |
| 图像单模态 | `image_resnet_pred.csv` |
| 多模态融合 | `fusion_pred.csv` |

如果融合指标不高，也应保留结果并写明：当前图像预测仍是占位，融合提升主要依赖文本概率、图像向量和行为向量是否提供有效补充。
