# Day4 B 图像特征生成进度说明

适用角色：成员 B  
日期：2026-06-09  
数据来源：`tmp\2026年6月9日\processed\dataset_v1.csv` 与对应本地图片目录

> 注意：`tmp\2026年6月9日\` 只是 czl 本机临时接收 A 数据的目录，不会提交到 Git，其他同学默认没有这个路径。团队复现时请改用自己的 A 数据目录，或在本地准备 `data/processed/` 后使用 `data/processed/dataset_v1.csv`。

## 1. 当前结论

Day4 B 图像特征已完成实际生成。A 提供的 7723 条样本均已保留并输出到图像向量表，后续 C 或 Day5 融合可以按 `sample_id` 对齐。

本次使用预训练 ResNet18 提取 512 维图像特征，不训练图像分类头。`image_resnet_pred.csv` 仍是接口占位文件，`risk_prob` 固定为 `0.5`，不能作为正式图像单模态指标。

## 2. 已生成输出

| 文件 | 行数/维度 | 用途 |
| --- | --- | --- |
| `outputs/predictions/image_preprocess_check.csv` | 7723 行，7 列 | Day4 图片读取与预处理检查 |
| `outputs/predictions/image_embeddings.csv` | 7723 行，516 列 | `sample_id,image_path,status,message,img_emb_000...img_emb_511` |
| `outputs/predictions/image_resnet_pred.csv` | 7723 行，5 列 | C 的统一预测 CSV 接口检查 |
| `outputs/predictions/behavior_embeddings.csv` | 7723 行，18 列 | Day5 融合前置行为向量 |
| `outputs/predictions/behavior_feature_meta.json` | 13 维行为特征元数据 | 行为特征名与 scaler 统计 |

以上文件均位于 `outputs/`，默认被 `.gitignore` 忽略，不提交 Git。

## 3. 实际运行命令

以下命令记录的是本机临时数据跑法。其他同学不要直接照抄 `tmp\2026年6月9日`，需要把 `$base` 改成自己机器上 A 数据所在目录。

```powershell
$base = Resolve-Path 'tmp\2026年6月9日'

uv run python src\extract_image_features.py `
  --input "$base\processed\dataset_v1.csv" `
  --image-root "$base" `
  --output outputs\predictions\image_preprocess_check.csv

uv run python src\extract_resnet_features.py `
  --input "$base\processed\dataset_v1.csv" `
  --image-root "$base" `
  --weights default `
  --embeddings-output outputs\predictions\image_embeddings.csv `
  --pred-output outputs\predictions\image_resnet_pred.csv

uv run python src\extract_behavior_features.py `
  --behavior-input "$base\processed\behavior_features.csv" `
  --embeddings-output outputs\predictions\behavior_embeddings.csv `
  --meta-output outputs\predictions\behavior_feature_meta.json
```

如果已经把 A 的数据按项目规范放到 `data/processed/`，可把输入路径改成：

```powershell
uv run python src\extract_image_features.py --input data\processed\dataset_v1.csv --image-root . --output outputs\predictions\image_preprocess_check.csv
uv run python src\extract_resnet_features.py --input data\processed\dataset_v1.csv --image-root . --weights default
uv run python src\extract_behavior_features.py --behavior-input data\processed\behavior_features.csv
```

说明：本机普通 `uv run` 曾因 uv-managed Python 路径权限/启动问题失败，提升权限后环境检查通过。ResNet18 预训练权重已成功下载并缓存。

## 4. 验收结果

图片预处理检查：

| 状态 | 数量 | 说明 |
| --- | ---: | --- |
| `ok` | 7678 | 图片成功读取、转 RGB、resize 到 224x224、normalize |
| `missing_path` | 42 | 主表中无图片路径，保留样本 |
| `unsupported_format` | 3 | `.gif` 文件，不在 Day1 约定的 `jpg/jpeg/png` 范围内 |

ResNet 图像向量：

| 状态 | 数量 | 说明 |
| --- | ---: | --- |
| `ok` | 7678 | 成功提取 ResNet18 512 维特征 |
| `missing_path` | 42 | 输出 512 维零向量 |
| `unsupported_format` | 3 | 输出 512 维零向量 |

预测接口：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
```

- 行数：7723
- `model_name`：全部为 `image_resnet`
- `risk_prob`：全部为 `0.5`

行为向量：

- 行数：7723
- 行为维度：13
- 状态：全部 `ok`
- scaler 仅在 `split=train` 上 fit

## 5. Day5 衔接

B 现在可以进入 Day5 融合准备，但正式融合仍需要 A 的文本侧输出。最小可用输入建议如下：

| 输入 | 状态 | 说明 |
| --- | --- | --- |
| `image_embeddings.csv` | 已生成 | B 已完成 |
| `behavior_embeddings.csv` | 已生成 | 行为脚本已修复并跑通 |
| `text_embeddings.csv` | 待确认 | 需要 A 提供或本机重新运行文本特征脚本 |
| `text_tfidf_pred.csv` | 待确认 | 若先做保底融合，可使用文本概率作为弱特征 |

下一步建议先补齐文本 embedding，再做按 `sample_id` 的三模态 merge 和融合模型训练。
