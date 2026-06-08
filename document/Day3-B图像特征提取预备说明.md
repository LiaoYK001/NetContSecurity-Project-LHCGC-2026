# Day3 B 图像特征提取预备说明

适用角色：成员 B  
目标：提前准备 ResNet18 图像向量提取框架，等 A 提供 `dataset_v1.csv` 和图片后直接进入 Day3-4 图像分支。

## 1. 当前进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| Day1 接口规格 | 已完成 | 标签、字段、预测 CSV、图片规范已统一 |
| Day2 图片预处理 | 已完成 | `src/extract_image_features.py` 可检查图片路径和预处理状态 |
| Day3 ResNet 特征框架 | 已完成 | 新增 `src/extract_resnet_features.py` |
| A 数据交付 | 等待中 | 仍需 `data/processed/dataset_v1.csv` 和本地图片 |
| C 正式评估 | 等待中 | 需要 C 用统一测试集和指标函数评估 |

## 2. 新增脚本

脚本路径：

```text
src/extract_resnet_features.py
```

默认命令：

```powershell
uv run python src/extract_resnet_features.py
```

常用命令：

```powershell
uv run python src/extract_resnet_features.py --input data/processed/dataset_v1.csv --limit 20
```

如果当前环境无法下载预训练权重，可先用未训练骨架跑通流程：

```powershell
uv run python src/extract_resnet_features.py --weights none --limit 20
```

## 3. 输入与输出

输入 CSV 至少包含：

```csv
sample_id,image_path,label,split
```

默认输出：

```text
outputs/predictions/image_embeddings.csv
outputs/predictions/image_resnet_pred.csv
```

图像向量 CSV 格式：

```csv
sample_id,image_path,status,message,img_emb_000,...,img_emb_511
```

预测占位 CSV 格式：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
```

注意：当前预测 CSV 只是为了提前满足 C 的接口检查。由于还没有训练图像分类头，`risk_prob` 默认是 `0.5`，不代表正式图像模型指标。

## 4. 处理策略

- 默认模型：ResNet18。
- 输出维度：512。
- 预处理：RGB、`224x224`、ImageNet normalize。
- 正常图片：输出 ResNet18 512 维特征。
- 缺失图片、路径错误、格式不支持、图片损坏：保留样本，输出 512 维零向量。
- 默认处理全部 `split`；如只处理测试集，可加 `--split test`。

## 5. Day3-4 后续衔接

等 A 提供真实数据后，B 需要：

1. 先运行 Day2 图片预处理检查，确认图片路径问题比例。
2. 再运行 ResNet 特征提取，生成 `image_embeddings.csv`。
3. 把图像向量维度说明交给 C：`ResNet18 输出 512 维`。
4. 如果要做正式图像预测，需要在图像向量后接简单分类器；当前脚本的预测 CSV 只是接口占位。

## 6. 今日验收清单

- [x] ResNet18 特征提取脚本框架已建立。
- [x] 无 A 数据时能提示等待 `dataset_v1.csv`。
- [x] 缺失或异常图片会输出零向量，保持 `sample_id` 对齐。
- [x] 能生成图像向量 CSV 和统一 5 列预测 CSV。
- [x] 支持 `--weights none`，避免预训练权重下载失败时阻塞流程。
- [ ] 等 A 提供真实数据后，运行前 20 条样本检查和特征提取。
