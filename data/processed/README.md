# Processed Data Contract

`data/processed/` 是本地生成目录，完整 CSV 默认不提交 Git。

## `dataset_v1.csv`

一条微博一行，供图像、融合和评估使用。

```csv
sample_id,text,image_path,label,source,split
```

要求：

- `sample_id` 唯一。
- `label` 只能是 `normal` 或 `risk`。
- `split` 至少包含 `train` 和 `test`。
- `image_path` 是相对路径，可为空；缺图样本由图像脚本写零向量。

## `text_samples.csv`

供 TF-IDF 和 BERT 文本分支使用。

```csv
sample_id,text,label,split
```

## `behavior_features.csv`

供行为向量脚本使用。

```csv
sample_id,label,split,verified,reposts,comments,likes,...
```

行为特征会在 `split=train` 上 fit 标准化器，再转换全部 split。

## `all_images.csv`

多图展开表，同一个 `sample_id` 可以出现多行。

```csv
sample_id,image_path,image_index,label,split
```

注意：`all_images.csv` 不能作为融合主表。融合和最终评估必须使用 `dataset_v1.csv`。
