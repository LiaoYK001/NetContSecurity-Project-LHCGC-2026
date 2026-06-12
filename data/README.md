# Data Reproduction Guide

本仓库是 public，不提交完整数据集、原始图片、爬虫输出、模型权重或最终实验输出。老师或同学复现时，请把公开数据集下载到本地 ignored 目录，再运行脚本生成 `data/processed/`。

## 1. 先理解复现边界

别人 `git clone` 后可以直接获得：

```text
代码
README 和教程
字段合同
data/sample/ 脱敏小样例
```

别人不会直接获得：

```text
完整 Weibo 数据集
完整 data/processed/*.csv
原始图片
outputs/ 下的预测、图表和交接包
模型权重和缓存
```

仓库中可能包含 `outputs/demo/` 下的 public-safe 示例指标和图表，但这些只是展示用汇总结果，不包含完整数据或样本级预测。所以本仓库的复现方式仍然是：先下载公开数据集，再生成本项目 CSV，然后跑训练和评估流程。不能在没有任何数据的情况下复现最终指标。

## 2. 默认数据来源

本项目默认使用 EANN-KDD18 公开 Weibo 多模态谣言检测数据集：

- 论文：Wang et al., *EANN: Event Adversarial Neural Networks for Multi-Modal Fake News Detection*, KDD 2018
- GitHub: https://github.com/yaqingwang/EANN-KDD18

标签映射：

| 原始类别 | 本项目标签 | 说明 |
| --- | --- | --- |
| rumor | `risk` | 风险内容 |
| nonrumor | `normal` | 正常内容 |

## 3. 下载和放置数据

1. 打开 EANN-KDD18 GitHub 页面。
2. 按原仓库说明下载 Weibo 数据和图片文件。
3. 在本仓库根目录创建本地目录 `tmp/weibo/`。
4. 把解压后的 tweets、split pickle 和图片目录整理成下面结构。

推荐目录：

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

`tmp/` 已被 `.gitignore` 排除，不会进入 public 仓库。不要把下载的完整数据移动到可提交目录。

## 4. 生成 processed 数据

先同步环境：

```powershell
uv sync
```

从本地 Weibo 数据生成项目 CSV：

```powershell
uv run python src/prepare_data.py --weibo-root tmp/weibo --output-dir data/processed --handoff-dir outputs/handoff
```

生成文件：

```text
data/processed/dataset_v1.csv
data/processed/text_samples.csv
data/processed/behavior_features.csv
data/processed/all_images.csv
data/processed/dataset_stats.json
outputs/handoff/images_manifest.csv
```

这些完整文件默认不提交 Git。

## 5. 检查生成结果

运行下面命令检查行数和 `sample_id` 唯一性：

```powershell
uv run python -c "import pandas as pd; files=['data/processed/dataset_v1.csv','data/processed/text_samples.csv','data/processed/behavior_features.csv']; [print(f, len(pd.read_csv(f,dtype={'sample_id':'string'})), pd.read_csv(f,dtype={'sample_id':'string'})['sample_id'].nunique()) for f in files]"
```

合格标准：

```text
dataset_v1.csv、text_samples.csv、behavior_features.csv 都应该 sample_id 唯一
split 至少包含 train 和 test
label 只能是 normal 或 risk
```

`all_images.csv` 是多图展开表，同一个 `sample_id` 可以出现多行。它只用于图片清单检查，不能作为融合主表。融合和最终评估必须使用 `dataset_v1.csv`。

## 6. CSV 字段合同

`dataset_v1.csv` 一条内容一行，供图像、融合和评估使用：

```csv
sample_id,text,image_path,label,source,split
```

`text_samples.csv` 供 TF-IDF 和 BERT 文本分支使用：

```csv
sample_id,text,label,split
```

`behavior_features.csv` 供行为向量脚本使用：

```csv
sample_id,label,split,verified,reposts,comments,likes,...
```

`all_images.csv` 是多图展开表：

```csv
sample_id,image_path,image_index,label,split
```

## 7. 使用其他数据集时怎么接入

默认教程使用 Weibo 数据集，但不是唯一选择。其他公开数据集也可以接入，前提是先转换成同样的三个核心 CSV：

```text
data/processed/dataset_v1.csv
data/processed/text_samples.csv
data/processed/behavior_features.csv
```

最低要求：

1. `sample_id` 在三个核心表中唯一，并且能互相对齐。
2. `label` 统一为 `normal` 或 `risk`。
3. `split` 至少包含 `train` 和 `test`。
4. 文本缺失时要填空字符串，不要删除样本导致对不齐。
5. 图片缺失时 `image_path` 可以为空，图像脚本会用零向量保留样本。

## 8. 常见错误

| 问题 | 原因 | 处理方式 |
| --- | --- | --- |
| `Missing --weibo-root` | `tmp/weibo/` 路径不对 | 检查 `--weibo-root` 是否指向实际数据目录 |
| `Missing tweet file` | tweets 文件没有放到 `tweets/` 下 | 按推荐目录重新整理 |
| `Missing split file` | 缺少 `train_id.pickle` 等划分文件 | 从数据集重新下载或检查解压结果 |
| 图片大量 `file_not_found` | 图片目录名或 `--image-root` 不匹配 | 确认图片在 `tmp/weibo/rumor_images/` 和 `tmp/weibo/nonrumor_images/` |
| 融合时报重复 ID | 错把 `all_images.csv` 当主表 | 融合主表必须使用 `dataset_v1.csv` |

## 9. 可提交内容

可以提交：

```text
data/README.md
data/processed/README.md
data/sample/ 中的小体积脱敏样例
```

不要提交：

```text
完整 CSV
原图
真实用户链接
真实样本 ID 和原文片段
cookie、token、账号密码
模型权重和缓存
outputs/ 下的完整预测、错误案例和本地交接包
```
