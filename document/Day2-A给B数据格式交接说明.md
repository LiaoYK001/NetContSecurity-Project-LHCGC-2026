# Day2 A 给 B：`data/processed/` 数据格式交接说明

适用角色：成员 B（图像与多模态融合）  
撰写人：成员 A  
日期：2026-06-08  
范围：仅说明 `data/processed/` 目录下的表与 JSON；`outputs/` 中的向量与 Day3 预测见 [Day3-A文本与行为进度说明.md](Day3-A文本与行为进度说明.md)。

相关规格：[Day1-ABC开发组接口与任务规格.md](Day1-ABC开发组接口与任务规格.md)

---

## 0. 一句话总结

- **数据来源**：公开微博谣言数据集，经 `src/prepare_data.py` 从 `weibo/` 原始文件解析并清洗。
- **样本量**：7723 条（已去掉空文本）；主键为 **`sample_id`**（微博 `tweet_id`，字符串）。
- **B 最该先看**：[`data/processed/dataset_v1.csv`](../data/processed/dataset_v1.csv) —— 含 `image_path`、`label`、`split`，与 Day1 最小字段集一致。
- **多图场景**：查 [`data/processed/all_images.csv`](../data/processed/all_images.csv)。
- **汇总数字**：见 [`data/processed/dataset_stats.json`](../data/processed/dataset_stats.json)。

---

## 1. 文件总览与关系

| 文件 | 行数（约） | 粒度 | 生成脚本 | B 是否必读 |
| --- | --- | --- | --- | --- |
| `dataset_v1.csv` | 7723 | 1 行 = 1 样本 | `prepare_data.py` | **是（主表）** |
| `all_images.csv` | 12713 | 1 行 = 1 张图（按 `image_index`） | `prepare_data.py` | 多图时需要 |
| `text_samples.csv` | 7723 | 1 行 = 1 样本 | `prepare_data.py` | 否（A 文本分支用） |
| `behavior_features.csv` | 7723 | 1 行 = 1 样本 | `prepare_data.py` | 否（行为原始特征） |
| `behavior_features_enriched.csv` | 7723 | 1 行 = 1 样本 | `extract_behavior_features.py` | 否（A 扩展行为特征） |
| `dataset_stats.json` | — | 全库统计 | `prepare_data.py` | 建议扫一眼 |

```mermaid
flowchart LR
  subgraph raw [weibo 原始数据]
    tweets[tweets/*.txt]
    pickles[train/val/test_id.pickle]
    imgdir[rumor/nonrumor_images]
  end
  prepare[prepare_data.py]
  enrich[extract_behavior_features.py]
  subgraph processed [data/processed]
    v1[dataset_v1.csv]
    text[text_samples.csv]
    beh[behavior_features.csv]
    enriched[behavior_features_enriched.csv]
    imgs[all_images.csv]
    stats[dataset_stats.json]
  end
  tweets --> prepare
  pickles --> prepare
  imgdir --> prepare
  prepare --> v1
  prepare --> text
  prepare --> beh
  prepare --> imgs
  prepare --> stats
  beh --> enrich
  enrich --> enriched
```

说明：`prepare_data.py` 同时会写 `outputs/handoff/images_manifest.csv`（等于 `dataset_v1` 的 `sample_id,image_path,label,split` 四列），内容与主表图片列一致，本文不单独展开。

---

## 2. 全局约定

| 项目 | 约定 |
| --- | --- |
| 主键 | `sample_id`，字符串，全表唯一，与 C 评估、A/B 预测 CSV 对齐 |
| 标签 `label` | 二分类：`normal`（非谣言/正常） / `risk`（谣言/风险） |
| 划分 `split` | `train` / `val` / `test`（来自 `weibo/*.pickle`） |
| 路径 | 均为**仓库根目录相对路径**；读图时 `Path(repo_root) / image_path` |
| 编码 | UTF-8 CSV，首行为表头 |
| 缺失图片 | Day1 约定：保留样本，`image_path` 为空；图像向量填零并记录 |

当前规模（`dataset_stats.json` 快照）：

| 指标 | 数值 |
| --- | --- |
| 导出样本数 | 7723 |
| 有本地图 | 7681 |
| 无本地图 | 42 |
| `normal` | 3615 |
| `risk` | 4108 |
| `train` / `val` / `test` | 5415 / 843 / 1465 |

---

## 3. 逐文件说明

### 3.1 `dataset_v1.csv`（B 主入口）

**用途**：全组对齐用的主样本表，满足 Day1 §3.1 最小字段集。B 做图像预处理、ResNet 特征提取、与 C 对齐评估时，**以本表为样本清单**。

**粒度**：1 行 = 1 个 `sample_id`（已过滤清洗后文本为空的记录）。

**字段**：

| 字段 | 类型 | 可空 | 说明 |
| --- | --- | --- | --- |
| `sample_id` | string | 否 | 微博 ID，全表唯一 |
| `text` | string | 否 | 清洗后正文（去除 URL、手机号、邮箱、`@用户` 等，见 `prepare_data.clean_text`） |
| `image_path` | string | 是 | **首张**本地匹配图片的相对路径；无图则为空字符串 `""` |
| `label` | string | 否 | `normal` 或 `risk` |
| `source` | string | 否 | 固定为 `weibo_rumor_dataset` |
| `split` | string | 否 | `train` / `val` / `test` |

**示例行**（节选）：

```csv
sample_id,text,image_path,label,source,split
3900416838856950,柯迪指挥官报告，我们已登上长城,weibo/nonrumor_images/62b31d36gw1ex8ts40449j20si0iygns.jpg,normal,weibo_rumor_dataset,train
```

B 实际读取图像时通常只需：`sample_id`, `image_path`, `label`, `split`。

---

### 3.2 `all_images.csv`（多图展开表）

**用途**：一条微博原文可能对应多张图片 URL；本表列出**所有能在本地 `weibo/*_images/` 匹配到的图片**，供 B 做多图实验或抽查。

**粒度**：1 行 = 一个 `(sample_id, image_index)` 组合。

**字段**：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sample_id` | string | 与主表一致 |
| `image_path` | string | 相对路径；无本地图时为空 |
| `image_index` | int | 从 0 起编号；**无本地图时为 -1**（仍保留一行占位） |
| `label` | string | 与主表一致 |
| `split` | string | 与主表一致 |

**与 `dataset_v1` 的关系**：

- 若某 `sample_id` 有图：`dataset_v1.image_path` = 本表中该 `sample_id` 且 `image_index=0` 的 `image_path`。
- 若某 `sample_id` 无图：主表 `image_path` 为空；本表有一行 `image_index=-1`、`image_path` 为空。

**示例**（同一样本多图）：

```csv
sample_id,image_path,image_index,label,split
3900416838856950,weibo/nonrumor_images/62b31d36gw1ex8ts40449j20si0iygns.jpg,0,normal,train
3900416838856950,weibo/nonrumor_images/62b31d36gw1ex8tsickrwj20tf0ivdhc.jpg,1,normal,train
```

---

### 3.3 `text_samples.csv`（A 文本分支输入）

**用途**：仅保留文本与标签、划分，供 A 做 TF-IDF / BERT 等文本模型。行集合与 `dataset_v1` 一致（同一批 `sample_id`）。

**粒度**：1 行 = 1 个 `sample_id`。

**字段**：

| 字段 | 说明 |
| --- | --- |
| `sample_id` | 主键 |
| `text` | 与 `dataset_v1.text` 相同 |
| `label` | `normal` / `risk` |
| `split` | `train` / `val` / `test` |

B 做融合时一般**不必**读本表；文本向量由 A 产出在 `outputs/predictions/text_embeddings.csv`（另见 A 后续交接）。

---

### 3.4 `behavior_features.csv`（行为基线特征，未标准化）

**用途**：从微博元数据行与文本统计抽取的数值特征，供 A 行为分析与扩展。数值为**原始尺度**，未做 StandardScaler。

**粒度**：1 行 = 1 个 `sample_id`。

**字段**：

| 字段 | 含义 |
| --- | --- |
| `sample_id`, `label`, `split` | 对齐主表 |
| `verified` | 是否认证用户（0/1） |
| `reposts` | 转发数 |
| `comments` | 评论数 |
| `likes` | 点赞数 |
| `engagement_total` | 转发 + 评论 + 点赞 |
| `interaction_ratio` | `likes / max(reposts + comments, 1)` |
| `followers` | 粉丝数（见 §5：nonrumor 常为 0） |
| `following` | 关注数 |
| `posts_count` | 发微博数 |
| `num_image_urls` | 原文中图片 URL 个数 |
| `text_length` | 清洗后文本字符长度 |
| `url_mentions` | 文本中 URL 出现次数 |
| `at_mentions` | 文本中 `@用户` 出现次数 |

**示例行**（节选）：

```csv
sample_id,label,split,verified,reposts,comments,likes,engagement_total,interaction_ratio,followers,following,posts_count,num_image_urls,text_length,url_mentions,at_mentions
3900416838856950,normal,train,1,28,18,30,76,0.652174,0,0,0,9,15,0,0
```

---

### 3.5 `behavior_features_enriched.csv`（A 扩展行为特征，未标准化）

**用途**：在 §3.4 基础上由 `extract_behavior_features.py` 增加衍生列（重复文本比例、用户发帖频率代理、互动构成比、敏感词计数等）。仍为**原始尺度**；标准化后的融合向量在 `outputs/predictions/behavior_embeddings.csv`（A 另交，不在本文）。

**粒度**：1 行 = 1 个 `sample_id`（7723 行，与主表一致）。

**列结构**：`sample_id`, `label`, `split` + 下列 **27 个特征列**（与 `beh_emb_000` … `beh_emb_026` 一一对应）：

| 字段 | 含义 |
| --- | --- |
| `verified` | 同 §3.4 |
| `reposts`, `comments`, `likes` | 同 §3.4 |
| `engagement_total`, `interaction_ratio` | 同 §3.4 |
| `followers`, `following`, `posts_count` | 同 §3.4 |
| `num_image_urls`, `text_length` | 同 §3.4 |
| `link_count` | 链接数（来自 `url_mentions`） |
| `mention_count` | `@` 提及数（来自 `at_mentions`） |
| `fan_follow_ratio` | 粉丝数 / max(关注数, 1) |
| `engagement_per_char` | 总互动 / max(文本长度, 1) |
| `engagement_per_follower` | 总互动 / max(粉丝数, 1) |
| `like_share` | 点赞占总互动比例 |
| `comment_share` | 评论占总互动比例 |
| `repost_share` | 转发占总互动比例 |
| `repeat_ratio` | 与同正文样本重复程度（全库归一化） |
| `is_duplicate_text` | 正文是否在库内重复（0/1） |
| `user_post_count` | 该用户在数据集中出现次数 |
| `user_post_frequency` | 用户发帖频率代理 |
| `hours_since_first_post` | 相对该用户首条帖的小时数 |
| `has_local_image` | 主表是否有本地图（0/1） |
| `image_link_ratio` | 图片 URL 数相对链接+图片 URL 的比例 |
| `sensitive_word_count` | 命中预设敏感词表的次数 |

B 若只做图像分支，可忽略本表；第 5 天融合时由 A 提供标准化后的 `behavior_embeddings.csv` 即可。

---

### 3.6 `dataset_stats.json`（机器可读汇总）

**用途**：快速查看导出规模、标签与划分分布、各产物路径；写报告或自检时用。

**主要字段**：

| 字段 | 说明 |
| --- | --- |
| `parsed_records` | 从原始推文解析到的记录数 |
| `exported_rows` | 最终写入 `dataset_v1` 的行数 |
| `empty_text_dropped` | 因空文本丢弃的数量 |
| `with_local_image` / `without_local_image` | 有/无本地图样本数 |
| `label_counts` | 各标签计数 |
| `split_counts` | 各划分计数 |
| `outputs` | 各 CSV 相对路径映射 |

---

## 4. B 侧最小使用方式（仅 `data/processed`）

1. **样本列表与划分**：读 `dataset_v1.csv`，按 `split` 过滤 train/val/test。
2. **单图路径**：使用 `image_path`；加载前检查 `os.path.exists(repo_root / image_path)`。
3. **多图**：对给定 `sample_id` 在 `all_images.csv` 中筛选，`image_index >= 0`，按 `image_index` 排序。
4. **无图样本**（当前 42 条）：`image_path == ""` 或 `all_images.image_index == -1`；按 Day1 用**零向量**占位，并在日志中记录 `sample_id`。
5. **标签**：训练/评估用 `label`；交给 C 的预测 CSV 须用 Day1 统一格式（`true_label` / `pred_label` 等），但 `sample_id` 必须与本表一致。

推荐 B 首日自检命令（与 [Day2-B图像预处理进度说明.md](Day2-B图像预处理进度说明.md) 一致）：

```bash
python3 src/extract_image_features.py --input data/processed/dataset_v1.csv --limit 20
```

---

## 5. 已知数据限制（必读）

1. **nonrumor 用户画像字段常为 0**  
   原始 `train_nonrumor.txt` / `test_nonrumor.txt` 的元数据列格式与 rumor 文件不同，`followers` / `following` / `posts_count` 在 nonrumor 样本上常被填为 0。行为特征仍可用于 rumor 子集或相对统计，但不宜单独解读 nonrumor 的粉丝数。

2. **42 条样本无本地图片**  
   原文可能有图 URL，但 `weibo/rumor_images` 或 `weibo/nonrumor_images` 中未匹配到文件。B 应对这些样本填零向量，不要从表中删除。

3. **文本已脱敏清洗**  
   手机号、邮箱、URL、`@用户` 等已在 `text` 中替换或删除，与原始推文不完全一致。

4. **Git 不提交真实数据**  
   `data/processed/` 在 `.gitignore` 中；B 需从 A 或共享盘获取本地 CSV，不能仅依赖 `git clone`。

---

## 6. 如何重新生成

**主表与基础 processed 文件**（需先有 `weibo/` 原始数据）：

```bash
python3 src/prepare_data.py --weibo-root weibo --output-dir data/processed --handoff-dir outputs/handoff
```

**扩展行为表**（依赖上一步产出的 `behavior_features.csv` 等）：

```bash
python3 src/extract_behavior_features.py \
  --behavior-input data/processed/behavior_features.csv \
  --text-input data/processed/text_samples.csv \
  --dataset-input data/processed/dataset_v1.csv \
  --weibo-root weibo \
  --enriched-output data/processed/behavior_features_enriched.csv
```

若环境已配置 `uv`，可将 `python3` 换成 `uv run python`。

---

## 7. 变更记录

| 日期 | 版本 | 变更人 | 说明 |
| --- | --- | --- | --- |
| 2026-06-08 | v1.0 | 成员 A | 首版：覆盖 `data/processed/` 六类产物，供 B Day2 图像预处理与后续 ResNet 使用 |
