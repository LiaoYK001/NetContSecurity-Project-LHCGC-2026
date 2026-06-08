# Day3 A 文本与行为进度说明

适用角色：成员 A（撰写）/ 成员 B、C（阅读）  
日期：2026-06-09  
相关：[Day1 §5 预测格式](Day1-ABC开发组接口与任务规格.md) · [Day2-A `data/processed/](Day2-A给B数据格式交接说明.md)`

---

## 0. 一句话

- Day3 三条流水线（TF-IDF 基线 / BERT 文本向量 / 行为向量）脚本与默认输出路径已就绪。
- **C**：用 `text_tfidf_pred.csv` 做文本基线评估（Day1 五列）。
- **B**：用 `text_embeddings.csv`、`behavior_embeddings.csv` 按 `sample_id` 与图像向量融合。

---

## 1. 进度


| 项         | 状态                                           | 说明                               |
| --------- | -------------------------------------------- | -------------------------------- |
| Day2 数据   | 已完成                                          | `data/processed/` 见下表，详见 Day2-A  |
| TF-IDF 基线 | 已跑通                                          | 7723 行预测 + 指标 JSON               |
| BERT 文本向量 | 已生成                                          | `bert-base-chinese`，768 维，7723 行 |
| 行为向量      | 已生成                                          | **13 维**标准化 `beh_emb_`*，7723 行   |
| 文本/行为输入   | `text_samples.csv` / `behavior_features.csv` | 见 §2                             |


---

## 2. 清洗后 CSV 含义（`data/processed/`）


| 文件                      | 行数（约） | 含义                                                              | 谁用                                                   |
| ----------------------- | ----- | --------------------------------------------------------------- | ---------------------------------------------------- |
| `dataset_v1.csv`        | 7723  | 主样本表：一条微博 = 一行。含清洗后 `text`、首张本地图 `image_path`、`label`、`split` 等 | **B 必读**；C 评估对齐也靠 `sample_id`                        |
| `all_images.csv`        | 12713 | 多图展开表：同一 `sample_id` 可有多行，每行一张图（`image_index` 0,1,2…）           | B 做多图实验时用；主表只保留 index=0                              |
| `text_samples.csv`      | 7723  | 纯文本表：`sample_id` + `text` + `label` + `split`（与主表同批样本）          | A 跑 TF-IDF / BERT；B 融合不读，用 A 的 `text_embeddings.csv` |
| `behavior_features.csv` | 7723  | 行为原始特征（未标准化）：转发/评论/点赞、粉丝、文本长度、URL 数等                            | A 行为分析；B 融合不读，用 A 的 `behavior_embeddings.csv`        |
| `dataset_stats.json`    | —     | 全库统计：样本数、标签/划分分布、有图无图数量、各文件路径                                   | 快速自检                                                 |


---

## 3. 运行命令

### 3.1 一次性：生成 `data/processed/`（需 `weibo/` 原始数据）

```bash
uv sync
uv run python src/prepare_data.py --weibo-root weibo --output-dir data/processed --handoff-dir outputs/handoff
```

### 3.2 C 用：TF-IDF 文本基线（预测 + 指标，**不是**融合向量）

```bash
uv run python src/train_text_baseline.py
```

产出：`outputs/predictions/text_tfidf_pred.csv`、`outputs/metrics/text_tfidf_metrics.json` 等。

### 3.3 B 用：两份融合向量 CSV（**成员 B 必读**）

```bash
uv run python src/extract_text_features.py
uv run python src/extract_behavior_features.py
```


| 命令                             | 读入                                     | 产出                                                              |
| ------------------------------ | -------------------------------------- | --------------------------------------------------------------- |
| `extract_text_features.py`     | `data/processed/text_samples.csv`      | `outputs/predictions/text_embeddings.csv`（768 维 `txt_emb_*`）    |
| `extract_behavior_features.py` | `data/processed/behavior_features.csv` | `outputs/predictions/behavior_embeddings.csv`（13 维 `beh_emb_*`） |


可选（给 C 的 BERT 分类头预测，**融合不需要**）：

```bash
uv run python src/extract_text_features.py --train-classifier
```

说明：`outputs/` 默认不提交 Git；B/C 向 A 索取文件或在本机按上表重跑。

---

## 4. TF-IDF 基线结果（当前版本）

模型：`text_tfidf`（TF-IDF + 逻辑回归），输入 `text_samples.csv`，词表 9423。  
标签：`risk` = 谣言微博，`normal` = 非谣言（见 Day2-A）。

### 4.1 分 split 指标


| Split    | 样本数  | Accuracy  | Precision | Recall | F1        |
| -------- | ---- | --------- | --------- | ------ | --------- |
| **test** | 1465 | **0.747** | 0.905     | 0.570  | **0.700** |
| val      | 843  | 0.675     | 0.863     | 0.471  | 0.610     |
| train    | 5415 | 0.889     | 0.967     | 0.820  | 0.888     |


### 4.2 Test 集混淆矩阵（行 = 真实，列 = 预测）


|                 | pred normal | pred risk |
| --------------- | ----------- | --------- |
| **true normal** | 664         | 45        |
| **true risk**   | 325         | 431       |


### 4.3 Test 集按类 F1


| 类别     | Precision | Recall | F1    |
| ------ | --------- | ------ | ----- |
| normal | 0.671     | 0.937  | 0.782 |
| risk   | 0.905     | 0.570  | 0.700 |


**解读**：模型偏保守，易把 `risk` 判成 `normal`（325 条漏检）；`risk` 召回偏低是后续多模态融合要改善的重点。  
完整 JSON：`outputs/metrics/text_tfidf_metrics.json`。

---

## 5. 给 B/C 交接清单

预测 CSV 列定义见 [Day1 §5.1](Day1-ABC开发组接口与任务规格.md)。向量表均以 `**sample_id`** 与 `dataset_v1.csv` 对齐。


| 路径                                               | 消费者 | 内容                                      | 状态                  |
| ------------------------------------------------ | --- | --------------------------------------- | ------------------- |
| `outputs/predictions/text_tfidf_pred.csv`        | C   | 五列；`model_name=text_tfidf`              | 7723 行              |
| `outputs/metrics/text_tfidf_metrics.json`        | C   | 分 split 指标                              | 见 §4                |
| `outputs/handoff/text_tfidf_error_cases.csv`     | C   | 错例样例                                    | 已生成                 |
| `outputs/predictions/text_embeddings.csv`        | B   | `txt_emb_000`…`767`（768 维）              | 7723 行              |
| `outputs/predictions/text_feature_meta.json`     | B   | 模型名、维度                                  | `bert-base-chinese` |
| `outputs/predictions/text_bert_pred.csv`         | C   | 可选；BERT 头预测（`--train-classifier` 且训练成功） | 视本地运行               |
| `outputs/predictions/behavior_embeddings.csv`    | B   | `beh_emb_000`…`012`（**13 维**）           | 7723 行              |
| `outputs/predictions/behavior_feature_meta.json` | B   | 特征名 + scaler 统计                         | `feature_dim: 13`   |


行为向量 13 列与原始特征一一对应（标准化前见 `behavior_features.csv`）：  
`verified`, `reposts`, `comments`, `likes`, `engagement_total`, `interaction_ratio`, `followers`, `following`, `posts_count`, `num_image_urls`, `text_length`, `url_mentions`, `at_mentions`。

C 评估时只用 `**split=test`** 子集；TF-IDF 与行为 StandardScaler **仅在 train 上 fit**。

---

## 6. 验收

- [ ] C 能读取 `text_tfidf_pred.csv`，列名与 Day1 §5.1 一致。
- [ ] B 能用 `sample_id` 将 `text_embeddings.csv`、`behavior_embeddings.csv` 与 `dataset_v1.csv` merge。
- [ ] 行为标准化与基线训练未使用 test/val 标签泄漏（行为脚本仅在 train fit scaler）。

---


| 日期         | 版本   | 撰写   | 说明                                                        |
| ---------- | ---- | ---- | --------------------------------------------------------- |
| 2026-06-09 | v1.1 | 成员 A | 补充 processed 表、运行命令、TF-IDF 基线结果；行为向量改为 13 维，删除 enriched 表 |
| 2026-06-09 | v1.0 | 成员 A | Day3 文本基线 + 文本/行为向量交接                                     |


