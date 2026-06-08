# Day3 A 文本与行为进度说明

适用角色：成员 A（撰写）/ 成员 B、C（阅读）  
日期：2026-06-09  
相关：[Day1 §5 预测格式](Day1-ABC开发组接口与任务规格.md) · [Day2-A `data/processed/`](Day2-A给B数据格式交接说明.md)

---

## 0. 一句话

- Day3 三条流水线（TF-IDF 基线 / BERT 文本向量 / 行为向量）脚本与默认输出路径已就绪。
- **C**：用 `text_tfidf_pred.csv` 做文本基线评估（Day1 五列）。
- **B**：用 `text_embeddings.csv`、`behavior_embeddings.csv` 按 `sample_id` 与图像向量融合。

---

## 1. 进度

| 项 | 状态 | 说明 |
| --- | --- | --- |
| Day2 数据 | 已完成 | `data/processed/text_samples.csv` 等，见 Day2-A |
| TF-IDF 基线 | 已跑通 | 7723 行预测 + 指标 JSON |
| BERT 文本向量 | 已生成 | `bert-base-chinese`，768 维，7723 行 |
| 行为向量 | 已生成 | 27 维标准化 `beh_emb_*`，7723 行 |
| 输入表 | `data/processed/text_samples.csv` | A 侧文本/行为脚本共用 |

---

## 2. 脚本与命令

| 脚本 | 作用 |
| --- | --- |
| `src/train_text_baseline.py` | TF-IDF + 逻辑回归 → 预测 CSV + 指标 |
| `src/extract_text_features.py` | 冻结 BERT → 768 维文本向量 |
| `src/extract_behavior_features.py` | 27 维行为特征 + StandardScaler → 行为向量 |

```powershell
uv run python src/train_text_baseline.py
uv run python src/extract_text_features.py --train-classifier
uv run python src/extract_behavior_features.py
```

说明：`outputs/` 产物默认不提交 Git；B/C 本地向 A 索取或各自重跑上述命令。

---

## 3. 给 B/C 交接

预测 CSV 列定义见 [Day1 §5.1](Day1-ABC开发组接口与任务规格.md)。向量表均以 **`sample_id`** 与 `dataset_v1.csv` 对齐。

| 路径 | 消费者 | 内容 | 状态 |
| --- | --- | --- | --- |
| `outputs/predictions/text_tfidf_pred.csv` | C | 五列；`model_name=text_tfidf` | test acc **0.747**，F1 **0.700** |
| `outputs/metrics/text_tfidf_metrics.json` | C | 分 split 指标 | 已生成 |
| `outputs/handoff/text_tfidf_error_cases.csv` | C | 错例样例 | 已生成 |
| `outputs/predictions/text_embeddings.csv` | B | `txt_emb_000`…`767`（768 维） | 7723 行 |
| `outputs/predictions/text_feature_meta.json` | B | 模型名、维度 | `bert-base-chinese` |
| `outputs/predictions/text_bert_pred.csv` | C | 可选；BERT 头预测（若加 `--train-classifier` 且训练成功） | 视本地运行 |
| `outputs/predictions/behavior_embeddings.csv` | B | `beh_emb_000`…`026`（27 维） | 7723 行 |
| `outputs/predictions/behavior_feature_meta.json` | B | 特征名 + scaler 统计 | 已生成 |
| `data/processed/behavior_features_enriched.csv` | A/B | 标准化前行为列（可读） | 7723 行 |

C 评估时只用 **`split=test`** 子集；A 的 scaler / TF-IDF 仅在 train 上 fit。

---

## 4. 验收

- [ ] C 能读取 `text_tfidf_pred.csv`，列名与 Day1 §5.1 一致。
- [ ] B 能用 `sample_id` 将 `text_embeddings.csv`、`behavior_embeddings.csv` 与 `dataset_v1.csv` merge。
- [ ] 行为标准化与基线训练未使用 test/val 标签泄漏（行为脚本仅在 train fit scaler）。

---

| 日期 | 版本 | 撰写 | 说明 |
| --- | --- | --- | --- |
| 2026-06-09 | v1.0 | 成员 A | Day3 文本基线 + 文本/行为向量交接 |
