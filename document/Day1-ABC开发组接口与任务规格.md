# Day1 ABC 开发组接口与任务规格

适用对象：成员 A / 成员 B / 成员 C  
当前执行人：成员 B 临时统筹 A/B/C  
目标：完成第 1 天开发组基础约定，让第 2 天数据准备、第 3 天文本和图像基线不会因为接口不一致而阻塞。

## 1. Day1 总目标

第 1 天不训练模型、不下载真实数据、不提交真实图片或模型权重。今天只完成以下基础交付：

| 交付项 | 结论 |
| --- | --- |
| 标签体系 | 二分类：`normal` / `risk` |
| 数据文件格式 | CSV 优先，后续如需 JSONL 再补充转换规则 |
| 最小数据字段 | `sample_id,text,image_path,label,source,split` |
| 预测结果格式 | `sample_id,true_label,pred_label,risk_prob,model_name` |
| 图片输入规范 | 相对路径、`jpg/jpeg/png`、`224x224`、缺失图片用零向量 |
| 目录规范 | `data/`、`models/`、`outputs/`、`src/` 按本文约定 |
| 实验记录 | 使用 `outputs/metrics/experiment_log_template.csv` 作为模板 |

## 2. 统一标签定义

短周期课程项目先做二分类，避免一开始把垃圾广告、诈骗、有害内容、异常账号拆成过多类别。

| 标签 | 含义 | 示例判断 |
| --- | --- | --- |
| `normal` | 正常内容或正常账号行为 | 普通评论、正常分享、无明显导流或诈骗意图 |
| `risk` | 风险内容或异常账号行为 | 垃圾广告、诈骗引流、异常刷屏、疑似有害内容 |

约定：

- 所有数据表中的标准答案列统一叫 `label`。
- 所有预测结果中的真实标签列统一叫 `true_label`。
- `label`、`true_label`、`pred_label` 的取值只能是 `normal` 或 `risk`。
- 多分类、细粒度风险类型只作为后续扩展，不进入 Day1 必做范围。

## 3. A 数据与文本约定

成员 A 负责第 2 天交付第一版可训练数据。Day1 先定字段，不要求今天拿到真实数据。

### 3.1 数据表字段

建议路径：`data/processed/dataset_v1.csv`

```csv
sample_id,text,image_path,label,source,split
sample_0001,示例文本,data/raw/images/sample_0001.jpg,risk,public_dataset,train
sample_0002,普通内容,,normal,public_dataset,test
```

| 字段 | 类型 | 是否可空 | 说明 |
| --- | --- | --- | --- |
| `sample_id` | string | 否 | 全组唯一样本 ID，A/B/C 对齐都靠它 |
| `text` | string | 原则上否 | 帖子、评论、标题或简介文本 |
| `image_path` | string | 是 | 仓库相对路径；没有图片时留空 |
| `label` | string | 否 | 标准答案，只能是 `normal` 或 `risk` |
| `source` | string | 否 | 数据来源类别或公开数据集名，不写隐私信息 |
| `split` | string | 否 | `train`、`val` 或 `test` |

### 3.2 数据清洗和脱敏约定

- 不提交 `data/raw/`、`data/interim/`、`data/processed/` 中的真实数据。
- 不保存 cookie、token、私信、账号密码、手机号、邮箱、二维码原图或未脱敏主页链接。
- 文本中出现手机号、邮箱、精确地址、账号名等信息时，清洗阶段统一替换为占位符。
- 第 2 天 A 至少给 B/C 一份本地 CSV 样例，并说明数据来源和标签映射方式。

### 3.3 A 给 B/C 的 Day2-Day3 交接

| 时间 | 给谁 | 文件或信息 | 必须满足 |
| --- | --- | --- | --- |
| Day2 | B | `dataset_v1.csv` 中的 `sample_id,image_path,label,split` | `image_path` 是相对路径，缺失图片可为空 |
| Day2 | C | `dataset_v1.csv` 字段说明 | `sample_id` 唯一，`split` 已存在 |
| Day3 | C | 文本预测 CSV | 使用本文第 5 节预测结果格式 |

## 4. B 图像与融合约定

成员 B 负责图像分支和后续融合。Day1 先保证图片路径、缺失图片和向量格式不会阻塞。

### 4.1 图片输入规范

| 项目 | 约定 |
| --- | --- |
| 路径格式 | `image_path` 使用仓库相对路径，例如 `data/raw/images/001.jpg` |
| 支持格式 | `jpg`、`jpeg`、`png` |
| 颜色模式 | 读取后统一转为 RGB |
| 输入尺寸 | ResNet 输入统一 resize 到 `224x224` |
| 归一化 | 使用 torchvision 预训练模型常用 ImageNet mean/std |
| 缺失图片 | 默认保留样本，图像向量填零，并在日志中记录 |
| 损坏图片 | 跳过或填零，但必须记录 `sample_id` 和失败原因 |

### 4.2 图像向量和预测输出

图像向量建议第 3-4 天输出，Day1 只定格式。

| 文件 | 建议路径 | 说明 |
| --- | --- | --- |
| 图像向量 | `outputs/predictions/image_embeddings.csv` | `sample_id` 加若干 `img_emb_*` 列 |
| 图像预测 | `outputs/predictions/image_resnet_pred.csv` | 使用统一预测结果格式 |
| 融合预测 | `outputs/predictions/fusion_pred.csv` | Day5 使用统一预测结果格式 |

图像分支保底路线：冻结预训练 ResNet18，只做特征提取。图片不足或质量差时，图像分支作为辅助实验，不阻塞文本基线和评估。

### 4.3 B 给 C 的 Day3 交接

| 时间 | 给谁 | 文件或信息 | 必须满足 |
| --- | --- | --- | --- |
| Day3-4 | C | 图像预测 CSV | 使用本文第 5 节预测结果格式 |
| Day3-4 | C | 图像向量维度说明 | 例如 `ResNet18 输出 512 维` |
| Day5 | C | 融合预测 CSV | 与文本、图像使用同一测试集 |

## 5. C 集成与评测约定

成员 C 负责统一目录、命令、评估格式和实验记录。Day1 先公布强制接口。

### 5.1 统一预测结果格式

所有模型预测 CSV 固定 5 列：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
sample_0001,risk,risk,0.87,text_tfidf
sample_0002,normal,normal,0.12,image_resnet
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sample_id` | string | 必须和数据表一致 |
| `true_label` | string | 真实标签，只能是 `normal` 或 `risk` |
| `pred_label` | string | 预测标签，只能是 `normal` 或 `risk` |
| `risk_prob` | float | 预测为 `risk` 的概率，范围 0 到 1 |
| `model_name` | string | 模型名，如 `text_tfidf`、`image_resnet`、`fusion_v1` |

评估时只比较同一 `split=test` 的样本。若 A/B 的预测样本数不同，C 先按 `sample_id` 检查差异，不直接把指标放在同一张表里比较。

### 5.2 实验记录模板

模板路径：`outputs/metrics/experiment_log_template.csv`

字段：

```csv
experiment_id,date,owner,model_name,data_version,input_path,output_path,accuracy,precision,recall,f1,notes
```

第 2 天开始，每次实验至少记录实验编号、模型名、数据版本、输入路径、输出路径和备注。指标没有跑出前留空即可。

### 5.3 统一命令命名

后续脚本按以下命名，Day1 只定规则：

```powershell
uv run python src/prepare_data.py
uv run python src/train_text_baseline.py
uv run python src/extract_text_features.py
uv run python src/extract_image_features.py
uv run python src/train_fusion.py
uv run python src/evaluate.py
uv run python src/plot_figures.py
```

## 6. 目录结构约定

```text
.
├── data/
│   ├── raw/                 # 原始公开数据，本地保存，不提交真实内容
│   ├── interim/             # 清洗中间结果，本地保存
│   └── processed/           # 清洗后训练数据，本地保存
├── models/                  # 模型权重，本地保存
├── outputs/
│   ├── predictions/         # 预测 CSV 和向量输出，本地保存
│   │   └── ablation/        # 消融实验预测
│   ├── metrics/             # 指标表和实验记录模板
│   ├── figures/             # 图表输出，本地保存
│   └── handoff/             # 第 7 天交接包，本地整理
└── src/                     # 后续代码，可提交
```

注意：真实 `data/`、`models/`、`outputs/` 内容不提交。仓库只保留 `.gitkeep` 和模板文件。

## 7. Day2-Day3 交接清单

| 时间 | 负责人 | 交付物 | 验收标准 |
| --- | --- | --- | --- |
| Day2 | A | 数据字段表和 `dataset_v1.csv` 本地样例 | 至少包含 `sample_id,text,image_path,label,source,split` |
| Day2 | A | 数据来源和脱敏说明 | 说明公开来源、标签映射、隐私处理 |
| Day2 | B | 图片预处理流程框架 | 能说明读取、RGB、resize、normalize、缺失图片处理 |
| Day2 | C | 实验记录表 | 模板字段完整，可从 Day3 开始填 |
| Day3 | A | 文本基线预测 CSV | 统一 5 列，`model_name=text_tfidf` |
| Day3-4 | B | 图像基线预测 CSV | 统一 5 列，`model_name=image_resnet` |
| Day3 | C | 第一版评估函数 | 能算 accuracy、precision、recall、F1、混淆矩阵 |

## 8. Git 协作建议

Day1 的 Git 目标是让三个人后续不会互相覆盖。

| 分支 | 负责人 | 用途 |
| --- | --- | --- |
| `main` | 全组 | 稳定版本，只通过 PR 合并 |
| `dev-a` | A | 数据、文本、行为特征 |
| `dev-b` | B | 图像、融合、结构图 |
| `dev-c` | C | 评估、图表、交接包 |

建议：

- 远程 `main` 分支保护需要 GitHub 权限，由有权限的同学在网页设置。
- 每天结束前各自在自己的 dev 分支提交代码或文档。
- PR 交叉 review：A 的 PR 由 B/C 看，B 的 PR 由 A/C 看，C 的 PR 由 A/B 看。
- 提交前必须检查 `git status --short`，确认没有 `.env`、真实数据、模型权重或大文件。

## 9. Day1 验收清单

- [ ] 标签方案已统一为 `normal` / `risk`。
- [ ] 数据字段表已明确：`sample_id,text,image_path,label,source,split`。
- [ ] `image_path` 已统一为仓库相对路径。
- [ ] 缺失图片处理策略已定为保留样本并使用零向量。
- [ ] 预测结果 CSV 已统一为 `sample_id,true_label,pred_label,risk_prob,model_name`。
- [ ] 目录结构已约定，真实数据、模型和输出不提交。
- [ ] 实验记录模板已建立。
- [ ] Day2-Day3 交接清单已明确。

## 10. 保底路线

- A 保底：公开文本数据集 + TF-IDF + 逻辑回归。
- B 保底：冻结 ResNet18 提取图像特征；融合困难时使用分数加权。
- C 保底：手动汇总 CSV，用 scikit-learn 计算指标和混淆矩阵。

先保证完整流程能跑通，再升级 BERT、ResNet50、MLP、OCR 或注意力融合。
