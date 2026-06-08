# Day2 B 图像预处理进度说明

适用角色：成员 B  
日期：2026-06-08  
目标：完成图片预处理流程框架，为 Day3-4 的 ResNet 图像特征提取做准备。

## 1. 当前项目进度

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| Day1 A/B/C 接口规格 | 已完成 | 已明确标签、数据字段、图片规范、预测 CSV 格式和目录结构 |
| Day1 目录脚手架 | 已完成 | 已建立 `data/`、`outputs/`、`models/`、`src/` 占位结构 |
| Day1 实验记录模板 | 已完成 | 已建立 `outputs/metrics/experiment_log_template.csv` |
| Day2 B 图片预处理框架 | 已完成 | 新增 `src/extract_image_features.py` |
| Day2 A 数据交付 | 已完成 | 见 [Day2-A给B数据格式交接说明.md](Day2-A给B数据格式交接说明.md)；本地应有 `data/processed/dataset_v1.csv`（7723 行） |

## 2. B 今天完成的内容

新增脚本：`src/extract_image_features.py`

脚本作用：

- 读取 A 后续提供的 `data/processed/dataset_v1.csv`。
- 检查必要字段：`sample_id`、`image_path`、`label`、`split`。
- 使用仓库相对路径解析 `image_path`。
- 支持 `jpg`、`jpeg`、`png` 图片。
- 成功读取图片后转为 RGB，resize 到 `224x224`，并使用 ImageNet mean/std 做 normalize。
- 对空路径、文件不存在、格式不支持、图片损坏等情况生成检查记录，而不是直接中断。

默认命令：

```powershell
uv run python src/extract_image_features.py
```

可选只检查前 20 条：

```powershell
uv run python src/extract_image_features.py --limit 20
```

如果 A 的 CSV 放在其他位置：

```powershell
uv run python src/extract_image_features.py --input data/processed/dataset_v1.csv --image-root .
```

## 3. 输出文件说明

默认本地输出：

```text
outputs/predictions/image_preprocess_check.csv
```

输出字段：

```csv
sample_id,image_path,status,message,width,height,mode
```

状态含义：

| `status` | 含义 | 后续处理 |
| --- | --- | --- |
| `ok` | 图片可读取，预处理成功 | Day3 可进入 ResNet 特征提取 |
| `missing_path` | `image_path` 为空 | 后续图像向量填零 |
| `file_not_found` | 路径不为空但文件不存在 | 通知 A 核对路径，保底填零 |
| `unsupported_format` | 不是 `jpg/jpeg/png` | 优先让 A 转换格式或过滤 |
| `image_error` | 图片损坏或无法解码 | 记录样本，保底填零 |

注意：`outputs/` 中的真实检查结果不会提交到 Git。

## 4. A 数据交接（已就绪）

字段与文件含义见 **[Day2-A给B数据格式交接说明.md](Day2-A给B数据格式交接说明.md)**。

主表路径：

```text
data/processed/dataset_v1.csv
```

至少包含：

```csv
sample_id,text,image_path,label,source,split
```

B 实际需要读取：

```csv
sample_id,image_path,label,split
```

B 拿到本地 CSV 后立刻运行：

```powershell
uv run python src/extract_image_features.py --input data/processed/dataset_v1.csv --limit 20
```

若 `file_not_found` 很多，需要把检查结果反馈给 A，优先修正 `image_path`。

## 5. Day3 衔接事项

Day3-4 的目标不是继续改预处理规则，而是在这个基础上接入 ResNet18/ResNet50：

- 使用同一份 `dataset_v1.csv`。
- 对 `status=ok` 的图片提取图像向量。
- 对缺失或异常图片使用零向量。
- 输出图像向量维度说明，例如 `ResNet18 输出 512 维`。
- 后续图像预测 CSV 必须遵守 Day1 统一格式：

```csv
sample_id,true_label,pred_label,risk_prob,model_name
```

## 6. 今日验收清单

- [x] 图片预处理脚本已建立。
- [x] 脚本不会下载模型权重或训练模型。
- [x] 支持无数据时提示等待 A 的 CSV。
- [x] 支持缺失图片和损坏图片记录状态。
- [x] 输出路径位于 `outputs/`，不会提交真实检查结果。
- [ ] 使用本地 `dataset_v1.csv` 完成真实样本的前 20 条检查（见 Day2-A 交接说明）。
