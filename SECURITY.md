# Security Policy

本仓库是课程项目 public 仓库，主要风险来自三类内容：

1. 真实数据、原文、图片、账号、链接、cookie、token 等隐私或凭据。
2. 大模型、PyTorch、Transformers 等第三方依赖的供应链安全告警。
3. 来历不明的模型权重、pickle 文件或脚本在本地执行时带来的风险。

## Public Repository Boundary

不要提交：

```text
完整 data/processed/*.csv
tmp/ 原始数据
outputs/predictions/*.csv
outputs/handoff/text_tfidf_error_cases.csv
真实 sample_id、原文、用户链接、头像、原图
.env、cookie、token、账号密码
模型权重、Hugging Face 缓存、PyTorch 缓存
```

可以提交：

```text
代码
README 和教程
字段合同
data/sample/ 脱敏小样例
outputs/demo/ 汇总指标和图表
document/DE交接区/ public-safe 报告/PPT 交接材料
```

## PyTorch Security Alert

GitHub 可能提示：

```text
PyTorch is vulnerable to memory corruption through its torch.jit.script function
Affected package: torch
Affected versions: <= 2.12.0
Patched version: None
```

当前项目状态：

1. `pyproject.toml` 使用 CUDA 版 PyTorch，当前锁定为 `torch 2.11.0+cu128` 和 `torchvision 0.26.0+cu128`。
2. 本项目代码没有调用 `torch.jit.script`、`torch.jit.trace` 或 TorchScript 加载逻辑。
3. PyTorch 在本项目中只用于冻结 BERT / ResNet 特征提取，不执行用户上传模型代码。
4. GitHub 告警显示没有可用 patched version，因此不能通过简单升级彻底关闭该告警。

缓解措施：

1. 不在本项目中新增 `torch.jit.script` / TorchScript 相关代码。
2. 不加载不可信的 `.pt`、`.pth`、`.ckpt`、`.safetensors` 或 pickle 模型文件。
3. 只使用官方 Hugging Face / PyTorch 来源的公开预训练模型。
4. 模型权重和缓存只保存在本地，不进入 GitHub。
5. 等 PyTorch 上游发布 patched version 后，再升级 `torch` / `torchvision` 并重新生成 `uv.lock`。

结论：

> 当前不建议为了消除告警盲目改依赖版本，因为该 advisory 标注没有 patched version。现阶段应保留 PyTorch 依赖以保证 BERT / ResNet 流程可复现，同时通过“不使用 torch.jit.script、不加载不可信模型、不提交权重和缓存”的方式降低风险。

## Reporting a Vulnerability

如果发现仓库中误提交了敏感数据或凭据，请立即：

1. 停止继续传播该 commit 或压缩包。
2. 删除敏感文件并提交修复。
3. 如涉及 token、cookie、账号或密码，立即在对应平台轮换或撤销。
4. 通知项目负责人检查 GitHub history、PR diff 和 release/zip 包。
