**Git 常用指令与小组协作指南**

**1. 每天开始前**
先确认自己在哪个分支、工作区是否干净：

```powershell
git status
git branch
```

切到自己的分支：

```powershell
git checkout dev-b
```

如果还没有分支，先从当前分支新建：

```powershell
git checkout -b dev-b
```

如果远程 `main` 有更新，先同步：

```powershell
git checkout main
git pull origin main
git checkout dev-b
git merge main
```

**2. 查看改了什么**

查看简短状态：

```powershell
git status --short
```

查看具体改动：

```powershell
git diff
```

查看已经 `git add` 的改动：

```powershell
git diff --cached
```

**3. 添加文件**

添加指定文件，推荐这样，比较安全：

```powershell
git add document/Day1-ABC开发组接口与任务规格.md
git add .gitignore
git add outputs/metrics/experiment_log_template.csv
```

添加目录：

```powershell
git add data models outputs src
```

不建议无脑用 `git add .`，除非你先确认没有 `.env`、真实数据、模型权重。

**4. 提交前检查**

每次 commit 前必须跑：

```powershell
git status --short
```

可以提交的：

```text
代码文件
文档文件
.gitignore
.gitkeep
模板 CSV
```

不要提交的：

```text
.env
.venv/
data/里的真实数据
outputs/里的真实实验输出
models/里的模型权重
*.pt / *.pth / *.ckpt
```

**5. 提交**

```powershell
git commit -m "docs: complete day1 abc interface specification"
```

常见提交信息格式：

```text
docs: 更新文档
feat: 新增功能
fix: 修复问题
refactor: 重构代码
test: 添加测试
chore: 项目杂项配置
```

例子：

```powershell
git commit -m "feat: add text baseline training script"
git commit -m "docs: add data field specification"
git commit -m "fix: handle missing image paths"
```

**6. 推送到远程**

第一次推送新分支：

```powershell
git push -u origin dev-b
```

之后推送同一个分支：

```powershell
git push
```

**7. 开 Pull Request**

推荐流程：

```text
1. 推送自己的分支到 GitHub
2. 打开 GitHub 仓库
3. New pull request
4. base 选 main
5. compare 选自己的分支，比如 dev-b
6. 写清楚这次改了什么
7. 指定 A 或 C review
```

PR 描述可以这样写：

```text
本次完成 Day1 ABC 接口规格：
- 明确 normal/risk 标签
- 明确数据字段和预测 CSV 格式
- 明确图像输入规范
- 新增实验记录模板
- 更新 .gitignore，避免提交真实数据和模型
```

**8. 常用分支建议**

```text
main      稳定主分支
dev-a     A：数据、文本、行为特征
dev-b     B：图像、多模态融合
dev-c     C：评估、图表、交接包
```

如果是临时任务，也可以用更清楚的名字：

```powershell
git checkout -b dev-b-day1-abc-spec
```

**9. 出现 LF/CRLF warning 怎么办**

比如：

```text
warning: LF will be replaced by CRLF
```

这个通常不用管，是 Windows 换行符提示，不是错误。只要 `git status --short` 里没有敏感文件，就可以继续 commit。

**10. 最重要的安全检查**

提交前永远看一眼：

```powershell
git status --short
```

如果看到这些，先停下来：

```text
.env
.venv/
data/raw/xxx
data/processed/xxx.csv
outputs/predictions/xxx.csv
models/xxx.pt
```

这个项目是 public 仓库，原则是：提交代码、文档、模板；不提交真实数据、密钥、模型权重。
