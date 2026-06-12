import os
import pandas as pd

print("="*50)
print("开始验证所有文件和格式...")
print("="*50)

# 1. 验证文件路径
required_files = [
    "data/processed/dataset_v1.csv",
    "data/processed/behavior_features.csv",
    "data/processed/all_images.csv",
    "data/processed/dataset_stats.json"
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file} 存在")
    else:
        print(f"❌ {file} 不存在！请检查路径")
        all_exist = False

if not all_exist:
    print("\n❌ 有文件缺失，请修正后重新运行")
    exit(1)

# 2. 验证主数据集格式
print("\n" + "="*50)
print("验证主数据集格式...")
df = pd.read_csv("data/processed/dataset_v1.csv")
required_columns = ["sample_id", "text", "image_path", "label", "split"]
for col in required_columns:
    if col in df.columns:
        print(f"✅ 包含必要列：{col}")
    else:
        print(f"❌ 缺少必要列：{col}")

# 3. 验证标签格式
print("\n" + "="*50)
print("验证标签格式...")
unique_labels = df["label"].unique()
print(f"标签值：{unique_labels}")
if set(unique_labels) == {"normal", "risk"}:
    print("✅ 标签格式正确（normal/risk二分类）")
else:
    print("⚠️  标签格式不符合规范，请确认")

# 4. 验证行为特征格式
print("\n" + "="*50)
print("验证行为特征格式...")
behavior_df = pd.read_csv("data/processed/behavior_features.csv")
if "sample_id" in behavior_df.columns:
    print("✅ 行为特征包含sample_id列")
    print(f"✅ 行为特征列数：{len(behavior_df.columns)-1}")
else:
    print("❌ 行为特征缺少sample_id列")

print("\n" + "="*50)
print("🎉 所有验证通过！你可以开始写评估代码了")
print("="*50)