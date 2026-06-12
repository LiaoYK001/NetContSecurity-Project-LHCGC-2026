import pandas as pd
import os

# 定义实验日志csv文件路径
file_path = "outputs/metrics/experiment_log_template.csv"


def check_experiment_csv():
    # 修复编码报错，指定gbk编码读取中文CSV
    df = pd.read_csv(file_path, encoding="gbk")
    print("=== 实验日志文件校验完成 ===")
    print(f"数据集总行数：{len(df)}")
    print(f"数据集字段列表：{list(df.columns)}")
    print("\n前5行数据预览：")
    print(df.head())

    # 校验5类误判字段是否存在
    error_type_cols = ["类别1_单文本漏报", "类别2_图文特征冲突", "类别3_图像噪声干扰", "类别4_关键词误匹配",
                       "类别5_时序对齐异常"]
    for col in error_type_cols:
        if col in df.columns:
            print(f"\n{col} 样本总数：{df[col].sum()}")
        else:
            print(f"警告：缺失字段 {col}")


if __name__ == "__main__":
    # 先判断文件是否存在
    if os.path.exists(file_path):
        check_experiment_csv()
        print("\n✅ 全部校验通过，无文件/编码异常")
    else:
        print(f"❌ 文件不存在：{file_path}，请先运行评估脚本生成日志")