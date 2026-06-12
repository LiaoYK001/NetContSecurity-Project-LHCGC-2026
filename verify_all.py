import pandas as pd
import os

# 定义评估指标汇总csv文件路径（由 src/evaluate.py 生成）
file_path = "outputs/metrics/ablation_metrics.csv"


def check_experiment_csv() -> bool:
    df = pd.read_csv(file_path, encoding="utf-8")
    print("=== 指标汇总文件读取完成 ===")
    print(f"数据集总行数：{len(df)}")
    print(f"字段列表：{list(df.columns)}")

    required_cols = ["model_group", "accuracy", "precision", "recall", "f1", "roc_auc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ 缺失必要字段：{missing}")
        return False

    print("✅ 字段校验通过")
    return True


if __name__ == "__main__":
    # 先判断文件是否存在
    if os.path.exists(file_path):
        ok = check_experiment_csv()
        raise SystemExit(0 if ok else 1)
    else:
        print(f"❌ 文件不存在：{file_path}，请先运行评估脚本生成日志")
        raise SystemExit(1)