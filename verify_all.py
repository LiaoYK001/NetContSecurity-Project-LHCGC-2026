from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

METRICS_PATH = Path("outputs/metrics/ablation_metrics.csv")
REQUIRED_COLUMNS = ["model_group", "sample_count", "accuracy", "precision", "recall", "f1", "roc_auc"]
EXPECTED_MODELS = {"text_only", "image_only", "behavior_only", "text_image", "text_image_behavior"}


def check_metrics_csv() -> bool:
    df = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")
    print("=== 消融指标汇总文件校验 ===")
    print(f"数据行数：{len(df)}")
    print(f"字段列表：{list(df.columns)}")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        print(f"[ERROR] 缺失必要字段：{missing}")
        return False

    actual_models = set(df["model_group"].astype(str))
    missing_models = sorted(EXPECTED_MODELS - actual_models)
    if missing_models:
        print(f"[ERROR] 缺少必要消融组：{missing_models}")
        return False

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    out_of_range = {}
    for column in metric_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        bad_rows = df[numeric.isna() | (numeric < 0) | (numeric > 1)]["model_group"].tolist()
        if bad_rows:
            out_of_range[column] = bad_rows
    if out_of_range:
        print(f"[ERROR] 指标存在非 0-1 范围数值或空值：{out_of_range}")
        return False

    print("[OK] 消融指标字段、模型组和数值范围校验通过")
    print("\n前5行预览：")
    print(df.head())
    return True


if __name__ == "__main__":
    if not os.path.exists(METRICS_PATH):
        print(f"[ERROR] 文件不存在：{METRICS_PATH}")
        print("请先运行：uv run python src/evaluate.py")
        raise SystemExit(1)

    raise SystemExit(0 if check_metrics_csv() else 1)
