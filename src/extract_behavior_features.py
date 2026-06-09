"""成员 A：构建多模态融合用的标准化行为向量。

读取 prepare_data.py 生成的 ``data/processed/behavior_features.csv``，
使用只在 train split 上 fit 的 StandardScaler 做标准化，
并输出给成员 B 融合使用的行为向量。

输出：
  - outputs/predictions/behavior_embeddings.csv     （sample_id + beh_emb_*，并保留 label/split/status/message）
  - outputs/predictions/behavior_feature_meta.json  （特征名与 scaler 统计）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

META_COLUMNS = {"sample_id", "label", "split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 behavior_features.csv 标准化为行为向量。"
    )
    parser.add_argument(
        "--behavior-input",
        default="data/processed/behavior_features.csv",
        help="prepare_data.py 生成的行为特征表。",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/behavior_embeddings.csv",
        help="给成员 B 融合使用的标准化行为向量。",
    )
    parser.add_argument(
        "--meta-output",
        default="outputs/predictions/behavior_feature_meta.json",
        help="特征名与 scaler 元数据。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 行，便于快速试跑。",
    )
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column not in META_COLUMNS]
    if not columns:
        raise ValueError("没有找到行为特征列。")
    return columns


def scale_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    matrix = df[feature_columns].to_numpy(dtype=np.float64)
    split_array = df["split"].astype(str).to_numpy()
    train_mask = split_array == "train"
    if not train_mask.any():
        raise ValueError("没有找到 split=train 的样本，无法 fit scaler。")

    # 只在训练集上 fit scaler，避免 val/test 信息泄漏到标准化参数中。
    scaler = StandardScaler()
    scaler.fit(matrix[train_mask])
    scaled = scaler.transform(matrix)

    meta = {
        "feature_names": feature_columns,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_rows": int(train_mask.sum()),
        "total_rows": int(len(df)),
    }
    return scaled.astype(np.float32), meta


def make_embeddings_frame(
    df: pd.DataFrame,
    scaled_matrix: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index in range(len(df)):
        row = df.iloc[row_index]
        payload: dict[str, object] = {
            "sample_id": str(row["sample_id"]),
            "label": str(row["label"]),
            "split": str(row["split"]),
            "status": "ok",
            "message": f"已标准化 {len(feature_columns)} 个行为特征",
        }
        for feat_index, value in enumerate(scaled_matrix[row_index]):
            payload[f"beh_emb_{feat_index:03d}"] = float(value)
        rows.append(payload)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    behavior_path = Path(args.behavior_input)
    if not behavior_path.exists():
        print(
            f"[WAITING_FOR_A] 缺少 {behavior_path}。请先运行 src/prepare_data.py 或等待成员 A 交付数据。",
            file=sys.stderr,
        )
        return 2

    behavior_df = pd.read_csv(
        behavior_path,
        dtype={"sample_id": "string", "label": "string", "split": "string"},
    )
    missing = sorted(META_COLUMNS - set(behavior_df.columns))
    if missing:
        raise ValueError(
            f"{behavior_path} 缺少必要字段：{', '.join(missing)}"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须为正数")
        behavior_df = behavior_df.head(args.limit)

    columns = feature_columns(behavior_df)
    for column in columns:
        behavior_df[column] = pd.to_numeric(behavior_df[column], errors="coerce").fillna(0.0)

    scaled_matrix, scaler_meta = scale_features(behavior_df, columns)

    embeddings_output = Path(args.embeddings_output)
    meta_output = Path(args.meta_output)
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)

    emb_frame = make_embeddings_frame(behavior_df, scaled_matrix, columns)
    emb_frame.to_csv(embeddings_output, index=False, encoding="utf-8")

    meta = {
        "feature_dim": len(columns),
        "feature_names": columns,
        "source_csv": str(behavior_path),
        "scaler": scaler_meta,
        "outputs": {
            "embeddings_csv": str(embeddings_output),
        },
    }
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] 已写出行为向量 -> {embeddings_output}")
    print(f"[DONE] 已写出元数据 -> {meta_output}")
    print(f"[SUMMARY] 行数={len(emb_frame)} 维度={len(columns)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
