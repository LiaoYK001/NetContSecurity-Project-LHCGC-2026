"""Build normalized behavior feature vectors for multimodal fusion (member A).

Reads ``data/processed/behavior_features.csv`` from prepare_data.py, applies
StandardScaler (fit on train split), and writes vectors for member B fusion.

Outputs:
  - outputs/predictions/behavior_embeddings.csv     (sample_id + beh_emb_*)
  - outputs/predictions/behavior_feature_meta.json  (feature names + scaler stats)
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
        description="Scale behavior_features.csv into normalized behavior vectors."
    )
    parser.add_argument(
        "--behavior-input",
        default="data/processed/behavior_features.csv",
        help="Behavior table from prepare_data.py.",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/behavior_embeddings.csv",
        help="Scaled behavior vectors for member B fusion.",
    )
    parser.add_argument(
        "--meta-output",
        default="outputs/predictions/behavior_feature_meta.json",
        help="Feature names and scaler metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (dry run).",
    )
    return parser.parse_args()


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column not in META_COLUMNS]
    if not columns:
        raise ValueError("No behavior feature columns found.")
    return columns


def scale_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    matrix = df[feature_columns].to_numpy(dtype=np.float64)
    split_array = df["split"].astype(str).to_numpy()
    train_mask = split_array == "train"
    if not train_mask.any():
        raise ValueError("No train split rows found for scaler fitting.")

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
            "message": f"scaled {len(feature_columns)} behavior features",
        }
        for feat_index, value in enumerate(scaled_matrix[row_index]):
            payload[f"beh_emb_{feat_index:03d}"] = float(value)
        rows.append(payload)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    behavior_path = Path(args.behavior_input)
    if not behavior_path.exists():
        print(f"[WAITING_FOR_A] missing {behavior_path}. Run src/prepare_data.py first.", file=sys.stderr)
        return 2

    behavior_df = pd.read_csv(
        behavior_path,
        dtype={"sample_id": "string", "label": "string", "split": "string"},
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
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

    print(f"[DONE] wrote behavior embeddings -> {embeddings_output}")
    print(f"[DONE] wrote meta -> {meta_output}")
    print(f"[SUMMARY] rows={len(emb_frame)} dim={len(columns)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
