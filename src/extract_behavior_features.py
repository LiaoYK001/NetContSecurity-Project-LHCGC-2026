"""Build normalized behavior feature vectors for multimodal fusion (member A).

Enriches ``data/processed/behavior_features.csv`` with derived signals:
  - link count, @mention count, image URL count
  - text duplicate / repeat ratio within the dataset
  - user posting frequency proxies from raw Weibo metadata
  - engagement composition and account ratio features

Outputs:
  - data/processed/behavior_features_enriched.csv   (human-readable raw features)
  - outputs/predictions/behavior_embeddings.csv     (sample_id + beh_emb_*)
  - outputs/predictions/behavior_feature_meta.json  (feature names + scaler stats)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TWEET_FILES: tuple[tuple[str, bool], ...] = (
    ("tweets/train_nonrumor.txt", False),
    ("tweets/train_rumor.txt", True),
    ("tweets/test_nonrumor.txt", False),
    ("tweets/test_rumor.txt", True),
)

SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "诈骗",
    "中奖",
    "转账",
    "汇款",
    "兼职",
    "刷单",
    "贷款",
    "免息",
    "红包",
    "领取",
    "点击链接",
    "加微信",
    "加qq",
    "验证码",
    "银行卡",
    "投资",
    "返利",
    "代购",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract enriched and normalized behavior feature vectors."
    )
    parser.add_argument(
        "--behavior-input",
        default="data/processed/behavior_features.csv",
        help="Base behavior table from prepare_data.py.",
    )
    parser.add_argument(
        "--text-input",
        default="data/processed/text_samples.csv",
        help="Cleaned text table for duplicate/repeat statistics.",
    )
    parser.add_argument(
        "--dataset-input",
        default="data/processed/dataset_v1.csv",
        help="Main dataset for image-presence feature.",
    )
    parser.add_argument(
        "--weibo-root",
        default="weibo",
        help="Raw Weibo directory for user/timestamp metadata.",
    )
    parser.add_argument(
        "--enriched-output",
        default="data/processed/behavior_features_enriched.csv",
        help="Raw enriched behavior features before scaling.",
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
        help="Process only the first N rows after merge (dry run).",
    )
    return parser.parse_args()


def null_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if stripped == "" or stripped.lower() == "null":
        return None
    return stripped


def parse_timestamp(raw: str | None) -> float | None:
    cleaned = null_to_none(raw)
    if cleaned is None:
        return None
    if cleaned.isdigit():
        millis = int(cleaned)
        return millis / 1000.0 if millis > 10_000_000_000 else float(millis)
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt).timestamp()
        except ValueError:
            continue
    return None


def load_user_activity(weibo_root: Path) -> pd.DataFrame:
    """Return per-tweet user activity fields and per-user posting frequency stats."""
    rows: list[dict[str, object]] = []
    for rel_path, is_rumor in TWEET_FILES:
        path = weibo_root / rel_path
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        index = 0
        while index < len(lines):
            meta_line = lines[index].strip()
            if not meta_line:
                index += 1
                continue
            if index + 2 >= len(lines):
                break
            parts = meta_line.split("|")
            tweet_id = parts[0].strip()
            timestamp_raw = parts[4] if len(parts) > 4 else None
            user_id = null_to_none(parts[9]) if len(parts) > 9 else None
            if not user_id and is_rumor and len(parts) > 3:
                user_id = null_to_none(parts[3])
            rows.append(
                {
                    "sample_id": tweet_id,
                    "user_id": user_id or "",
                    "timestamp_epoch": parse_timestamp(timestamp_raw),
                }
            )
            index += 3

    if not rows:
        return pd.DataFrame(columns=["sample_id", "user_post_count", "user_post_frequency", "hours_since_first_post"])

    activity_df = pd.DataFrame(rows)
    valid_time = activity_df.dropna(subset=["timestamp_epoch"]).copy()
    user_stats: dict[str, dict[str, float]] = {}
    for user_id, group in valid_time.groupby("user_id"):
        if not user_id:
            continue
        timestamps = sorted(group["timestamp_epoch"].astype(float).tolist())
        count = len(timestamps)
        if count <= 1:
            mean_gap_hours = float(count)
        else:
            gaps = [
                (timestamps[i] - timestamps[i - 1]) / 3600.0
                for i in range(1, len(timestamps))
            ]
            mean_gap_hours = float(np.mean(gaps))
        user_stats[user_id] = {
            "user_post_count": float(count),
            "user_post_frequency": float(1.0 / max(mean_gap_hours, 1.0)),
            "first_post_epoch": float(timestamps[0]),
        }

    def lookup_user_stat(user_id: str, key: str, default: float = 0.0) -> float:
        if not user_id:
            return default
        return float(user_stats.get(user_id, {}).get(key, default))

    activity_df["user_post_count"] = activity_df["user_id"].map(
        lambda uid: lookup_user_stat(uid, "user_post_count")
    )
    activity_df["user_post_frequency"] = activity_df["user_id"].map(
        lambda uid: lookup_user_stat(uid, "user_post_frequency")
    )

    def hours_since_first(row: pd.Series) -> float:
        ts = row["timestamp_epoch"]
        uid = row["user_id"]
        if pd.isna(ts) or not uid:
            return 0.0
        first_epoch = user_stats.get(uid, {}).get("first_post_epoch")
        if first_epoch is None:
            return 0.0
        return max((float(ts) - float(first_epoch)) / 3600.0, 0.0)

    activity_df["hours_since_first_post"] = activity_df.apply(hours_since_first, axis=1)
    return activity_df[
        ["sample_id", "user_post_count", "user_post_frequency", "hours_since_first_post"]
    ]


def count_sensitive_words(text: str) -> int:
    lowered = text.lower()
    return sum(1 for keyword in SENSITIVE_KEYWORDS if keyword in lowered or keyword in text)


def build_enriched_features(
    behavior_df: pd.DataFrame,
    text_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    activity_df: pd.DataFrame,
) -> pd.DataFrame:
    text_map = text_df.set_index("sample_id")["text"].astype(str).to_dict()
    dup_sizes = text_df.groupby("text")["sample_id"].transform("count")
    text_df = text_df.copy()
    text_df["text_dup_group_size"] = dup_sizes.values

    image_presence = dataset_df.assign(
        has_local_image=dataset_df["image_path"].fillna("").astype(str).str.strip().ne("").astype(int)
    )[["sample_id", "has_local_image"]]

    merged = behavior_df.merge(
        text_df[["sample_id", "text", "text_dup_group_size"]],
        on="sample_id",
        how="left",
    )
    merged = merged.merge(image_presence, on="sample_id", how="left")
    merged = merged.merge(activity_df, on="sample_id", how="left")

    merged["link_count"] = merged["url_mentions"].fillna(0).astype(float)
    merged["mention_count"] = merged["at_mentions"].fillna(0).astype(float)
    merged["repeat_ratio"] = (
        (merged["text_dup_group_size"].fillna(1).astype(float) - 1.0)
        / max(len(merged) - 1, 1)
    ).clip(0.0, 1.0)
    merged["is_duplicate_text"] = (merged["text_dup_group_size"].fillna(1) > 1).astype(int)
    merged["sensitive_word_count"] = merged["text"].fillna("").map(count_sensitive_words)

    merged["fan_follow_ratio"] = merged["followers"].astype(float) / (
        merged["following"].astype(float).clip(lower=1.0)
    )
    merged["engagement_per_char"] = merged["engagement_total"].astype(float) / (
        merged["text_length"].astype(float).clip(lower=1.0)
    )
    merged["engagement_per_follower"] = merged["engagement_total"].astype(float) / (
        merged["followers"].astype(float).clip(lower=1.0)
    )
    merged["like_share"] = merged["likes"].astype(float) / (
        merged["engagement_total"].astype(float).clip(lower=1.0)
    )
    merged["comment_share"] = merged["comments"].astype(float) / (
        merged["engagement_total"].astype(float).clip(lower=1.0)
    )
    merged["repost_share"] = merged["reposts"].astype(float) / (
        merged["engagement_total"].astype(float).clip(lower=1.0)
    )
    merged["image_link_ratio"] = merged["num_image_urls"].astype(float) / (
        merged["link_count"] + merged["num_image_urls"].astype(float) + 1.0
    )

    for column in ("user_post_count", "user_post_frequency", "hours_since_first_post"):
        merged[column] = merged[column].fillna(0.0)

    feature_columns = [
        "verified",
        "reposts",
        "comments",
        "likes",
        "engagement_total",
        "interaction_ratio",
        "followers",
        "following",
        "posts_count",
        "num_image_urls",
        "text_length",
        "link_count",
        "mention_count",
        "fan_follow_ratio",
        "engagement_per_char",
        "engagement_per_follower",
        "like_share",
        "comment_share",
        "repost_share",
        "repeat_ratio",
        "is_duplicate_text",
        "user_post_count",
        "user_post_frequency",
        "hours_since_first_post",
        "has_local_image",
        "image_link_ratio",
        "sensitive_word_count",
    ]
    for column in feature_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    keep_columns = ["sample_id", "label", "split", *feature_columns]
    return merged[keep_columns]


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
    text_path = Path(args.text_input)
    dataset_path = Path(args.dataset_input)
    weibo_root = Path(args.weibo_root)

    for path in (behavior_path, text_path, dataset_path):
        if not path.exists():
            print(f"[WAITING_FOR_A] missing {path}. Run src/prepare_data.py first.", file=sys.stderr)
            return 2

    behavior_df = pd.read_csv(
        behavior_path,
        dtype={"sample_id": "string", "label": "string", "split": "string"},
    )
    text_df = pd.read_csv(
        text_path,
        dtype={"sample_id": "string", "text": "string", "label": "string", "split": "string"},
    )
    dataset_df = pd.read_csv(
        dataset_path,
        dtype={"sample_id": "string", "image_path": "string", "label": "string", "split": "string"},
    )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        behavior_df = behavior_df.head(args.limit)
        sample_ids = set(behavior_df["sample_id"].astype(str))
        text_df = text_df[text_df["sample_id"].astype(str).isin(sample_ids)]
        dataset_df = dataset_df[dataset_df["sample_id"].astype(str).isin(sample_ids)]

    activity_df = (
        load_user_activity(weibo_root)
        if weibo_root.exists()
        else pd.DataFrame(
            columns=["sample_id", "user_post_count", "user_post_frequency", "hours_since_first_post"]
        )
    )

    enriched_df = build_enriched_features(behavior_df, text_df, dataset_df, activity_df)
    feature_columns = [column for column in enriched_df.columns if column not in {"sample_id", "label", "split"}]
    scaled_matrix, scaler_meta = scale_features(enriched_df, feature_columns)

    enriched_output = Path(args.enriched_output)
    embeddings_output = Path(args.embeddings_output)
    meta_output = Path(args.meta_output)
    enriched_output.parent.mkdir(parents=True, exist_ok=True)
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)

    enriched_df.to_csv(enriched_output, index=False, encoding="utf-8")
    emb_frame = make_embeddings_frame(enriched_df, scaled_matrix, feature_columns)
    emb_frame.to_csv(embeddings_output, index=False, encoding="utf-8")

    meta = {
        "feature_dim": len(feature_columns),
        "feature_names": feature_columns,
        "scaler": scaler_meta,
        "outputs": {
            "enriched_csv": str(enriched_output),
            "embeddings_csv": str(embeddings_output),
        },
    }
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote enriched features -> {enriched_output}")
    print(f"[DONE] wrote behavior embeddings -> {embeddings_output}")
    print(f"[DONE] wrote meta -> {meta_output}")
    print(f"[SUMMARY] rows={len(emb_frame)} dim={len(feature_columns)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
