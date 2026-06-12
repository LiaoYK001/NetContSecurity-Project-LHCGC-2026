"""为课程项目准备本地 EANN-KDD18 Weibo 数据。

Prepare local EANN-KDD18 Weibo data for this coursework project.

中文：本脚本把公开 Weibo 数据集转换成文本、图像、行为、融合和评估脚本共同使用的 CSV 合同。
English: This script converts the public Weibo dataset into the CSV contracts used by text, image, behavior, fusion, and evaluation scripts.

中文：脚本只写入本地生成文件，完整输出不要提交到 public 仓库。
English: The script writes local generated files only; do not commit full outputs to the public repository.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


SOURCE_NAME = "weibo_rumor_dataset"
REQUIRED_TWEET_FILES = {
    ("train", "risk"): "train_rumor.txt",
    ("train", "normal"): "train_nonrumor.txt",
    ("test", "risk"): "test_rumor.txt",
    ("test", "normal"): "test_nonrumor.txt",
}
SPLIT_PICKLES = {
    "train": "train_id.pickle",
    "val": "validate_id.pickle",
    "test": "test_id.pickle",
}


@dataclass
class ParsedRecord:
    sample_id: str
    text: str
    image_urls: list[str]
    label: str
    split: str
    verified: int
    reposts: float
    comments: float
    likes: float
    followers: float
    following: float
    posts_count: float
    source: str = SOURCE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate project processed CSVs from local EANN-KDD18 Weibo files."
    )
    parser.add_argument(
        "--weibo-root",
        default="weibo",
        help="Directory containing tweets/, train_id.pickle, validate_id.pickle, test_id.pickle, and image folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for dataset_v1.csv, all_images.csv, text_samples.csv, behavior_features.csv, and dataset_stats.json.",
    )
    parser.add_argument(
        "--handoff-dir",
        default="outputs/handoff",
        help="Directory for local helper files such as images_manifest.csv.",
    )
    return parser.parse_args()


def load_pickle_ids(weibo_root: Path) -> dict[str, str]:
    """读取官方 split pickle，建立 sample_id 到 split 的映射。

    Load official split pickle files and build the sample_id-to-split mapping.
    """
    sample_to_split: dict[str, str] = {}
    for split, filename in SPLIT_PICKLES.items():
        path = weibo_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        ids = payload.keys() if isinstance(payload, dict) else payload
        for raw_id in ids:
            sample_to_split[str(raw_id)] = split
    return sample_to_split


def as_float(value: str | None) -> float:
    if value is None:
        return 0.0
    value = str(value).strip()
    if value in {"", "null", "None", "nan"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", str(text))
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", " ", text)
    text = re.sub(r"1[3-9]\d{9}", "[PHONE]", text)
    text = re.sub(r"@\S+", "[USER]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_image_urls(raw: str) -> list[str]:
    urls = [part.strip() for part in str(raw).split("|")]
    return [url for url in urls if url and url != "null"]


def image_filename(url: str) -> str:
    return Path(urlparse(url).path).name


def resolve_local_image(weibo_root: Path, label: str, url: str) -> str:
    folder = "rumor_images" if label == "risk" else "nonrumor_images"
    filename = image_filename(url)
    if not filename:
        return ""
    candidate = weibo_root / folder / filename
    if not candidate.exists():
        return ""
    return str(Path("weibo") / folder / filename).replace("\\", "/")


def parse_record(meta_line: str, image_line: str, text_line: str, label: str, split: str) -> ParsedRecord:
    """把 Weibo 三行原始记录解析成项目内部记录对象。

    Parse one three-line Weibo record into the project internal record object.
    """
    parts = meta_line.rstrip("\n").split("|")
    if not parts or not parts[0].strip():
        raise ValueError(f"Invalid meta line: {meta_line!r}")

    return ParsedRecord(
        sample_id=parts[0].strip(),
        text=clean_text(text_line),
        image_urls=parse_image_urls(image_line),
        label=label,
        split=split,
        verified=1 if len(parts) > 5 and parts[5].strip().lower() == "true" else 0,
        reposts=as_float(parts[6] if len(parts) > 6 else None),
        comments=as_float(parts[7] if len(parts) > 7 else None),
        likes=as_float(parts[8] if len(parts) > 8 else None),
        followers=as_float(parts[11] if len(parts) > 11 else None),
        following=as_float(parts[12] if len(parts) > 12 else None),
        posts_count=as_float(parts[13] if len(parts) > 13 else None),
    )


def parse_tweet_file(path: Path, fallback_split: str, label: str, sample_to_split: dict[str, str]) -> list[ParsedRecord]:
    """读取一个 tweet 文本文件，并只保留官方 split 中出现的样本。

    Read one tweet text file and keep only samples listed in the official splits.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing tweet file: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) % 3 != 0:
        raise ValueError(f"{path} line count is not divisible by 3")

    records: list[ParsedRecord] = []
    for index in range(0, len(lines), 3):
        meta_line, image_line, text_line = lines[index : index + 3]
        sample_id = meta_line.split("|", 1)[0].strip()
        if sample_id not in sample_to_split:
            continue
        split = sample_to_split.get(sample_id, fallback_split)
        records.append(parse_record(meta_line, image_line, text_line, label, split))
    return records


def build_records(weibo_root: Path) -> list[ParsedRecord]:
    """汇总 rumor/nonrumor 文件，去重并生成统一样本列表。

    Combine rumor/nonrumor files, deduplicate them, and produce the unified sample list.
    """
    sample_to_split = load_pickle_ids(weibo_root)
    records: list[ParsedRecord] = []
    tweets_root = weibo_root / "tweets"
    for (fallback_split, label), filename in REQUIRED_TWEET_FILES.items():
        records.extend(parse_tweet_file(tweets_root / filename, fallback_split, label, sample_to_split))

    deduped: dict[str, ParsedRecord] = {}
    for record in records:
        if record.text:
            deduped[record.sample_id] = record
    return list(deduped.values())


def build_outputs(records: list[ParsedRecord], weibo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把内部记录转换成四张项目 CSV 表。

    Convert internal records into the four project CSV tables.
    """
    dataset_rows: list[dict[str, object]] = []
    text_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    for record in records:
        local_images = [resolve_local_image(weibo_root, record.label, url) for url in record.image_urls]
        local_images = [path for path in local_images if path]
        first_image = local_images[0] if local_images else ""

        dataset_rows.append(
            {
                "sample_id": record.sample_id,
                "text": record.text,
                "image_path": first_image,
                "label": record.label,
                "source": record.source,
                "split": record.split,
            }
        )
        text_rows.append(
            {
                "sample_id": record.sample_id,
                "text": record.text,
                "label": record.label,
                "split": record.split,
            }
        )
        text_length = len(record.text)
        url_mentions = len(record.image_urls)
        at_mentions = record.text.count("[USER]")
        engagement_total = record.reposts + record.comments + record.likes
        behavior_rows.append(
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "split": record.split,
                "verified": record.verified,
                "reposts": record.reposts,
                "comments": record.comments,
                "likes": record.likes,
                "engagement_total": engagement_total,
                "interaction_ratio": engagement_total / max(record.followers, 1.0),
                "followers": record.followers,
                "following": record.following,
                "posts_count": record.posts_count,
                "num_image_urls": len(record.image_urls),
                "text_length": text_length,
                "url_mentions": url_mentions,
                "at_mentions": at_mentions,
            }
        )

        if local_images:
            for image_index, image_path in enumerate(local_images):
                image_rows.append(
                    {
                        "sample_id": record.sample_id,
                        "image_path": image_path,
                        "image_index": image_index,
                        "label": record.label,
                        "split": record.split,
                    }
                )
        else:
            image_rows.append(
                {
                    "sample_id": record.sample_id,
                    "image_path": "",
                    "image_index": -1,
                    "label": record.label,
                    "split": record.split,
                }
            )

    dataset = pd.DataFrame(dataset_rows).sort_values("sample_id").reset_index(drop=True)
    text = pd.DataFrame(text_rows).sort_values("sample_id").reset_index(drop=True)
    behavior = pd.DataFrame(behavior_rows).sort_values("sample_id").reset_index(drop=True)
    images = pd.DataFrame(image_rows).sort_values(["sample_id", "image_index"]).reset_index(drop=True)
    return dataset, text, behavior, images


def validate_unique(df: pd.DataFrame, name: str) -> None:
    if df["sample_id"].duplicated().any():
        example = df.loc[df["sample_id"].duplicated(), "sample_id"].astype(str).iloc[0]
        raise ValueError(f"{name} sample_id duplicated, example: {example}")


def write_outputs(
    dataset: pd.DataFrame,
    text: pd.DataFrame,
    behavior: pd.DataFrame,
    images: pd.DataFrame,
    output_dir: Path,
    handoff_dir: Path,
) -> None:
    """写出 processed CSV、图片清单和数据统计。

    Write processed CSV files, the image manifest, and dataset statistics.
    """
    validate_unique(dataset, "dataset_v1.csv")
    validate_unique(text, "text_samples.csv")
    validate_unique(behavior, "behavior_features.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dir / "dataset_v1.csv", index=False, encoding="utf-8")
    text.to_csv(output_dir / "text_samples.csv", index=False, encoding="utf-8")
    behavior.to_csv(output_dir / "behavior_features.csv", index=False, encoding="utf-8")
    images.to_csv(output_dir / "all_images.csv", index=False, encoding="utf-8")
    dataset[["sample_id", "image_path", "label", "split"]].to_csv(
        handoff_dir / "images_manifest.csv",
        index=False,
        encoding="utf-8",
    )

    stats = {
        "parsed_records": int(len(dataset)),
        "split_records": int(len(dataset)),
        "exported_rows": int(len(dataset)),
        "empty_text_dropped": 0,
        "with_local_image": int(dataset["image_path"].astype(str).str.len().gt(0).sum()),
        "without_local_image": int(dataset["image_path"].astype(str).str.len().eq(0).sum()),
        "label_counts": {k: int(v) for k, v in dataset["label"].value_counts().items()},
        "split_counts": {k: int(v) for k, v in dataset["split"].value_counts().items()},
        "outputs": {
            "dataset_v1": str(output_dir / "dataset_v1.csv"),
            "behavior_features": str(output_dir / "behavior_features.csv"),
            "text_samples": str(output_dir / "text_samples.csv"),
            "all_images": str(output_dir / "all_images.csv"),
            "images_manifest": str(handoff_dir / "images_manifest.csv"),
        },
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    """执行 Weibo 数据转换主流程。

    Run the main Weibo data conversion workflow.
    """
    weibo_root = Path(args.weibo_root)
    if not weibo_root.exists():
        raise FileNotFoundError(f"Missing --weibo-root: {weibo_root}")

    records = build_records(weibo_root)
    dataset, text, behavior, images = build_outputs(records, weibo_root)
    write_outputs(dataset, text, behavior, images, Path(args.output_dir), Path(args.handoff_dir))
    print(f"[DONE] exported rows={len(dataset)}")
    print(f"[DONE] local images={dataset['image_path'].astype(str).str.len().gt(0).sum()}")
    print(f"[DONE] output dir={args.output_dir}")
    return 0


def main() -> int:
    """命令行入口。

    Command-line entry point.
    """
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
