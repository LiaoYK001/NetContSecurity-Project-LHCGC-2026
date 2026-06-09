"""Day2 成员 B：图片路径与预处理检查。

在 Day3/4 提取 ResNet 特征前，检查 ``image_path`` 是否可读，
并验证 RGB 转换、224x224 resize 和 ImageNet normalize 流程。
本脚本不训练模型，也不下载权重。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError


REQUIRED_COLUMNS = {"sample_id", "image_path", "label", "split"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass
class ImageCheckResult:
    sample_id: str
    image_path: str
    status: str
    message: str
    width: int | None
    height: int | None
    mode: str | None


def build_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 image_path 字段，并运行 Day2 图片预处理流程。"
    )
    parser.add_argument(
        "--input",
        default="data/processed/dataset_v1.csv",
        help="成员 A 提供的 CSV，必须包含 sample_id,image_path,label,split。",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/image_preprocess_check.csv",
        help="本地检查报告 CSV。outputs/ 默认被 git 忽略。",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="相对 image_path 的基准目录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：只检查前 N 行。",
    )
    return parser.parse_args()


def resolve_image_path(image_root: Path, value: object) -> tuple[Path | None, str]:
    if pd.isna(value) or str(value).strip() == "":
        return None, ""

    raw_path = Path(str(value).strip())
    if raw_path.is_absolute():
        return raw_path, str(raw_path)
    return image_root / raw_path, str(raw_path)


def validate_dataframe(df: pd.DataFrame, input_path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"{input_path} 缺少必要字段：{', '.join(missing_columns)}"
        )


def check_one_image(
    row: pd.Series,
    image_root: Path,
    preprocess=None,
) -> ImageCheckResult:
    sample_id = str(row["sample_id"])
    resolved_path, display_path = resolve_image_path(image_root, row.get("image_path"))

    if resolved_path is None:
        return ImageCheckResult(
            sample_id=sample_id,
            image_path="",
            status="missing_path",
            message="image_path 为空；保留样本，后续使用零向量",
            width=None,
            height=None,
            mode=None,
        )

    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return ImageCheckResult(
            sample_id=sample_id,
            image_path=display_path,
            status="unsupported_format",
            message=f"期望 jpg/jpeg/png，实际为 {suffix or '无扩展名'}",
            width=None,
            height=None,
            mode=None,
        )

    if not resolved_path.exists():
        return ImageCheckResult(
            sample_id=sample_id,
            image_path=display_path,
            status="file_not_found",
            message="图片文件不存在；保留样本，后续使用零向量",
            width=None,
            height=None,
            mode=None,
        )

    if preprocess is None:
        preprocess = build_preprocess()

    try:
        with Image.open(resolved_path) as image:
            original_width, original_height = image.size
            original_mode = image.mode
            image_rgb = image.convert("RGB")
            tensor = preprocess(image_rgb)
            if tuple(tensor.shape) != (3, 224, 224):
                raise RuntimeError(f"预处理后的 tensor shape 异常：{tuple(tensor.shape)}")
    except (OSError, UnidentifiedImageError, RuntimeError, ImportError) as exc:
        return ImageCheckResult(
            sample_id=sample_id,
            image_path=display_path,
            status="image_error",
            message=str(exc),
            width=None,
            height=None,
            mode=None,
        )

    return ImageCheckResult(
        sample_id=sample_id,
        image_path=display_path,
        status="ok",
        message="已读取、转 RGB、resize 到 224x224 并 normalize",
        width=original_width,
        height=original_height,
        mode=original_mode,
    )


def run_check(input_path: Path, output_path: Path, image_root: Path, limit: int | None) -> int:
    if not input_path.exists():
        print(
            f"[WAITING_FOR_A] 未找到 {input_path}。"
            "请先向成员 A 获取 data/processed/dataset_v1.csv，再进行图片检查。",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(
        input_path,
        dtype={
            "sample_id": "string",
            "image_path": "string",
            "label": "string",
            "split": "string",
        },
    )
    validate_dataframe(df, input_path)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须为正整数")
        df = df.head(limit)

    # Day2 只验证图片能否进入统一预处理流程，不产生模型特征。
    preprocess = build_preprocess()
    results = [
        check_one_image(row, image_root=image_root, preprocess=preprocess)
        for _, row in df.iterrows()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).to_csv(
        output_path, index=False, encoding="utf-8"
    )

    status_counts = pd.Series([result.status for result in results]).value_counts()
    summary = ", ".join(
        f"{status}={count}" for status, count in status_counts.sort_index().items()
    )
    print(f"[DONE] 已检查 {len(results)} 行 -> {output_path}")
    print(f"[SUMMARY] {summary if summary else '无样本'}")
    return 0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root)
    return run_check(
        input_path=input_path,
        output_path=output_path,
        image_root=image_root,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
