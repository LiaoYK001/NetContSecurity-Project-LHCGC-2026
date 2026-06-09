"""Day2 image preprocessing check for member B.

This script validates the image_path column before the Day3/4 ResNet feature
step. It does not train a model or download weights.
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
    cached = getattr(build_preprocess, "_cached", None)
    if cached is not None:
        return cached

    from torchvision import transforms

    cached = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    build_preprocess._cached = cached
    return cached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image_path entries and run Day2 image preprocessing."
    )
    parser.add_argument(
        "--input",
        default="data/processed/dataset_v1.csv",
        help="CSV from member A. Must include sample_id,image_path,label,split.",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/image_preprocess_check.csv",
        help="Local check report CSV. outputs/ is ignored by git.",
    )
    parser.add_argument(
        "--image-root",
        default=".",
        help="Base directory for relative image_path values.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally check only the first N rows.",
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
            f"{input_path} is missing required columns: {', '.join(missing_columns)}"
        )


def check_one_image(row: pd.Series, image_root: Path) -> ImageCheckResult:
    sample_id = str(row["sample_id"])
    resolved_path, display_path = resolve_image_path(image_root, row.get("image_path"))

    if resolved_path is None:
        return ImageCheckResult(
            sample_id=sample_id,
            image_path="",
            status="missing_path",
            message="image_path is empty; keep sample and use zero vector later",
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
            message=f"expected jpg/jpeg/png, got {suffix or 'no extension'}",
            width=None,
            height=None,
            mode=None,
        )

    if not resolved_path.exists():
        return ImageCheckResult(
            sample_id=sample_id,
            image_path=display_path,
            status="file_not_found",
            message="image file does not exist; keep sample and use zero vector later",
            width=None,
            height=None,
            mode=None,
        )

    try:
        preprocess = build_preprocess()
        with Image.open(resolved_path) as image:
            original_width, original_height = image.size
            original_mode = image.mode
            image_rgb = image.convert("RGB")
            tensor = preprocess(image_rgb)
            if tuple(tensor.shape) != (3, 224, 224):
                raise RuntimeError(f"unexpected tensor shape: {tuple(tensor.shape)}")
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
        message="loaded, converted to RGB, resized to 224x224, normalized",
        width=original_width,
        height=original_height,
        mode=original_mode,
    )


def run_check(input_path: Path, output_path: Path, image_root: Path, limit: int | None) -> int:
    if not input_path.exists():
        print(
            f"[WAITING_FOR_A] {input_path} not found. "
            "Ask member A for data/processed/dataset_v1.csv before image checks.",
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
            raise ValueError("--limit must be a positive integer")
        df = df.head(limit)

    results = [check_one_image(row, image_root=image_root) for _, row in df.iterrows()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).to_csv(
        output_path, index=False, encoding="utf-8"
    )

    status_counts = pd.Series([result.status for result in results]).value_counts()
    summary = ", ".join(
        f"{status}={count}" for status, count in status_counts.sort_index().items()
    )
    print(f"[DONE] checked {len(results)} rows -> {output_path}")
    print(f"[SUMMARY] {summary if summary else 'no rows'}")
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
