"""Day3/4 ResNet feature extraction scaffold for member B.

The script prepares image embeddings for multimodal fusion. It keeps samples
with missing or broken images by writing zero vectors, so A/B/C can keep
sample_id alignment.
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
RESNET18_DIM = 512
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureResult:
    sample_id: str
    true_label: str
    image_path: str
    status: str
    message: str
    embedding: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen ResNet image embeddings for Day3/4."
    )
    parser.add_argument(
        "--input",
        default="data/processed/dataset_v1.csv",
        help="CSV from member A. Must include sample_id,image_path,label,split.",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/image_embeddings.csv",
        help="Local embedding CSV. outputs/ is ignored by git.",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/image_resnet_pred.csv",
        help="Local placeholder prediction CSV for C's interface check.",
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
        help="Optionally process only the first N rows after split filtering.",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18"],
        help="Feature backbone. Day3 scaffold currently supports resnet18.",
    )
    parser.add_argument(
        "--weights",
        default="default",
        choices=["default", "none"],
        help="Use pretrained default weights or an untrained backbone for dry runs.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, such as test. Default processes all rows.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Placeholder threshold for the interface prediction CSV.",
    )
    return parser.parse_args()


def zero_embedding() -> list[float]:
    return [0.0] * RESNET18_DIM


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


def build_preprocess():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def build_resnet18(weights_mode: str):
    import torch
    from torchvision import models

    if weights_mode == "default":
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_one_feature(
    row: pd.Series,
    image_root: Path,
    model_cache: dict[str, object],
    weights_mode: str,
) -> FeatureResult:
    sample_id = str(row["sample_id"])
    true_label = str(row["label"])
    resolved_path, display_path = resolve_image_path(image_root, row.get("image_path"))

    if resolved_path is None:
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path="",
            status="missing_path",
            message="image_path is empty; using zero vector",
            embedding=zero_embedding(),
        )

    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="unsupported_format",
            message=f"expected jpg/jpeg/png, got {suffix or 'no extension'}; using zero vector",
            embedding=zero_embedding(),
        )

    if not resolved_path.exists():
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="file_not_found",
            message="image file does not exist; using zero vector",
            embedding=zero_embedding(),
        )

    try:
        import torch

        if "preprocess" not in model_cache:
            model_cache["preprocess"] = build_preprocess()
        if "model" not in model_cache:
            model_cache["model"] = build_resnet18(weights_mode)

        with Image.open(resolved_path) as image:
            image_rgb = image.convert("RGB")
            tensor = model_cache["preprocess"](image_rgb).unsqueeze(0)
            with torch.no_grad():
                embedding_tensor = model_cache["model"](tensor).squeeze(0)
        embedding = [float(value) for value in embedding_tensor.tolist()]
        if len(embedding) != RESNET18_DIM:
            raise RuntimeError(f"unexpected embedding length: {len(embedding)}")
    except (OSError, UnidentifiedImageError, RuntimeError, ImportError) as exc:
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="image_error",
            message=f"{exc}; using zero vector",
            embedding=zero_embedding(),
        )

    return FeatureResult(
        sample_id=sample_id,
        true_label=true_label,
        image_path=display_path,
        status="ok",
        message=f"extracted ResNet18 {RESNET18_DIM}-dim embedding",
        embedding=embedding,
    )


def make_embeddings_frame(results: list[FeatureResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "sample_id": result.sample_id,
            "image_path": result.image_path,
            "status": result.status,
            "message": result.message,
        }
        row.update(
            {
                f"img_emb_{index:03d}": value
                for index, value in enumerate(result.embedding)
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_prediction_frame(results: list[FeatureResult], threshold: float) -> pd.DataFrame:
    risk_prob = 0.5
    pred_label = "risk" if risk_prob >= threshold else "normal"
    return pd.DataFrame(
        [
            {
                "sample_id": result.sample_id,
                "true_label": result.true_label,
                "pred_label": pred_label,
                "risk_prob": risk_prob,
                "model_name": "image_resnet",
            }
            for result in results
        ]
    )


def run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    embeddings_output = Path(args.embeddings_output)
    pred_output = Path(args.pred_output)
    image_root = Path(args.image_root)

    if not input_path.exists():
        print(
            f"[WAITING_FOR_A] {input_path} not found. "
            "Ask member A for data/processed/dataset_v1.csv before ResNet features.",
            file=sys.stderr,
        )
        return 2

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.model != "resnet18":
        raise ValueError("Day3 scaffold currently supports only --model resnet18")

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

    if args.split:
        df = df[df["split"].astype(str) == args.split]

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        df = df.head(args.limit)

    model_cache: dict[str, object] = {}
    results = [
        extract_one_feature(
            row=row,
            image_root=image_root,
            model_cache=model_cache,
            weights_mode=args.weights,
        )
        for _, row in df.iterrows()
    ]

    embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    pred_output.parent.mkdir(parents=True, exist_ok=True)
    make_embeddings_frame(results).to_csv(embeddings_output, index=False, encoding="utf-8")
    make_prediction_frame(results, args.threshold).to_csv(
        pred_output, index=False, encoding="utf-8"
    )

    status_counts = pd.Series([result.status for result in results]).value_counts()
    summary = ", ".join(
        f"{status}={count}" for status, count in status_counts.sort_index().items()
    )
    print(f"[DONE] wrote embeddings -> {embeddings_output}")
    print(f"[DONE] wrote placeholder predictions -> {pred_output}")
    print(f"[SUMMARY] {summary if summary else 'no rows'}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
