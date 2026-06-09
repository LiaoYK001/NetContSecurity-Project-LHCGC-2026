"""Day3/4 成员 B：ResNet 图像特征提取。

为多模态融合准备图像向量。缺失、格式不支持或损坏的图片不会删除样本，
而是写入 512 维零向量，保证 A/B/C 后续都能按 ``sample_id`` 对齐。
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
        description="提取冻结 ResNet 图像向量，供 Day3/4 与后续融合使用。"
    )
    parser.add_argument(
        "--input",
        default="data/processed/dataset_v1.csv",
        help="成员 A 提供的 CSV，必须包含 sample_id,image_path,label,split。",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/image_embeddings.csv",
        help="本地图像向量 CSV。outputs/ 默认被 git 忽略。",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/image_resnet_pred.csv",
        help="给成员 C 检查接口用的本地占位预测 CSV。",
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
        help="可选：在 split 过滤后只处理前 N 行。",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["resnet18"],
        help="图像特征骨干网络。当前脚本仅支持 resnet18。",
    )
    parser.add_argument(
        "--weights",
        default="default",
        choices=["default", "none"],
        help="使用默认预训练权重，或用未训练骨架做流程试跑。",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="可选 split 过滤，例如 test；默认处理全部行。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="接口占位预测 CSV 使用的阈值。",
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
            f"{input_path} 缺少必要字段：{', '.join(missing_columns)}"
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
            message="image_path 为空；使用零向量",
            embedding=zero_embedding(),
        )

    suffix = resolved_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="unsupported_format",
            message=f"期望 jpg/jpeg/png，实际为 {suffix or '无扩展名'}；使用零向量",
            embedding=zero_embedding(),
        )

    if not resolved_path.exists():
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="file_not_found",
            message="图片文件不存在；使用零向量",
            embedding=zero_embedding(),
        )

    try:
        import torch

        # 延迟加载预处理和模型，避免缺图样本也触发权重加载。
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
            raise RuntimeError(f"图像向量长度异常：{len(embedding)}")
    except (OSError, UnidentifiedImageError, RuntimeError, ImportError) as exc:
        return FeatureResult(
            sample_id=sample_id,
            true_label=true_label,
            image_path=display_path,
            status="image_error",
            message=f"{exc}；使用零向量",
            embedding=zero_embedding(),
        )

    return FeatureResult(
        sample_id=sample_id,
        true_label=true_label,
        image_path=display_path,
        status="ok",
        message=f"已提取 ResNet18 {RESNET18_DIM} 维图像向量",
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
    # 当前没有训练图像分类头，所以预测文件只用于接口占位。
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
            f"[WAITING_FOR_A] 未找到 {input_path}。"
            "请先向成员 A 获取 data/processed/dataset_v1.csv，再提取 ResNet 特征。",
            file=sys.stderr,
        )
        return 2

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold 必须在 0 到 1 之间")
    if args.model != "resnet18":
        raise ValueError("当前 Day3/4 脚本仅支持 --model resnet18")

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
            raise ValueError("--limit 必须为正整数")
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
    print(f"[DONE] 已写出图像向量 -> {embeddings_output}")
    print(f"[DONE] 已写出占位预测 -> {pred_output}")
    print(f"[SUMMARY] {summary if summary else '无样本'}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
