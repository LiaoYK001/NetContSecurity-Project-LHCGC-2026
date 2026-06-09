"""Day6 成员 B：一键生成消融实验预测和指标汇总。

本脚本复用 ``train_fusion.py`` 的训练逻辑，分别跑文本、图像、行为、
文本+图像、文本+图像+行为五组实验，输出给成员 C 画图出表使用。
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from train_fusion import run as run_fusion


ABLATION_CONFIGS = [
    ("text_only", "text", "仅文本"),
    ("image_only", "image", "仅图像"),
    ("behavior_only", "behavior", "仅行为"),
    ("text_image", "text,image", "文本+图像"),
    ("text_image_behavior", "text,image,behavior", "文本+图像+行为"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行 Day6 五组消融实验，并生成指标汇总。"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/dataset_v1.csv",
        help="成员 A 的主数据表，至少包含 sample_id,label,split。",
    )
    parser.add_argument(
        "--text-embeddings",
        default="outputs/predictions/text_embeddings.csv",
        help="文本向量 CSV；存在时可用于标准消融。",
    )
    parser.add_argument(
        "--text-pred",
        default="outputs/predictions/text_tfidf_pred.csv",
        help="文本基线预测 CSV；保底消融使用其中的 risk_prob。",
    )
    parser.add_argument(
        "--image-embeddings",
        default="outputs/predictions/image_embeddings.csv",
        help="ResNet 图像向量 CSV。",
    )
    parser.add_argument(
        "--behavior-embeddings",
        default="outputs/predictions/behavior_embeddings.csv",
        help="行为向量 CSV。",
    )
    parser.add_argument(
        "--pred-dir",
        default="outputs/predictions/ablation",
        help="消融预测 CSV 输出目录。",
    )
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics",
        help="消融指标 JSON 输出目录。",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/metrics/ablation_summary.csv",
        help="消融指标汇总 CSV。",
    )
    parser.add_argument(
        "--summary-json",
        default="outputs/metrics/ablation_summary.json",
        help="消融指标汇总 JSON。",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "embeddings", "scores"],
        help="文本侧输入模式：auto 优先文本向量，scores 使用文本预测概率。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只取前 N 行样本，便于快速试跑。",
    )
    return parser.parse_args()


def build_fusion_args(
    args: argparse.Namespace,
    name: str,
    feature_groups: str,
) -> Namespace:
    pred_dir = Path(args.pred_dir)
    metrics_dir = Path(args.metrics_dir)
    return Namespace(
        dataset=args.dataset,
        text_embeddings=args.text_embeddings,
        text_pred=args.text_pred,
        image_embeddings=args.image_embeddings,
        behavior_embeddings=args.behavior_embeddings,
        pred_output=str(pred_dir / f"{name}_pred.csv"),
        metrics_output=str(metrics_dir / f"ablation_{name}_metrics.json"),
        mode=args.mode,
        feature_groups=feature_groups,
        model_name=f"ablation_{name}",
        limit=args.limit,
    )


def summarize_one(name: str, label: str, metrics_path: Path, pred_path: Path) -> dict[str, object]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_metrics = metrics["splits"].get("test", {})
    return {
        "ablation_name": name,
        "display_name": label,
        "model_name": metrics["model_name"],
        "text_source": metrics["text_source"],
        "feature_dim": metrics["feature_dim"],
        "text_dim": metrics["feature_groups"]["text"],
        "image_dim": metrics["feature_groups"]["image"],
        "behavior_dim": metrics["feature_groups"]["behavior"],
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "prediction_csv": str(pred_path),
        "metrics_json": str(metrics_path),
    }


def run(args: argparse.Namespace) -> int:
    pred_dir = Path(args.pred_dir)
    metrics_dir = Path(args.metrics_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for name, feature_groups, label in ABLATION_CONFIGS:
        fusion_args = build_fusion_args(args, name, feature_groups)
        print(f"[RUN] {label} -> {fusion_args.pred_output}")
        run_fusion(fusion_args)
        pred_path = Path(fusion_args.pred_output)
        metrics_path = Path(fusion_args.metrics_output)
        rows.append(summarize_one(name, label, metrics_path, pred_path))

    summary_frame = pd.DataFrame(rows)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(summary_csv, index=False, encoding="utf-8")
    summary_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] 已写出消融汇总 CSV -> {summary_csv}")
    print(f"[DONE] 已写出消融汇总 JSON -> {summary_json}")
    print(f"[SUMMARY] 消融组合数={len(rows)}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
