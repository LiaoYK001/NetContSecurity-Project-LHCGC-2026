"""Day3 成员 A：训练 TF-IDF + 逻辑回归文本基线。

Day3 Member A: train a TF-IDF + Logistic Regression text baseline.

中文：读取 ``data/processed/text_samples.csv``，只在 ``split=train`` 上训练，并输出统一 5 列预测 CSV。
English: Read ``data/processed/text_samples.csv``, train only on ``split=train``, and export the unified 5-column prediction CSV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

REQUIRED_COLUMNS = {"sample_id", "text", "label", "split"}
MODEL_NAME = "text_tfidf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 TF-IDF + 逻辑回归文本基线。"
    )
    parser.add_argument(
        "--input",
        default="data/processed/text_samples.csv",
        help="输入 CSV，至少包含 sample_id,text,label,split。",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/text_tfidf_pred.csv",
        help="给成员 C 使用的统一 5 列预测 CSV。",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/metrics/text_tfidf_metrics.json",
        help="按 split 统计的指标 JSON。",
    )
    parser.add_argument(
        "--errors-output",
        default="outputs/handoff/text_tfidf_error_cases.csv",
        help="导出错分样本，供报告或 PPT 使用。",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="TF-IDF 最大词表大小。",
    )
    parser.add_argument(
        "--ngram-range",
        default="1,2",
        help="N-gram 范围，格式为 'min,max'，例如 1,2。",
    )
    parser.add_argument(
        "--error-limit",
        type=int,
        default=3,
        help="最多导出的 test 错分样本数量。",
    )
    return parser.parse_args()


def parse_ngram_range(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--ngram-range 必须写成 '1,2' 这样的格式")
    low, high = int(parts[0]), int(parts[1])
    if low <= 0 or high < low:
        raise ValueError("--ngram-range 必须满足 0 < min <= max")
    return low, high


def validate_dataframe(df: pd.DataFrame, path: Path) -> None:
    """校验文本输入表是否满足最小字段和标签要求。

    Validate that the text input table satisfies the minimum columns and label contract.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{sorted(missing)}")
    labels = set(df["label"].astype(str).unique())
    if not labels.issubset({"normal", "risk"}):
        raise ValueError(f"label 只能是 normal/risk，当前得到：{sorted(labels)}")


def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, float | dict[str, float]]:
    """计算单个 split 的文本分类指标。

    Compute text classification metrics for one split.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="risk", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="risk", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="risk", zero_division=0)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def export_error_cases(
    df: pd.DataFrame,
    pred_frame: pd.DataFrame,
    output_path: Path,
    limit: int,
) -> None:
    """导出少量错分样本，供 D/E 做错误案例分析。

    Export a small set of misclassified samples for D/E error-case analysis.
    """
    merged = pred_frame.merge(
        df[["sample_id", "text", "split"]],
        on="sample_id",
        how="left",
    )
    test_errors = merged[
        (merged["split"].astype(str) == "test")
        & (merged["true_label"].astype(str) != merged["pred_label"].astype(str))
    ].copy()
    if test_errors.empty:
        test_errors = merged[
            merged["true_label"].astype(str) != merged["pred_label"].astype(str)
        ].copy()
    test_errors = test_errors.sort_values("risk_prob", ascending=False).head(limit)
    test_errors["text_preview"] = (
        test_errors["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 160)
    )
    columns = [
        "sample_id",
        "split",
        "true_label",
        "pred_label",
        "risk_prob",
        "text_preview",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_errors[columns].to_csv(output_path, index=False, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    """训练文本保底模型并写出预测、指标和错例文件。

    Train the text fallback model and write predictions, metrics, and error cases.
    """
    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"[WAITING_FOR_A] 未找到 {input_path}。请先运行 src/prepare_data.py 或等待成员 A 交付数据。",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(
        input_path,
        dtype={
            "sample_id": "string",
            "text": "string",
            "label": "string",
            "split": "string",
        },
    )
    validate_dataframe(df, input_path)

    train_df = df[df["split"].astype(str) == "train"]
    if train_df.empty:
        raise ValueError("没有找到 split=train 的训练样本。")

    label_encoder = LabelEncoder()
    label_encoder.fit(["normal", "risk"])

    # 只用 train split 拟合 TF-IDF 和分类器，避免 val/test 信息泄漏。
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=parse_ngram_range(args.ngram_range),
        min_df=2,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_df["text"].fillna("").astype(str))
    y_train = train_df["label"].astype(str).to_numpy()

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(x_train, y_train)

    x_all = vectorizer.transform(df["text"].fillna("").astype(str))
    prob_matrix = classifier.predict_proba(x_all)
    risk_index = int(np.where(classifier.classes_ == "risk")[0][0])
    risk_probs = prob_matrix[:, risk_index]
    pred_labels = classifier.classes_[np.argmax(prob_matrix, axis=1)]

    # 统一写出 Day1 约定的 5 列预测格式，供成员 C 直接评估。
    pred_frame = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "true_label": df["label"].astype(str),
            "pred_label": pred_labels,
            "risk_prob": np.round(risk_probs, 6),
            "model_name": MODEL_NAME,
        }
    )

    pred_output = Path(args.pred_output)
    metrics_output = Path(args.metrics_output)
    errors_output = Path(args.errors_output)
    pred_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    pred_frame.to_csv(pred_output, index=False, encoding="utf-8")

    metrics: dict[str, object] = {
        "model_name": MODEL_NAME,
        "input_path": str(input_path),
        "pred_output": str(pred_output),
        "rows": int(len(df)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "class_distribution": {k: int(v) for k, v in df["label"].value_counts().items()},
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().items()},
    }
    split_metrics: dict[str, object] = {}
    label_order = ["normal", "risk"]
    for split_name in sorted(df["split"].astype(str).unique()):
        mask = df["split"].astype(str) == split_name
        split_metrics[split_name] = compute_split_metrics(
            y_true=df.loc[mask, "label"].astype(str).to_numpy(),
            y_pred=pred_frame.loc[mask, "pred_label"].astype(str).to_numpy(),
            labels=label_order,
        )
    metrics["splits"] = split_metrics
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    export_error_cases(df, pred_frame, errors_output, args.error_limit)

    test_metrics = split_metrics.get("test", {})
    print(f"[DONE] 已写出预测文件 -> {pred_output}")
    print(f"[DONE] 已写出指标文件 -> {metrics_output}")
    print(f"[DONE] 已写出错例文件 -> {errors_output}")
    if test_metrics:
        print(
            "[SUMMARY] test "
            f"acc={test_metrics.get('accuracy', 0):.4f} "
            f"prec={test_metrics.get('precision', 0):.4f} "
            f"rec={test_metrics.get('recall', 0):.4f} "
            f"f1={test_metrics.get('f1', 0):.4f}"
        )
    return 0


def main() -> int:
    """命令行入口。

    Command-line entry point.
    """
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
