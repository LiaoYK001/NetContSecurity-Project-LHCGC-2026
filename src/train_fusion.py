"""Day5 成员 B：训练多模态融合模型。

Day5 Member B: train the multimodal fusion model.

中文：按 ``sample_id`` 一对一合并文本、图像和行为特征，只在 ``split=train`` 上训练，并输出统一 5 列预测 CSV。
English: Merge text, image, and behavior features one-to-one by ``sample_id``, train only on ``split=train``, and export the unified 5-column prediction CSV.

中文：优先使用 BERT 文本向量；如果文本向量缺失，可退回 TF-IDF 风险概率作为保底文本特征。
English: Prefer BERT text embeddings; if they are missing, fall back to the TF-IDF risk probability as a text feature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


REQUIRED_DATASET_COLUMNS = {"sample_id", "label", "split"}
PREDICTION_COLUMNS = ["sample_id", "true_label", "pred_label", "risk_prob", "model_name"]
MODEL_NAME = "fusion_v1"
AVAILABLE_FEATURE_GROUPS = {"text", "image", "behavior"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 Day5 多模态融合模型，并输出统一 5 列预测 CSV。"
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/dataset_v1.csv",
        help="成员 A 的主数据表，至少包含 sample_id,label,split。",
    )
    parser.add_argument(
        "--text-embeddings",
        default="outputs/predictions/text_embeddings.csv",
        help="文本向量 CSV；存在时优先用于标准融合。",
    )
    parser.add_argument(
        "--text-pred",
        default="outputs/predictions/text_tfidf_pred.csv",
        help="文本基线预测 CSV；文本向量缺失时使用 risk_prob 做保底融合。",
    )
    parser.add_argument(
        "--image-embeddings",
        default="outputs/predictions/image_embeddings.csv",
        help="成员 B 生成的图像向量 CSV。",
    )
    parser.add_argument(
        "--behavior-embeddings",
        default="outputs/predictions/behavior_embeddings.csv",
        help="行为向量 CSV。",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/fusion_pred.csv",
        help="融合模型输出的统一 5 列预测 CSV。",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/metrics/fusion_metrics.json",
        help="融合模型按 split 统计的指标 JSON。",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "embeddings", "scores"],
        help="文本侧输入模式：auto 优先文本向量，scores 使用文本预测概率。",
    )
    parser.add_argument(
        "--feature-groups",
        default="text,image,behavior",
        help="参与训练的模态组合，用逗号分隔，可选 text,image,behavior。",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="写入预测 CSV 的 model_name。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只取前 N 行样本，便于快速试跑。",
    )
    return parser.parse_args()


def parse_feature_groups(raw: str) -> list[str]:
    raw_groups = [group.strip() for group in raw.split(",") if group.strip()]
    groups: list[str] = []
    for group in raw_groups:
        if group not in groups:
            groups.append(group)
    if not groups:
        raise ValueError("--feature-groups 至少需要包含一个模态")
    unknown = sorted(set(groups) - AVAILABLE_FEATURE_GROUPS)
    if unknown:
        raise ValueError(
            f"--feature-groups 包含未知模态：{', '.join(unknown)}；"
            "可选 text,image,behavior"
        )
    return groups


def read_dataset(path: Path, limit: int | None) -> pd.DataFrame:
    """读取融合主表，并校验 sample_id 唯一性。

    Read the fusion master table and validate sample_id uniqueness.
    """
    if not path.exists():
        print(
            f"[WAITING_FOR_A] 缺少 {path}。请先准备 dataset_v1.csv。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    df = pd.read_csv(path, dtype={"sample_id": "string", "label": "string", "split": "string"})
    missing = sorted(REQUIRED_DATASET_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{', '.join(missing)}")

    if df["sample_id"].duplicated().any():
        example = df.loc[df["sample_id"].duplicated(), "sample_id"].astype(str).iloc[0]
        raise ValueError(f"{path} sample_id 存在重复值，例如：{example}")

    labels = set(df["label"].astype(str).unique())
    if not labels.issubset({"normal", "risk"}):
        raise ValueError(f"{path} label 只能是 normal/risk，当前得到：{sorted(labels)}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit 必须为正数")
        df = df.head(limit)
    if df.empty:
        raise ValueError("没有可用于融合的样本。")
    return df[["sample_id", "label", "split"]].copy()


def embedding_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"没有找到 {prefix} 开头的向量列。")
    return columns


def merge_embeddings(base: pd.DataFrame, path: Path, prefix: str, name: str) -> tuple[pd.DataFrame, list[str]]:
    """按 sample_id 一对一合并某个模态的向量表。

    Merge one modality embedding table one-to-one by sample_id.
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少{name}：{path}")

    emb_df = pd.read_csv(path, dtype={"sample_id": "string"})
    if "sample_id" not in emb_df.columns:
        raise ValueError(f"{path} 缺少必要字段：sample_id")
    if emb_df["sample_id"].duplicated().any():
        example = emb_df.loc[emb_df["sample_id"].duplicated(), "sample_id"].astype(str).iloc[0]
        raise ValueError(f"{path} sample_id 存在重复值，例如：{example}")
    columns = embedding_columns(emb_df, prefix)
    merged = base.merge(emb_df[["sample_id", *columns]], on="sample_id", how="left", validate="one_to_one")
    for column in columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged, columns


def merge_text_scores(base: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    """合并 TF-IDF 文本预测概率，作为文本侧保底特征。

    Merge TF-IDF text prediction probability as the fallback text feature.
    """
    if not path.exists():
        raise FileNotFoundError(f"缺少文本预测概率：{path}")

    pred_df = pd.read_csv(path, dtype={"sample_id": "string"})
    missing = sorted({"sample_id", "risk_prob"} - set(pred_df.columns))
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{', '.join(missing)}")
    if pred_df["sample_id"].duplicated().any():
        example = pred_df.loc[pred_df["sample_id"].duplicated(), "sample_id"].astype(str).iloc[0]
        raise ValueError(f"{path} sample_id 存在重复值，例如：{example}")
    merged = base.merge(pred_df[["sample_id", "risk_prob"]], on="sample_id", how="left", validate="one_to_one")
    merged = merged.rename(columns={"risk_prob": "txt_score_risk_prob"})
    merged["txt_score_risk_prob"] = pd.to_numeric(
        merged["txt_score_risk_prob"], errors="coerce"
    ).fillna(0.5)
    return merged, ["txt_score_risk_prob"]


def choose_text_features(
    base: pd.DataFrame,
    text_embeddings_path: Path,
    text_pred_path: Path,
    mode: str,
) -> tuple[pd.DataFrame, list[str], str]:
    """根据模式选择 BERT 文本向量或 TF-IDF 保底分数。

    Choose BERT text embeddings or the TF-IDF fallback score according to the mode.
    """
    if mode in {"auto", "embeddings"} and text_embeddings_path.exists():
        merged, columns = merge_embeddings(base, text_embeddings_path, "txt_emb_", "文本向量")
        return merged, columns, "text_embeddings"

    if mode == "embeddings":
        raise FileNotFoundError(f"指定了 --mode embeddings，但缺少 {text_embeddings_path}")

    merged, columns = merge_text_scores(base, text_pred_path)
    return merged, columns, "text_tfidf_score"


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="risk", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="risk", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label="risk", zero_division=0)),
    }


def run(args: argparse.Namespace) -> int:
    """执行多模态融合训练和预测导出主流程。

    Run the main multimodal fusion training and prediction export workflow.
    """
    dataset_path = Path(args.dataset)
    text_embeddings_path = Path(args.text_embeddings)
    text_pred_path = Path(args.text_pred)
    image_embeddings_path = Path(args.image_embeddings)
    behavior_embeddings_path = Path(args.behavior_embeddings)
    pred_output = Path(args.pred_output)
    metrics_output = Path(args.metrics_output)

    base = read_dataset(dataset_path, args.limit)
    selected_groups = parse_feature_groups(args.feature_groups)

    feature_frame = base.copy()
    text_columns: list[str] = []
    image_columns: list[str] = []
    behavior_columns: list[str] = []
    text_source = "not_used"

    if "text" in selected_groups:
        # 文本侧优先走向量拼接，缺少文本向量时用 TF-IDF 概率保底。
        feature_frame, text_columns, text_source = choose_text_features(
            base=feature_frame,
            text_embeddings_path=text_embeddings_path,
            text_pred_path=text_pred_path,
            mode=args.mode,
        )
    if "image" in selected_groups:
        feature_frame, image_columns = merge_embeddings(
            feature_frame, image_embeddings_path, "img_emb_", "图像向量"
        )
    if "behavior" in selected_groups:
        feature_frame, behavior_columns = merge_embeddings(
            feature_frame, behavior_embeddings_path, "beh_emb_", "行为向量"
        )

    feature_columns = [*text_columns, *image_columns, *behavior_columns]
    if not feature_columns:
        raise ValueError("没有可用于训练的融合特征列。")
    split_array = feature_frame["split"].astype(str).to_numpy()
    train_mask = split_array == "train"
    if not train_mask.any():
        raise ValueError("没有找到 split=train 的样本，无法训练融合模型。")

    x_all = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    y_all = feature_frame["label"].astype(str).to_numpy()

    # scaler 只在 train split 上 fit，避免验证集/测试集信息泄漏。
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_all[train_mask])
    x_scaled = scaler.transform(x_all)

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    classifier.fit(x_train, y_all[train_mask])

    risk_index = int(np.where(classifier.classes_ == "risk")[0][0])
    prob_matrix = classifier.predict_proba(x_scaled)
    risk_prob = prob_matrix[:, risk_index]
    pred_label = classifier.classes_[np.argmax(prob_matrix, axis=1)]

    pred_frame = pd.DataFrame(
        {
            "sample_id": feature_frame["sample_id"].astype(str),
            "true_label": y_all,
            "pred_label": pred_label,
            "risk_prob": np.round(risk_prob, 6),
            "model_name": args.model_name,
        },
        columns=PREDICTION_COLUMNS,
    )

    pred_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pred_frame.to_csv(pred_output, index=False, encoding="utf-8")

    metrics = {
        "model_name": args.model_name,
        "selected_feature_groups": selected_groups,
        "text_source": text_source,
        "rows": int(len(feature_frame)),
        "feature_dim": int(len(feature_columns)),
        "feature_groups": {
            "text": len(text_columns),
            "image": len(image_columns),
            "behavior": len(behavior_columns),
        },
        "inputs": {
            "dataset": str(dataset_path),
            "text_embeddings": str(text_embeddings_path),
            "text_pred": str(text_pred_path),
            "image_embeddings": str(image_embeddings_path),
            "behavior_embeddings": str(behavior_embeddings_path),
        },
        "splits": {},
    }
    for split_name in sorted(feature_frame["split"].astype(str).unique()):
        mask = feature_frame["split"].astype(str) == split_name
        metrics["splits"][split_name] = split_metrics(
            y_true=y_all[mask.to_numpy()],
            y_pred=pred_frame.loc[mask, "pred_label"].astype(str).to_numpy(),
        )
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] 已写出融合预测 -> {pred_output}")
    print(f"[DONE] 已写出融合指标 -> {metrics_output}")
    print(
        f"[SUMMARY] 行数={len(feature_frame)} 特征维度={len(feature_columns)} "
        f"文本来源={text_source}"
    )
    return 0


def main() -> int:
    """命令行入口。

    Command-line entry point.
    """
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
