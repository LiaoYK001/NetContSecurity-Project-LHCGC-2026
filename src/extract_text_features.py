"""成员 A：使用冻结 BERT / RoBERTa 提取中文文本向量。

Member A: extract Chinese text embeddings with a frozen BERT / RoBERTa encoder.

中文：读取文本 CSV，输出 ``text_embeddings.csv``，默认不微调大模型，只把文本变成向量供融合使用。
English: Read the text CSV and export ``text_embeddings.csv``; by default, the encoder is frozen and only produces features for fusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

REQUIRED_COLUMNS = {"sample_id", "text", "label", "split"}

MODEL_PRESETS: dict[str, dict[str, object]] = {
    "bert-base-chinese": {
        "hf_name": "bert-base-chinese",
        "embedding_dim": 768,
    },
    "chinese-roberta": {
        "hf_name": "hfl/chinese-roberta-wwm-ext",
        "embedding_dim": 768,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取冻结 BERT/RoBERTa 文本向量，供多模态融合使用。"
    )
    parser.add_argument(
        "--input",
        default="data/processed/text_samples.csv",
        help="输入 CSV，至少包含 sample_id,text,label,split。",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/text_embeddings.csv",
        help="给成员 B/C 融合使用的文本向量 CSV。",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/text_bert_pred.csv",
        help="使用统一 5 列接口输出的预测 CSV。",
    )
    parser.add_argument(
        "--meta-output",
        default="outputs/predictions/text_feature_meta.json",
        help="记录模型名、向量维度和行数的 JSON 元数据。",
    )
    parser.add_argument(
        "--model",
        default="bert-base-chinese",
        choices=sorted(MODEL_PRESETS),
        help="预训练中文文本编码器。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer 截断长度。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="推理 batch size。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 行，便于快速试跑。",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="可选 split 过滤，例如 train 或 test。",
    )
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="在 train split 的文本向量上训练逻辑回归头，并写出预测文件。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="文本编码使用的 torch 设备。",
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame, path: Path) -> None:
    """校验文本向量输入表是否满足最小字段要求。

    Validate that the text embedding input table satisfies the minimum column contract.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要字段：{sorted(missing)}")


def resolve_device(requested: str) -> str:
    """解析推理设备，auto 优先 CUDA，不可用时回退 CPU。

    Resolve the inference device; auto prefers CUDA and falls back to CPU.
    """
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境不可用。")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_key: str, device: str):
    """加载 Hugging Face tokenizer 和冻结文本编码器。

    Load the Hugging Face tokenizer and frozen text encoder.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    preset = MODEL_PRESETS[model_key]
    hf_name = str(preset["hf_name"])
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name)
    model.eval()
    model.to(device)
    return tokenizer, model, int(preset["embedding_dim"])


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """批量编码文本，返回每条样本的 CLS 向量。

    Encode texts in batches and return the CLS embedding for each sample.
    """
    import torch

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="编码", unit="batch"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            # BERT/RoBERTa 都可以使用最后一层 hidden state 的 CLS 位置作为句向量。
            batch_emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        embeddings.append(batch_emb)
    if not embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(embeddings).astype(np.float32)


def make_embeddings_frame(
    sample_ids: list[str],
    labels: list[str],
    splits: list[str],
    embeddings: np.ndarray,
    embedding_dim: int,
) -> pd.DataFrame:
    """把文本向量矩阵转换成项目约定的 CSV 表。

    Convert the text embedding matrix into the project CSV contract.
    """
    if embeddings.shape[0] != len(sample_ids):
        raise ValueError("向量行数与样本数量不一致")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"期望向量维度为 {embedding_dim}，实际得到 {embeddings.shape[1]}"
        )

    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        row: dict[str, object] = {
            "sample_id": sample_id,
            "label": labels[index],
            "split": splits[index],
            "status": "ok",
            "message": f"已提取 {embedding_dim} 维文本向量",
        }
        for dim_index, value in enumerate(embeddings[index]):
            row[f"txt_emb_{dim_index:03d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def make_prediction_frame(
    sample_ids: list[str],
    labels: list[str],
    splits: list[str],
    embeddings: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """可选：在冻结文本向量上训练轻量分类头并生成预测表。

    Optionally train a lightweight classifier on frozen text embeddings and generate predictions.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    split_array = np.array(splits)
    train_mask = split_array == "train"
    if not train_mask.any():
        raise ValueError("没有找到 split=train 的样本，无法训练分类头。")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(embeddings[train_mask], y[train_mask])
    prob_matrix = classifier.predict_proba(embeddings)
    if "risk" not in label_encoder.classes_:
        raise ValueError("label 必须同时包含 risk 和 normal")
    risk_index = int(np.where(label_encoder.classes_ == "risk")[0][0])
    risk_probs = prob_matrix[:, risk_index]
    pred_indices = np.argmax(prob_matrix, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_indices)

    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": labels,
            "pred_label": pred_labels,
            "risk_prob": risk_probs.round(6),
            "model_name": model_name,
        }
    )


def run(args: argparse.Namespace) -> int:
    """执行文本向量提取主流程。

    Run the main text embedding extraction workflow.
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

    if args.split:
        df = df[df["split"].astype(str) == args.split]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit 必须为正数")
        df = df.head(args.limit)

    if df.empty:
        raise ValueError("过滤后没有可处理的样本。")

    device = resolve_device(args.device)
    tokenizer, model, embedding_dim = load_encoder(args.model, device)
    texts = df["text"].fillna("").astype(str).tolist()

    # 这里只提取冻结编码器的文本向量；默认不训练 BERT/RoBERTa 本体。
    embeddings = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    sample_ids = df["sample_id"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    splits = df["split"].astype(str).tolist()

    embeddings_output = Path(args.embeddings_output)
    pred_output = Path(args.pred_output)
    meta_output = Path(args.meta_output)
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)

    emb_frame = make_embeddings_frame(
        sample_ids=sample_ids,
        labels=labels,
        splits=splits,
        embeddings=embeddings,
        embedding_dim=embedding_dim,
    )
    emb_frame.to_csv(embeddings_output, index=False, encoding="utf-8")

    meta = {
        "model_key": args.model,
        "hf_name": MODEL_PRESETS[args.model]["hf_name"],
        "embedding_dim": embedding_dim,
        "rows": len(emb_frame),
        "max_length": args.max_length,
        "device": device,
        "classifier_trained": bool(args.train_classifier),
    }

    if args.train_classifier:
        # 可选分类头只在 train split 上训练，用于生成统一 5 列预测 CSV。
        pred_frame = make_prediction_frame(
            sample_ids=sample_ids,
            labels=labels,
            splits=splits,
            embeddings=embeddings,
            model_name=f"text_{args.model.replace('-', '_')}",
        )
        pred_frame.to_csv(pred_output, index=False, encoding="utf-8")
        print(f"[DONE] 已写出预测文件 -> {pred_output}")

    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 已写出文本向量 -> {embeddings_output}")
    print(f"[DONE] 已写出元数据 -> {meta_output}")
    print(f"[SUMMARY] 行数={len(emb_frame)} 维度={embedding_dim} 设备={device}")
    return 0


def main() -> int:
    """命令行入口。

    Command-line entry point.
    """
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
