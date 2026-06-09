"""Extract Chinese text embeddings with BERT / RoBERTa (member A, standard plan).

Reads ``data/processed/text_samples.csv`` (or ``dataset_v1.csv``), outputs:
  - outputs/predictions/text_embeddings.csv   (sample_id + txt_emb_* plus label/split/status/message)
  - outputs/predictions/text_bert_pred.csv      (optional classifier on train split)
The backbone stays frozen by default. A lightweight logistic head is trained only
on the train split when ``--train-classifier`` is set.
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
        description="Extract frozen BERT/RoBERTa text embeddings for multimodal fusion."
    )
    parser.add_argument(
        "--input",
        default="data/processed/text_samples.csv",
        help="CSV with sample_id,text,label,split.",
    )
    parser.add_argument(
        "--embeddings-output",
        default="outputs/predictions/text_embeddings.csv",
        help="Embedding CSV for member B/C fusion.",
    )
    parser.add_argument(
        "--pred-output",
        default="outputs/predictions/text_bert_pred.csv",
        help="Prediction CSV using the unified 5-column interface.",
    )
    parser.add_argument(
        "--meta-output",
        default="outputs/predictions/text_feature_meta.json",
        help="JSON with model name, embedding dimension, and row counts.",
    )
    parser.add_argument(
        "--model",
        default="bert-base-chinese",
        choices=sorted(MODEL_PRESETS),
        help="Pretrained Chinese text encoder.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (useful for dry runs).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split filter, e.g. train or test.",
    )
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Fit a logistic head on train-split embeddings and write predictions.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for encoding.",
    )
    return parser.parse_args()


def validate_dataframe(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_key: str, device: str):
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
    import torch

    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode", unit="batch"):
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
            # CLS token from last hidden state works for both BERT and RoBERTa.
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
    if embeddings.shape[0] != len(sample_ids):
        raise ValueError("embedding row count does not match sample count")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(
            f"expected embedding dim {embedding_dim}, got {embeddings.shape[1]}"
        )

    rows: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        row: dict[str, object] = {
            "sample_id": sample_id,
            "label": labels[index],
            "split": splits[index],
            "status": "ok",
            "message": f"extracted {embedding_dim}-dim text embedding",
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    split_array = np.array(splits)
    train_mask = split_array == "train"
    if not train_mask.any():
        raise ValueError("No train split rows found; cannot train classifier.")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(embeddings[train_mask], y[train_mask])
    prob_matrix = classifier.predict_proba(embeddings)
    if "risk" not in label_encoder.classes_:
        raise ValueError("labels must include 'risk' and 'normal'")
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
    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"[WAITING_FOR_A] {input_path} not found. Run src/prepare_data.py first.",
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
            raise ValueError("--limit must be positive")
        df = df.head(args.limit)

    if df.empty:
        raise ValueError("No rows to process after filtering.")

    device = resolve_device(args.device)
    tokenizer, model, embedding_dim = load_encoder(args.model, device)
    texts = df["text"].fillna("").astype(str).tolist()
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
        pred_frame = make_prediction_frame(
            sample_ids=sample_ids,
            labels=labels,
            splits=splits,
            embeddings=embeddings,
            model_name=f"text_{args.model.replace('-', '_')}",
        )
        pred_frame.to_csv(pred_output, index=False, encoding="utf-8")
        print(f"[DONE] wrote predictions -> {pred_output}")

    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote embeddings -> {embeddings_output}")
    print(f"[DONE] wrote meta -> {meta_output}")
    print(f"[SUMMARY] rows={len(emb_frame)} dim={embedding_dim} device={device}")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
