import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABEL_MAP = {
    "bukan_aduan": 0,
    "not_aduan": 0,
    "non_aduan": 0,
    "aduan_text": 1,
    "aduan": 1,
}

DEFAULT_MODEL_NAME = "indobenchmark/indobert-base-p1"


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)

    normalization_map = {
        "gak": "tidak",
        "nggak": "tidak",
        "ga": "tidak",
        "ngga": "tidak",
        "gmn": "bagaimana",
        "gimana": "bagaimana",
        "knp": "kenapa",
        "pls": "tolong",
        "plis": "tolong",
        "telyu": "telkom university",
        "tel-u": "telkom university",
    }

    for old, new in normalization_map.items():
        text = re.sub(rf"\b{old}\b", new, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label_value: str) -> int:
    key = str(label_value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Label '{label_value}' tidak dikenali")
    return LABEL_MAP[key]


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
                return df
        except Exception:
            continue

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
        return df
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"File kosong: {path}")

        header = lines[0].lstrip("\ufeff").strip()
        rows = lines[1:] if header.lower() == "full_text" else lines
        return pd.DataFrame({"full_text": rows})


def load_base_data(aduan_path: str, bukan_aduan_path: str) -> pd.DataFrame:
    a_df = read_csv_with_fallback(aduan_path)
    b_df = read_csv_with_fallback(bukan_aduan_path)

    for df_name, df in [("aduan", a_df), ("bukan_aduan", b_df)]:
        if "aduan_text" not in df.columns or "aduan_type" not in df.columns:
            raise ValueError(f"Kolom aduan_text/aduan_type tidak ditemukan di dataset {df_name}")

    base_df = pd.concat([a_df[["aduan_text", "aduan_type"]], b_df[["aduan_text", "aduan_type"]]], ignore_index=True)
    base_df = base_df.dropna(subset=["aduan_text", "aduan_type"]).copy()
    base_df["text"] = base_df["aduan_text"].astype(str).map(preprocess_text)
    base_df["label"] = base_df["aduan_type"].map(normalize_label)
    return base_df[["text", "label"]]


def load_review_data(review_path: str, text_col: str = "full_text", label_col: str = "reviewed_label") -> pd.DataFrame:
    if not os.path.exists(review_path):
        return pd.DataFrame(columns=["text", "label"])

    df = read_csv_with_fallback(review_path)
    if text_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame(columns=["text", "label"])

    review_df = df[[text_col, label_col]].copy()
    review_df[label_col] = review_df[label_col].astype(str).str.strip().str.lower()
    review_df = review_df[review_df[label_col] != ""]

    if review_df.empty:
        return pd.DataFrame(columns=["text", "label"])

    review_df["text"] = review_df[text_col].astype(str).map(preprocess_text)
    review_df["label"] = review_df[label_col].map(normalize_label)
    review_df = review_df.dropna(subset=["label"])
    return review_df[["text", "label"]]


def build_training_dataframe(aduan_path: str, bukan_aduan_path: str, review_path: str) -> pd.DataFrame:
    base_df = load_base_data(aduan_path, bukan_aduan_path)
    review_df = load_review_data(review_path)
    merged = pd.concat([base_df, review_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return merged


@dataclass
class EncodedData:
    train: Dataset
    val: Dataset


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def tokenize_split(
    df: pd.DataFrame,
    tokenizer,
    test_size: float,
    random_state: int,
    max_length: int,
) -> EncodedData:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    return EncodedData(train=train_ds, val=val_ds)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "precision_aduan": precision,
        "recall_aduan": recall,
        "f1_aduan": f1,
    }


def train_deep_model(args) -> None:
    df = build_training_dataframe(args.aduan_path, args.bukan_aduan_path, args.review_path)
    if df.empty:
        raise ValueError("Training dataframe kosong")

    print("=== TRAIN DATA SUMMARY ===")
    print(df["label"].value_counts().rename(index={0: "bukan_aduan", 1: "aduan_text"}))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    encoded = tokenize_split(
        df=df,
        tokenizer=tokenizer,
        test_size=args.val_size,
        random_state=args.random_state,
        max_length=args.max_length,
    )

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=df["label"].values,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_aduan",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=encoded.train,
        eval_dataset=encoded.val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("=== VALIDATION METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "model_name": args.model_name,
        "label_map": LABEL_MAP,
        "id2label": {"0": "bukan_aduan", "1": "aduan_text"},
        "recommended_threshold": args.threshold,
    }
    with open(os.path.join(args.output_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"Model tersimpan di: {args.output_dir}")


def load_inference_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_dataframe(args) -> None:
    df = read_csv_with_fallback(args.raw_path)
    if args.raw_text_col not in df.columns:
        raise ValueError(f"Kolom '{args.raw_text_col}' tidak ditemukan di raw file")

    tokenizer, model, device = load_inference_model(args.model_dir)

    texts = df[args.raw_text_col].astype(str).tolist()
    cleaned = [preprocess_text(t) for t in texts]

    probs_aduan: List[float] = []
    batch_size = args.batch_size

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            encoded = tokenizer(batch, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probs_aduan.extend(probs[:, 1].tolist())

    out = df.copy()
    out["prob_aduan_text"] = probs_aduan
    out["prob_bukan_aduan"] = [1.0 - p for p in probs_aduan]
    out["pred_label"] = ["aduan_text" if p >= args.threshold else "bukan_aduan" for p in probs_aduan]
    out["pred_confidence"] = [max(p, 1.0 - p) for p in probs_aduan]

    output_cols = [
        args.raw_text_col,
        "pred_label",
        "pred_confidence",
        "prob_bukan_aduan",
        "prob_aduan_text",
    ]
    rest = [c for c in out.columns if c not in output_cols]
    out = out[output_cols + rest]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print("=== INFERENCE SUMMARY ===")
    print(f"Input rows : {len(out)}")
    print(out["pred_label"].value_counts())
    print(f"Output CSV : {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deep learning classifier aduan vs bukan aduan")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_p = subparsers.add_parser("train", help="Fine-tuning model deep learning")
    train_p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_p.add_argument("--aduan-path", default="kode_aduan_text_klasifikasi/contoh_teks_aduan.csv")
    train_p.add_argument("--bukan-aduan-path", default="kode_aduan_text_klasifikasi/contoh_teks_bukan_aduan.csv")
    train_p.add_argument("--review-path", default="kode_aduan_text_klasifikasi/penting/hard_cases_review.csv")
    train_p.add_argument("--output-dir", default="kode_aduan_text_klasifikasi/ml_models/indobert_aduan")
    train_p.add_argument("--epochs", type=int, default=4)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--learning-rate", type=float, default=2e-5)
    train_p.add_argument("--val-size", type=float, default=0.2)
    train_p.add_argument("--random-state", type=int, default=42)
    train_p.add_argument("--max-length", type=int, default=128)
    train_p.add_argument("--threshold", type=float, default=0.65)

    pred_p = subparsers.add_parser("predict", help="Inferensi deep learning ke CSV")
    pred_p.add_argument("--model-dir", default="kode_aduan_text_klasifikasi/ml_models/indobert_aduan")
    pred_p.add_argument("--raw-path", default="kode_aduan_text_klasifikasi/penting/all_dataset.csv")
    pred_p.add_argument("--raw-text-col", default="full_text")
    pred_p.add_argument("--output-path", default="kode_aduan_text_klasifikasi/penting/all_dataset_labeled_indobert.csv")
    pred_p.add_argument("--batch-size", type=int, default=32)
    pred_p.add_argument("--max-length", type=int, default=128)
    pred_p.add_argument("--threshold", type=float, default=0.65)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_deep_model(args)
    else:
        predict_dataframe(args)


if __name__ == "__main__":
    main()
