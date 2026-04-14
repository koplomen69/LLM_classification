import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from svm_aduan_classifier import build_model, load_dataset, normalize_label, preprocess_text


def _read_csv_auto(path: str) -> pd.DataFrame:
    # Prefer explicit delimiters first because reviewed files often use ';'.
    for sep in [";", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                break
        except Exception:
            df = None
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    return df


def load_reviewed_data(
    reviewed_path: str,
    text_col: str = "full_text",
    reviewed_label_col: str = "reviewed_label",
) -> pd.DataFrame:
    if not os.path.exists(reviewed_path):
        raise FileNotFoundError(f"File reviewed tidak ditemukan: {reviewed_path}")

    df = _read_csv_auto(reviewed_path)
    if text_col not in df.columns:
        raise ValueError(f"Kolom text '{text_col}' tidak ditemukan di {reviewed_path}")
    if reviewed_label_col not in df.columns:
        raise ValueError(f"Kolom reviewed label '{reviewed_label_col}' tidak ditemukan di {reviewed_path}")

    reviewed_df = df[[text_col, reviewed_label_col]].copy()
    reviewed_df[reviewed_label_col] = reviewed_df[reviewed_label_col].astype(str).str.strip().str.lower()
    reviewed_df = reviewed_df[reviewed_df[reviewed_label_col] != ""]

    if reviewed_df.empty:
        raise ValueError("Belum ada label manual di file reviewed. Isi kolom reviewed_label terlebih dahulu.")

    reviewed_df["text"] = reviewed_df[text_col].astype(str).map(preprocess_text)
    reviewed_df["label"] = reviewed_df[reviewed_label_col].map(normalize_label)
    return reviewed_df[["text", "label"]]


def build_augmented_training_data(
    base_aduan_path: str,
    base_bukan_path: str,
    reviewed_path: str,
    text_col_reviewed: str,
    reviewed_label_col: str,
) -> pd.DataFrame:
    base_aduan = load_dataset(base_aduan_path)
    base_bukan = load_dataset(base_bukan_path)

    base_df = pd.concat([base_aduan, base_bukan], ignore_index=True)
    base_df = base_df[["aduan_text", "label"]].rename(columns={"aduan_text": "text"})

    reviewed_df = load_reviewed_data(
        reviewed_path=reviewed_path,
        text_col=text_col_reviewed,
        reviewed_label_col=reviewed_label_col,
    )

    merged_df = pd.concat([base_df, reviewed_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return merged_df


def train_active_learning_model(
    merged_df: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    X = merged_df["text"]
    y = merged_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=== ACTIVE LEARNING EVALUATION ===")
    print(classification_report(y_test, y_pred, target_names=["bukan_aduan", "aduan_text"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain SVM dengan data review manual (active learning)."
    )

    parser.add_argument(
        "--base-aduan-path",
        default="kode_aduan_text_klasifikasi/contoh_teks_aduan.csv",
        help="Path dataset aduan dasar",
    )
    parser.add_argument(
        "--base-bukan-path",
        default="kode_aduan_text_klasifikasi/contoh_teks_bukan_aduan.csv",
        help="Path dataset bukan aduan dasar",
    )
    parser.add_argument(
        "--reviewed-path",
        default="kode_aduan_text_klasifikasi/penting/hard_cases_review.csv",
        help="Path CSV hasil review manual hard cases",
    )
    parser.add_argument(
        "--review-text-col",
        default="full_text",
        help="Kolom text pada file reviewed",
    )
    parser.add_argument(
        "--review-label-col",
        default="reviewed_label",
        help="Kolom label manual pada file reviewed",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi test split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--model-out",
        default="kode_aduan_text_klasifikasi/ml_models/svm_aduan_pipeline_active.joblib",
        help="Path output model active learning",
    )
    parser.add_argument(
        "--augmented-out",
        default="kode_aduan_text_klasifikasi/penting/augmented_training_data.csv",
        help="Path output dataset gabungan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged_df = build_augmented_training_data(
        base_aduan_path=args.base_aduan_path,
        base_bukan_path=args.base_bukan_path,
        reviewed_path=args.reviewed_path,
        text_col_reviewed=args.review_text_col,
        reviewed_label_col=args.review_label_col,
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.augmented_out), exist_ok=True)

    merged_df.to_csv(args.augmented_out, index=False)

    print("=== AUGMENTED DATA SUMMARY ===")
    print(f"Total training rows: {len(merged_df)}")
    print(merged_df["label"].value_counts().rename(index={0: "bukan_aduan", 1: "aduan_text"}))

    model = train_active_learning_model(
        merged_df=merged_df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    joblib.dump(model, args.model_out)
    print(f"Model active learning tersimpan di: {args.model_out}")
    print(f"Augmented training data tersimpan di: {args.augmented_out}")


if __name__ == "__main__":
    main()
