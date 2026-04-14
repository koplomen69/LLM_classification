import argparse
import re
from typing import Optional

import pandas as pd


INSTITUTIONAL_CONTEXT_PATTERNS = [
    r"\bbpp\b", r"\bukt\b", r"\bkrs\b", r"\bksm\b", r"\blms\b", r"\bigracias\b",
    r"\bsirama\b", r"\bputi\b", r"\bdosen\b", r"\bdosbing\b", r"\bdosbim\b",
    r"\bsidang\b", r"\bskripsi\b", r"\beprt\b", r"\bmbkm\b", r"\bmsib\b",
    r"\bparkir\b", r"\bktm\b", r"\bnilai\b", r"\bpembayaran\b", r"\byudisium\b",
]


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_institutional_context(text: str) -> bool:
    cleaned = preprocess_text(text)
    return any(re.search(pattern, cleaned) for pattern in INSTITUTIONAL_CONTEXT_PATTERNS)


def build_hybrid_label(
    row: pd.Series,
    svm_keep_threshold: float,
    dl_strong_bukan_threshold: float,
    dl_medium_bukan_threshold: float,
) -> tuple[str, str]:
    svm_label = row["svm_label"]
    dl_label = row["dl_label"]

    if svm_label == "bukan_aduan":
        return "bukan_aduan", "svm_bukan_base"

    # If both agree aduan, keep aduan.
    if dl_label == "aduan_text":
        return "aduan_text", "agree_aduan"

    # Disagreement case: SVM aduan vs DL bukan.
    svm_conf = float(row.get("svm_confidence", 0.0))
    dl_prob_bukan = float(row.get("dl_prob_bukan", 0.0))
    has_context = has_institutional_context(str(row.get("full_text", "")))

    if (not has_context) and dl_prob_bukan >= dl_strong_bukan_threshold:
        return "bukan_aduan", "dl_veto_non_context"

    if svm_conf < svm_keep_threshold and dl_prob_bukan >= dl_medium_bukan_threshold:
        return "bukan_aduan", "dl_veto_low_svm_conf"

    return "aduan_text", "svm_keep_disagreement"


def evaluate_against_review(df: pd.DataFrame, review_path: str) -> Optional[float]:
    review_df = pd.read_csv(review_path, sep=";", engine="python")
    review_df.columns = [str(c).lstrip("\ufeff").strip() for c in review_df.columns]

    if "full_text" not in review_df.columns or "reviewed_label" not in review_df.columns:
        return None

    review_df["reviewed_label"] = review_df["reviewed_label"].astype(str).str.strip().str.lower()
    review_df = review_df[review_df["reviewed_label"].isin(["aduan_text", "bukan_aduan"])][["full_text", "reviewed_label"]]
    review_df = review_df.drop_duplicates()

    merged = review_df.merge(df[["full_text", "pred_label"]], on="full_text", how="inner")
    if merged.empty:
        return None

    return float((merged["pred_label"] == merged["reviewed_label"]).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid classifier: SVM as base, IndoBERT as selective veto")

    parser.add_argument(
        "--svm-path",
        default="kode_aduan_text_klasifikasi/penting/all_dataset_labeled_svm_active_strict_v2.csv",
        help="Path output CSV dari model SVM",
    )
    parser.add_argument(
        "--dl-path",
        default="kode_aduan_text_klasifikasi/penting/all_dataset_labeled_indobert.csv",
        help="Path output CSV dari model deep learning",
    )
    parser.add_argument(
        "--output-path",
        default="kode_aduan_text_klasifikasi/penting/all_dataset_labeled_hybrid.csv",
        help="Path output CSV hybrid",
    )
    parser.add_argument(
        "--svm-keep-threshold",
        type=float,
        default=0.85,
        help="Jika confidence SVM aduan di bawah ini, DL boleh veto",
    )
    parser.add_argument(
        "--dl-strong-bukan-threshold",
        type=float,
        default=0.85,
        help="Threshold kuat probabilitas bukan_aduan dari DL untuk veto di non-context",
    )
    parser.add_argument(
        "--dl-medium-bukan-threshold",
        type=float,
        default=0.70,
        help="Threshold menengah probabilitas bukan_aduan dari DL untuk veto kasus low-confidence SVM",
    )
    parser.add_argument(
        "--review-path",
        default="",
        help="Opsional: path file review manual untuk evaluasi cepat",
    )
    parser.add_argument(
        "--disagreement-out",
        default="kode_aduan_text_klasifikasi/penting/disagreement_svm_vs_dl.csv",
        help="Path output untuk daftar disagreement SVM vs DL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    svm_df = pd.read_csv(args.svm_path)
    dl_df = pd.read_csv(args.dl_path)

    s = svm_df[["full_text", "pred_label", "pred_confidence"]].rename(
        columns={"pred_label": "svm_label", "pred_confidence": "svm_confidence"}
    )

    dl_prob_col = "prob_bukan_aduan" if "prob_bukan_aduan" in dl_df.columns else None
    if dl_prob_col is None:
        raise ValueError("Kolom prob_bukan_aduan tidak ditemukan pada file deep learning")

    d = dl_df[["full_text", "pred_label", dl_prob_col]].rename(
        columns={"pred_label": "dl_label", dl_prob_col: "dl_prob_bukan"}
    )

    merged = s.merge(d, on="full_text", how="inner")

    labels = []
    reasons = []
    for _, row in merged.iterrows():
        label, reason = build_hybrid_label(
            row,
            svm_keep_threshold=args.svm_keep_threshold,
            dl_strong_bukan_threshold=args.dl_strong_bukan_threshold,
            dl_medium_bukan_threshold=args.dl_medium_bukan_threshold,
        )
        labels.append(label)
        reasons.append(reason)

    merged["pred_label"] = labels
    merged["hybrid_reason"] = reasons

    disagreement_df = merged[
        (merged["svm_label"] == "aduan_text") & (merged["dl_label"] == "bukan_aduan")
    ].copy()
    disagreement_df = disagreement_df.sort_values(
        by=["svm_confidence", "dl_prob_bukan"], ascending=[True, False]
    )

    out_cols = [
        "full_text",
        "pred_label",
        "hybrid_reason",
        "svm_label",
        "svm_confidence",
        "dl_label",
        "dl_prob_bukan",
    ]

    merged[out_cols].to_csv(args.output_path, index=False)
    disagreement_df[out_cols].to_csv(args.disagreement_out, index=False)

    print("=== HYBRID SUMMARY ===")
    print(f"Rows merged : {len(merged)}")
    print(merged["pred_label"].value_counts())
    print("Top reasons:")
    print(merged["hybrid_reason"].value_counts().head(10))
    print(f"Output CSV : {args.output_path}")
    print(f"Disagreement CSV : {args.disagreement_out}")

    if args.review_path:
        acc = evaluate_against_review(merged, args.review_path)
        if acc is not None:
            print(f"Review accuracy (quick check): {acc:.4f}")


if __name__ == "__main__":
    main()
