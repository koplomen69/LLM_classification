import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_lookup(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    exact = {}
    norm = {}

    for _, row in df.iterrows():
        text = str(row["full_text"])
        label = str(row["pred_label"]).strip().lower()

        if text not in exact:
            exact[text] = label

        n = normalize_text(text)
        if n and n not in norm:
            norm[n] = label

    return exact, norm


def evaluate_model(manual_df: pd.DataFrame, pred_path: Path) -> Tuple[pd.DataFrame, dict]:
    pred_df = pd.read_csv(pred_path)
    pred_df["pred_label"] = pred_df["pred_label"].astype(str).str.strip().str.lower()

    if "full_text" not in pred_df.columns or "pred_label" not in pred_df.columns:
        raise ValueError(f"Kolom wajib tidak ditemukan di {pred_path}")

    exact_map, norm_map = build_lookup(pred_df[["full_text", "pred_label"]].copy())

    rows = []
    for _, r in manual_df.iterrows():
        text = str(r["text"])
        label = None
        matched_by = "none"

        if text in exact_map:
            label = exact_map[text]
            matched_by = "exact"
        else:
            n = normalize_text(text)
            if n in norm_map:
                label = norm_map[n]
                matched_by = "normalized"

        if label is None:
            rows.append({
                "text": text,
                "manual_label": "aduan_text",
                "pred_label": "not_found",
                "matched_by": matched_by,
                "is_missed_aduan": True,
                "reason": "not_found_in_prediction_file",
            })
        else:
            is_missed = label != "aduan_text"
            rows.append({
                "text": text,
                "manual_label": "aduan_text",
                "pred_label": label,
                "matched_by": matched_by,
                "is_missed_aduan": is_missed,
                "reason": "predicted_bukan_aduan" if is_missed else "ok",
            })

    result_df = pd.DataFrame(rows)

    total = len(result_df)
    found = int((result_df["pred_label"] != "not_found").sum())
    exact = int((result_df["matched_by"] == "exact").sum())
    normalized = int((result_df["matched_by"] == "normalized").sum())
    missed = int(result_df["is_missed_aduan"].sum())
    found_and_correct = int(((result_df["pred_label"] == "aduan_text")).sum())

    summary = {
        "model_file": str(pred_path),
        "manual_total": total,
        "coverage_found": found,
        "coverage_found_pct": (found / total) if total else 0.0,
        "matched_exact": exact,
        "matched_normalized": normalized,
        "missed_aduan": missed,
        "missed_aduan_pct_of_manual": (missed / total) if total else 0.0,
        "recall_on_found": (found_and_correct / found) if found else 0.0,
        "recall_on_manual_total": (found_and_correct / total) if total else 0.0,
    }

    return result_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bandingkan hasil model dengan dataset manual yang diasumsikan semua aduan_text")
    parser.add_argument("--manual", required=True, help="Path CSV manual, wajib punya kolom 'text'")
    parser.add_argument("--svm", required=True, help="Path CSV prediksi SVM")
    parser.add_argument("--dl", required=True, help="Path CSV prediksi Deep Learning")
    parser.add_argument("--hybrid", required=False, default="", help="Path CSV prediksi hybrid (opsional)")
    parser.add_argument("--outdir", required=True, help="Folder output hasil evaluasi")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manual_df = pd.read_csv(args.manual)
    if "text" not in manual_df.columns:
        raise ValueError("File manual harus memiliki kolom 'text'")

    manual_df = manual_df[["text"]].copy()
    manual_df["text"] = manual_df["text"].astype(str)
    manual_df = manual_df.drop_duplicates().reset_index(drop=True)

    model_inputs = {
        "svm": Path(args.svm),
        "dl": Path(args.dl),
    }
    if args.hybrid:
        model_inputs["hybrid"] = Path(args.hybrid)

    summaries = []
    missed_sets = {}

    for name, path in model_inputs.items():
        detail_df, summary = evaluate_model(manual_df, path)
        detail_path = outdir / f"detail_{name}_vs_manual.csv"
        detail_df.to_csv(detail_path, index=False)

        missed_df = detail_df[detail_df["is_missed_aduan"]].copy()
        missed_path = outdir / f"missed_{name}.csv"
        missed_df.to_csv(missed_path, index=False)

        summaries.append({"model": name, **summary, "detail_file": str(detail_path), "missed_file": str(missed_path)})
        missed_sets[name] = set(missed_df["text"].tolist())

    summary_df = pd.DataFrame(summaries)
    summary_path = outdir / "summary_manual_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    if "svm" in missed_sets and "dl" in missed_sets:
        both_missed = sorted(missed_sets["svm"].intersection(missed_sets["dl"]))
        only_svm_missed = sorted(missed_sets["svm"].difference(missed_sets["dl"]))
        only_dl_missed = sorted(missed_sets["dl"].difference(missed_sets["svm"]))

        pd.DataFrame({"text": both_missed}).to_csv(outdir / "missed_both_svm_and_dl.csv", index=False)
        pd.DataFrame({"text": only_svm_missed}).to_csv(outdir / "missed_only_svm.csv", index=False)
        pd.DataFrame({"text": only_dl_missed}).to_csv(outdir / "missed_only_dl.csv", index=False)

    print("=== SUMMARY ===")
    print(summary_df[[
        "model",
        "manual_total",
        "coverage_found",
        "coverage_found_pct",
        "missed_aduan",
        "missed_aduan_pct_of_manual",
        "recall_on_found",
        "recall_on_manual_total",
    ]].to_string(index=False))
    print(f"\nSummary file: {summary_path}")


if __name__ == "__main__":
    main()
