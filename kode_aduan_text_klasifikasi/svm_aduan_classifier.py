import argparse
import os
import re
from typing import Optional, Set, Tuple

import joblib
import pandas as pd
from pandas.errors import ParserError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC


LABEL_MAP = {
    "aduan_text": 1,
    "aduan": 1,
    "bukan_aduan": 0,
    "not_aduan": 0,
    "non_aduan": 0,
}


DEFAULT_NON_ADUAN_KEYWORDS = {
    "rekomendasi", "rekomen", "spill", "kuliner", "makan", "jajan", "cafe", "coffee",
    "warkop", "warung", "nobar", "pacar", "jomblo", "pdkt", "cowo", "cewe", "gebetan",
    "kos", "kost", "kosan", "rental", "ojol", "gofood", "gofud", "film", "karaoke",
    "main", "nongkrong", "mutualan", "gabut", "mabar", "spotify", "wisata", "mall",
}

ADUAN_INTENT_PATTERNS = [
    r"\bbpp\b", r"\bbayar\b", r"\bpembayaran\b", r"\bkrs\b", r"\bksm\b",
    r"\bsidang\b", r"\bdosen\b", r"\bdosbing\b", r"\bdosbim\b", r"\bputi\b",
    r"\blms\b", r"\bigracias\b", r"\bsirama\b", r"\bparkir\b", r"\bktm\b",
    r"\bmbkm\b", r"\bmsib\b", r"\beprt\b", r"\btak\b", r"\bnilai\b", r"\bskripsi\b",
]

NON_ADUAN_INTENT_PATTERNS = [
    r"\brekomendasi\b", r"\bspill\b", r"\btempat makan\b", r"\bmakan apa\b",
    r"\bcari pacar\b", r"\bmutualan\b", r"\bmabar\b", r"\bnobar\b", r"\bkuliner\b",
    r"\bcafe\b", r"\bkopi\b", r"\bkosan\b", r"\brental\b", r"\bdeliv\b", r"\bojol\b",
]

INSTITUTIONAL_CONTEXT_PATTERNS = [
    r"\bbpp\b", r"\bukt\b", r"\bkrs\b", r"\bksm\b", r"\blms\b", r"\bigracias\b",
    r"\bsirama\b", r"\bputi\b", r"\bdosen\b", r"\bdosbing\b", r"\bdosbim\b",
    r"\bsidang\b", r"\bskripsi\b", r"\beprt\b", r"\bmbkm\b", r"\bmsib\b",
    r"\bparkir\b", r"\bktm\b", r"\bnilai\b", r"\bpembayaran\b", r"\byudisium\b",
]

COMPLAINT_SIGNAL_PATTERNS = [
    r"\berror\b", r"\bgagal\b", r"\bmasalah\b", r"\bkendala\b", r"\btidak bisa\b",
    r"\bgabisa\b", r"\bkenapa\b", r"\bkapan\b", r"\bbelum\b", r"\bsusah\b",
    r"\btolong\b", r"\bkeluhan\b", r"\bditolak\b", r"\bnggak bisa\b", r"\bga bisa\b",
]


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
        "igracias": "igracias",
    }

    for old, new in normalization_map.items():
        text = re.sub(rf"\b{old}\b", new, text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label_value: str) -> int:
    key = str(label_value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(
            f"Label '{label_value}' tidak dikenali. Gunakan salah satu: {sorted(LABEL_MAP.keys())}"
        )
    return LABEL_MAP[key]


def load_dataset(csv_path: str, text_col: str = "aduan_text", label_col: str = "aduan_type") -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";")

    if text_col not in df.columns:
        raise ValueError(f"Kolom teks '{text_col}' tidak ditemukan di {csv_path}")
    if label_col not in df.columns:
        raise ValueError(f"Kolom label '{label_col}' tidak ditemukan di {csv_path}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).map(preprocess_text)
    df["label"] = df[label_col].map(normalize_label)

    return df


def load_raw_dataset(csv_path: str, text_col: str = "full_text") -> pd.DataFrame:
    # sep=None with python engine lets pandas auto-detect delimiter (e.g. ; or ,)
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [c.lstrip("\ufeff") if isinstance(c, str) else c for c in df.columns]
    except ParserError:
        # Fallback for messy one-column raw files with inconsistent separators.
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"File raw kosong: {csv_path}")

        # Drop header row if it matches expected text column.
        first_line = lines[0].strip().lower().lstrip("\ufeff")
        if first_line == text_col.strip().lower():
            lines = lines[1:]

        df = pd.DataFrame({text_col: lines})

    if text_col not in df.columns:
        raise ValueError(f"Kolom teks raw '{text_col}' tidak ditemukan di {csv_path}")

    raw_df = df.copy()
    raw_df[text_col] = raw_df[text_col].astype(str).fillna("")
    raw_df["_cleaned_text"] = raw_df[text_col].map(preprocess_text)
    return raw_df


def build_model() -> Pipeline:
    return Pipeline(
        memory=None,
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=12000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "svm",
                SVC(
                    kernel="linear",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame]:
    X = df["aduan_text"]
    y = df["label"]

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

    print("=== EVALUATION ===")
    print(classification_report(y_test, y_pred, target_names=["bukan_aduan", "aduan_text"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    eval_df = pd.DataFrame({
        "text": X_test,
        "y_true": y_test,
        "y_pred": y_pred,
    })
    return model, eval_df


def predict_text(model: Pipeline, text: str) -> Tuple[str, float]:
    cleaned_text = preprocess_text(text)
    proba = model.predict_proba([cleaned_text])[0]
    pred = int(proba.argmax())
    label = "aduan_text" if pred == 1 else "bukan_aduan"
    confidence = float(proba[pred])
    return label, confidence


def load_keyword_set(keyword_csv_path: str) -> Set[str]:
    if not os.path.exists(keyword_csv_path):
        return set()

    df = pd.read_csv(keyword_csv_path)
    if df.empty:
        return set()

    col = "Keyword" if "Keyword" in df.columns else df.columns[0]

    keywords = set()
    noisy_keywords = {
        "info", "saran", "tips", "minggu", "kelas", "kerja", "teman", "temen",
        "murah", "bagus", "keren", "tempat", "makan",
    }
    for raw_kw in df[col].dropna().astype(str).tolist():
        kw = preprocess_text(raw_kw).strip()
        if kw and kw not in noisy_keywords:
            keywords.add(kw)
    return keywords


def _keyword_hits_score(cleaned: str, tokens: Set[str], keywords: Set[str], single_weight: float, phrase_weight: float) -> float:
    score = 0.0
    for kw in keywords:
        if " " in kw:
            if kw in cleaned:
                score += phrase_weight
        elif kw in tokens:
            score += single_weight
    return score


def _pattern_hits_score(cleaned: str, patterns: list[str], weight: float) -> float:
    return sum(weight for pattern in patterns if re.search(pattern, cleaned))


def keyword_rule_score(text: str, aduan_keywords: Set[str], non_aduan_keywords: Set[str]) -> float:
    cleaned = preprocess_text(text)
    if not cleaned:
        return 0.0

    tokens = set(cleaned.split())
    score = 0.0
    score += _keyword_hits_score(cleaned, tokens, aduan_keywords, single_weight=1.0, phrase_weight=1.5)
    score -= _keyword_hits_score(cleaned, tokens, non_aduan_keywords, single_weight=1.0, phrase_weight=1.4)
    score += _pattern_hits_score(cleaned, ADUAN_INTENT_PATTERNS, weight=1.2)
    score -= _pattern_hits_score(cleaned, NON_ADUAN_INTENT_PATTERNS, weight=1.1)

    return score


def has_pattern_hit(cleaned_text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, cleaned_text) for pattern in patterns)


def count_pattern_hits(cleaned_text: str, patterns: list[str]) -> int:
    return sum(1 for pattern in patterns if re.search(pattern, cleaned_text))


def _strict_mode_gate(
    prob_aduan: float,
    adjusted_prob: float,
    rule_score: float,
    cleaned_text: str,
    threshold: float,
    high_conf_override: float,
) -> Optional[str]:
    context_hits = count_pattern_hits(cleaned_text, INSTITUTIONAL_CONTEXT_PATTERNS)
    complaint_hits = count_pattern_hits(cleaned_text, COMPLAINT_SIGNAL_PATTERNS)

    if prob_aduan < 0.55:
        return "bukan_aduan"

    if context_hits == 0 and not (prob_aduan >= 0.97 and rule_score >= 2.0):
        return "bukan_aduan"

    if context_hits > 0 and complaint_hits == 0 and not (prob_aduan >= high_conf_override and rule_score >= 1.5):
        return "bukan_aduan"

    if (context_hits < 1 or complaint_hits < 1) and adjusted_prob < max(threshold + 0.2, 0.78):
        return "bukan_aduan"

    if context_hits > 0 and complaint_hits > 0 and adjusted_prob < (threshold + 0.08):
        return "bukan_aduan"

    return None


def decide_label_with_rules(
    prob_aduan: float,
    rule_score: float,
    cleaned_text: str,
    threshold: float = 0.6,
    strict_aduan_mode: bool = True,
    high_conf_override: float = 0.9,
) -> Tuple[str, float]:
    adjusted_prob = max(0.0, min(1.0, prob_aduan + (0.05 * rule_score)))

    if strict_aduan_mode:
        gated_label = _strict_mode_gate(
            prob_aduan=prob_aduan,
            adjusted_prob=adjusted_prob,
            rule_score=rule_score,
            cleaned_text=cleaned_text,
            threshold=threshold,
            high_conf_override=high_conf_override,
        )
        if gated_label == "bukan_aduan":
            return "bukan_aduan", 1.0 - adjusted_prob

    if rule_score >= 2.0 and prob_aduan >= 0.35:
        return "aduan_text", adjusted_prob
    if rule_score <= -2.0 and prob_aduan <= 0.70:
        return "bukan_aduan", 1.0 - adjusted_prob

    if adjusted_prob >= threshold:
        return "aduan_text", adjusted_prob
    return "bukan_aduan", 1.0 - adjusted_prob


def export_hard_cases(
    labeled_df: pd.DataFrame,
    output_path: str,
    min_confidence: float = 0.45,
    max_confidence: float = 0.65,
) -> pd.DataFrame:
    hard_df = labeled_df[
        (labeled_df["pred_confidence"] >= min_confidence)
        & (labeled_df["pred_confidence"] <= max_confidence)
    ].copy()

    hard_df["reviewed_label"] = ""
    hard_df["review_notes"] = ""
    hard_df.to_csv(output_path, index=False)
    return hard_df


def predict_raw_dataframe(
    model: Pipeline,
    raw_df: pd.DataFrame,
    text_col: str,
    aduan_keywords: Optional[Set[str]] = None,
    non_aduan_keywords: Optional[Set[str]] = None,
    decision_threshold: float = 0.6,
    strict_aduan_mode: bool = True,
    high_conf_override: float = 0.9,
) -> pd.DataFrame:
    probas = model.predict_proba(raw_df["_cleaned_text"])
    aduan_keywords = aduan_keywords or set()
    non_aduan_keywords = non_aduan_keywords or DEFAULT_NON_ADUAN_KEYWORDS

    output_df = raw_df.copy()
    pred_labels = []
    pred_confidences = []
    rule_scores = []

    for i, row in output_df.iterrows():
        prob_aduan = float(probas[i][1])
        score = keyword_rule_score(row[text_col], aduan_keywords, non_aduan_keywords)
        label, confidence = decide_label_with_rules(
            prob_aduan,
            score,
            cleaned_text=row["_cleaned_text"],
            threshold=decision_threshold,
            strict_aduan_mode=strict_aduan_mode,
            high_conf_override=high_conf_override,
        )

        pred_labels.append(label)
        pred_confidences.append(confidence)
        rule_scores.append(score)

    output_df["pred_label"] = pred_labels
    output_df["pred_confidence"] = pred_confidences
    output_df["prob_bukan_aduan"] = probas[:, 0]
    output_df["prob_aduan_text"] = probas[:, 1]
    output_df["keyword_rule_score"] = rule_scores
    output_df.drop(columns=["_cleaned_text"], inplace=True)

    # Keep original text column near prediction columns for easier manual review.
    preferred_order = [
        text_col,
        "pred_label",
        "pred_confidence",
        "prob_bukan_aduan",
        "prob_aduan_text",
        "keyword_rule_score",
    ]
    ordered_cols = [c for c in preferred_order if c in output_df.columns]
    remaining_cols = [c for c in output_df.columns if c not in ordered_cols]
    return output_df[ordered_cols + remaining_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Klasifikasi teks aduan vs bukan aduan menggunakan SVM. "
            "Definisi aduan mengikuti label dataset yang Anda berikan."
        )
    )

    parser.add_argument(
        "--aduan-path",
        default="contoh_teks_aduan.csv",
        help="Path CSV data aduan",
    )
    parser.add_argument(
        "--bukan-aduan-path",
        default="contoh_teks_bukan_aduan.csv",
        help="Path CSV data bukan aduan",
    )
    parser.add_argument(
        "--model-out",
        default="ml_models/svm_aduan_pipeline.joblib",
        help="Path output model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test, default 0.2",
    )
    parser.add_argument(
        "--predict",
        default=None,
        help="Jika diisi, lakukan prediksi untuk satu teks setelah training",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Jalankan inferensi saja menggunakan model existing (tanpa training)",
    )
    parser.add_argument(
        "--model-path",
        default="ml_models/svm_aduan_pipeline.joblib",
        help="Path model existing untuk mode inference-only",
    )
    parser.add_argument(
        "--raw-path",
        default="kode_aduan_text_klasifikasi/penting/all_dataset.csv",
        help="Path CSV raw text untuk mode inference-only",
    )
    parser.add_argument(
        "--raw-text-col",
        default="full_text",
        help="Nama kolom teks pada file raw",
    )
    parser.add_argument(
        "--output-path",
        default="kode_aduan_text_klasifikasi/penting/all_dataset_labeled_svm.csv",
        help="Path output CSV hasil klasifikasi raw text",
    )
    parser.add_argument(
        "--keyword-path",
        default="kode_aduan_text_klasifikasi/list_kyword.csv",
        help="Path CSV keyword aduan untuk rule-based correction",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.6,
        help="Threshold keputusan akhir setelah koreksi keyword (0-1)",
    )
    parser.add_argument(
        "--strict-aduan-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mode ketat: butuh konteks institusional kuat untuk melabeli aduan_text",
    )
    parser.add_argument(
        "--high-conf-override",
        type=float,
        default=0.9,
        help="Jika mode ketat aktif, aduan non-konteks hanya lolos jika prob_aduan di atas nilai ini",
    )
    parser.add_argument(
        "--hard-cases-out",
        default="",
        help="Path output CSV hard cases untuk review manual (opsional)",
    )
    parser.add_argument(
        "--hard-min-confidence",
        type=float,
        default=0.45,
        help="Batas bawah confidence untuk hard cases",
    )
    parser.add_argument(
        "--hard-max-confidence",
        type=float,
        default=0.65,
        help="Batas atas confidence untuk hard cases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.inference_only:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {args.model_path}")

        model = joblib.load(args.model_path)
        raw_df = load_raw_dataset(args.raw_path, text_col=args.raw_text_col)

        aduan_keywords = load_keyword_set(args.keyword_path)
        non_aduan_keywords = DEFAULT_NON_ADUAN_KEYWORDS

        labeled_df = predict_raw_dataframe(
            model,
            raw_df,
            text_col=args.raw_text_col,
            aduan_keywords=aduan_keywords,
            non_aduan_keywords=non_aduan_keywords,
            decision_threshold=args.decision_threshold,
            strict_aduan_mode=args.strict_aduan_mode,
            high_conf_override=args.high_conf_override,
        )

        labeled_df.to_csv(args.output_path, index=False)

        hard_count = 0
        if args.hard_cases_out:
            hard_df = export_hard_cases(
                labeled_df,
                output_path=args.hard_cases_out,
                min_confidence=args.hard_min_confidence,
                max_confidence=args.hard_max_confidence,
            )
            hard_count = len(hard_df)

        print("=== INFERENCE SUMMARY ===")
        print(f"Input rows : {len(labeled_df)}")
        print(f"Aduan keywords loaded: {len(aduan_keywords)}")
        print(labeled_df["pred_label"].value_counts())
        print(f"Output CSV : {args.output_path}")
        if args.hard_cases_out:
            print(f"Hard cases : {hard_count}")
            print(f"Hard CSV   : {args.hard_cases_out}")
        return

    df_aduan = load_dataset(args.aduan_path)
    df_bukan = load_dataset(args.bukan_aduan_path)
    df = pd.concat([df_aduan, df_bukan], ignore_index=True)

    print("=== DATA SUMMARY ===")
    print(df["label"].value_counts().rename(index={0: "bukan_aduan", 1: "aduan_text"}))

    model, _ = train_and_evaluate(df, test_size=args.test_size)

    joblib.dump(model, args.model_out)
    print(f"Model SVM tersimpan di: {args.model_out}")

    if args.predict:
        label, confidence = predict_text(model, args.predict)
        print("=== SINGLE PREDICTION ===")
        print(f"Text       : {args.predict}")
        print(f"Prediksi   : {label}")
        print(f"Confidence : {confidence:.4f}")


if __name__ == "__main__":
    main()
