# balance_checker.py
from enhanced_classifyaduan import EnhancedAduanClassifier
import pandas as pd

def check_balance():
    """Check if model is balanced"""
    print("=== BALANCE CHECK ===")
    
    df = pd.read_csv('penting/aduan_approved.csv')
    
    # Prepare true labels
    if 'corrected_type' in df.columns:
        y_true = df['corrected_type'].fillna(df['aduan_type'])
    else:
        y_true = df['aduan_type']
    
    classifier = EnhancedAduanClassifier()
    classifier.load_ml_model()
    
    predictions = []
    
    for text in df['text']:
        classification, score, ml_prob = classifier.classify_text(text)
        predictions.append(classification)
    
    # Analysis
    pred_aduan = predictions.count('aduan_text')
    pred_not_aduan = predictions.count('not_aduan_text') 
    pred_review = predictions.count('review_needed')
    
    true_aduan = (y_true == 'aduan_text').sum()
    true_not_aduan = (y_true == 'not_aduan_text').sum()
    
    print(f"True Distribution: Aduan={true_aduan}, Not Aduan={true_not_aduan}")
    print(f"Pred Distribution: Aduan={pred_aduan}, Not Aduan={pred_not_aduan}, Review={pred_review}")
    
    # Balance metrics
    aduan_ratio_pred = pred_aduan / len(predictions)
    aduan_ratio_true = true_aduan / len(y_true)
    
    print(f"\nAduan Ratio - True: {aduan_ratio_true:.3f}, Pred: {aduan_ratio_pred:.3f}")
    
    if aduan_ratio_pred < 0.1:
        print("❌ Model TOO CONSERVATIVE - predicting too few aduan")
    elif aduan_ratio_pred > 0.3:
        print("❌ Model TOO AGGRESSIVE - predicting too many aduan")  
    else:
        print("✅ Model BALANCED - good aduan prediction ratio")
    
    # Test specific borderline cases
    print("\n=== BORDERLINE CASES TEST ===")
    borderline_texts = [
        "Telyu! tolong kasih semangat sender cape banget",  # Should be aduan
        "telyu! butuh temen ngobrol nih",  # Should be aduan  
        "rekomendasi cafe enak dong",  # Should not aduan
        "nilai KHS saya kok salah ya?",  # Should be aduan
    ]
    
    for text in borderline_texts:
        classification, score, ml_prob = classifier.classify_text(text)
        print(f"'{text}' -> {classification} (score: {score:.2f})")

if __name__ == "__main__":
    check_balance()