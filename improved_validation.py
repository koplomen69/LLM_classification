# improved_validation.py
from enhanced_classifyaduan import EnhancedAduanClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def detailed_validation():
    """Detailed validation dengan metrics lengkap"""
    print("=== DETAILED VALIDATION ===")
    
    # Load data
    df = pd.read_csv('penting/aduan_approved.csv')
    
    # Prepare true labels
    if 'corrected_type' in df.columns:
        y_true = df['corrected_type'].fillna(df['aduan_type'])
    else:
        y_true = df['aduan_type']
    
    # Convert to binary
    y_true_binary = (y_true == 'aduan_text').astype(int)
    
    # Predict with enhanced classifier
    classifier = EnhancedAduanClassifier()
    classifier.load_ml_model()
    
    y_pred = []
    y_pred_binary = []
    
    for text in df['text']:
        classification, score, ml_prob = classifier.classify_text(text)
        y_pred.append(classification)
        y_pred_binary.append(1 if classification == 'aduan_text' else 0)
    
    # Detailed analysis
    results_df = pd.DataFrame({
        'text': df['text'],
        'true_label': y_true,
        'pred_label': y_pred,
        'match': y_true == y_pred
    })
    
    # Calculate metrics
    accuracy = (results_df['match'].sum() / len(results_df)) * 100
    
    print(f"Overall Accuracy: {accuracy:.1f}%")
    print(f"Total samples: {len(results_df)}")
    print(f"Correct: {results_df['match'].sum()}")
    print(f"Wrong: {len(results_df) - results_df['match'].sum()}")
    
    # Show confusion matrix
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    print("Actual \\ Predicted | Not Aduan | Aduan")
    print("-------------------|-----------|------")
    print(f"Not Aduan         | {cm[0,0]:9} | {cm[0,1]:5}")
    print(f"Aduan             | {cm[1,0]:9} | {cm[1,1]:5}")
    
    # Show classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true_binary, y_pred_binary, 
                              target_names=['Not Aduan', 'Aduan']))
    
    # Show some misclassified examples
    print("\n=== MISCLASSIFIED EXAMPLES ===")
    misclassified = results_df[~results_df['match']]
    
    print("False Positives (Predicted Aduan, Actual Not Aduan):")
    fp = misclassified[(misclassified['pred_label'] == 'aduan_text') & 
                      (misclassified['true_label'] == 'not_aduan_text')]
    for idx, row in fp.head(5).iterrows():
        print(f"  - {row['text'][:80]}...")
    
    print("\nFalse Negatives (Predicted Not Aduan, Actual Aduan):")
    fn = misclassified[(misclassified['pred_label'] == 'not_aduan_text') & 
                      (misclassified['true_label'] == 'aduan_text')]
    for idx, row in fn.head(5).iterrows():
        print(f"  - {row['text'][:80]}...")
    
    return results_df

if __name__ == "__main__":
    results = detailed_validation()