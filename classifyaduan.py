# classifyaduan.py (UPDATED VERSION)
import os
import re
import pandas as pd
from tqdm import tqdm

# Import enhanced classifier
from enhanced_classifyaduan import EnhancedAduanClassifier

# Keep your existing ADUAN_KEYWORDS for keyword extraction
ADUAN_KEYWORDS = {
    # Academic-related complaints
    'bpp', 'ukt', 'semester', 'nilai', 'krs', 'khs', 'magang', 'msib', 'sks', 
    'kuesioner', 'tugas akhir', 'wisuda', 'yudisium', 'akademik',
    'konversi', 'ipk', 'gelar',
    
    # System and IT complaints
    'igracias', 'lms', 'error', 'sistem', 'login', 'sso', 'lemot', 'tidak bisa akses',
    'bermasalah', 'down', 'gangguan', 'reset', 'password',
    
    # Facility complaints
    'rusak', 'toilet', 'parkir', 'gedung', 'ruangan',
    'kebersihan', 'fasilitas',
    
    # Service complaints
    'terlambat', 'tidak memuaskan', 'pelayanan buruk', 'lambat respon',
    'tidak profesional', 'mengecewakan',
    
    # Clear complaint indicators
    'tolong', 'mohon', 'kenapa', 'gimana', 'tidak bisa',
    'segera ditangani', 'perbaiki', 'ditindaklanjuti',
    
    # Financial complaints
    'tagihan', 'pembayaran', 'biaya', 'tunggakan', 'denda', 'bayar'
}

def preprocess_text(text):
    """Preprocess text for keyword extraction (keep existing)"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Handle common text variations
    text = text.replace('telyu!', '')
    text = text.replace('telyu', '')
    text = text.replace('telu', '')
    text = text.replace('tel-u', '')
    
    # Handle negation variations
    negation_map = {
        'gak': 'tidak',
        'nggak': 'tidak',
        'gk': 'tidak',
        'ga': 'tidak',
        'ngga': 'tidak',
        'g': 'tidak'
    }
    for old, new in negation_map.items():
        text = re.sub(rf'\b{old}\b', new, text)
    
    # Handle question variations
    text = re.sub(r'\b(gmn|gmana|gimana)\b', 'bagaimana', text)
    text = re.sub(r'\b(knp|knpa|kenape)\b', 'kenapa', text)
    
    # Handle common academic terms
    text = text.replace('krs', 'kartu rencana studi')
    text = text.replace('khs', 'kartu hasil studi')
    text = text.replace('sks', 'sistem kredit semester')
    text = text.replace('bpp', 'biaya penyelenggaraan pendidikan')
    
    # Clean up punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    
    return text

def get_aduan_keywords(text):
    """Extract aduan keywords found in text (keep existing)"""
    processed_text = preprocess_text(text)
    found_keywords = []
    
    for keyword in ADUAN_KEYWORDS:
        if keyword in processed_text:
            found_keywords.append(keyword)
            
    return found_keywords if found_keywords else ["-"]

def classify_dataset(file_path, use_enhanced=True):
    """Process entire dataset using enhanced classifier"""
    print(f"Starting aduan classification for file: {file_path}")
    print(f"Using enhanced classifier: {use_enhanced}")
    
    try:
        # Initialize enhanced classifier
        classifier = EnhancedAduanClassifier()
        if use_enhanced:
            classifier.load_ml_model()  # Try to load ML model
        
        # Read file
        if file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, engine='python')
            except Exception as e1:
                try:
                    df = pd.read_csv(file_path, engine='c')
                except Exception as e2:
                    try:
                        df = pd.read_csv(file_path)
                    except Exception as e3:
                        df = pd.read_csv(file_path, encoding='utf-8-sig')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format. Please use CSV or XLSX.")

        # Verify text column exists
        text_column = 'text'
        if text_column not in df.columns:
            potential_text_columns = [col for col in df.columns if 'text' in col.lower()]
            if potential_text_columns:
                text_column = potential_text_columns[0]
            else:
                raise ValueError(f"No text column found in the dataset")

        print("Processing texts for aduan classification...")
        tqdm.pandas(desc="Classifying texts")
        
        # Apply enhanced classification
        if use_enhanced:
            # Use enhanced classifier
            classification_results = df[text_column].progress_apply(
                lambda x: classifier.classify_text(x)
            )
            
            # Extract results
            df['aduan_type'] = classification_results.apply(lambda x: x[0])
            df['confidence_score'] = classification_results.apply(lambda x: x[1])
            df['ml_probability'] = classification_results.apply(lambda x: x[2])
        else:
            # Fallback to basic classification (for testing)
            def basic_classify(text):
                classification, score, prob = classifier.classify_text(text)
                return classification
                
            df['aduan_type'] = df[text_column].progress_apply(basic_classify)
            df['confidence_score'] = 0
            df['ml_probability'] = None
        
        # Extract keywords using existing function
        df['aduan_keywords'] = df[text_column].progress_apply(get_aduan_keywords)
        
        # Create output path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_aduan_classified.xlsx"
        
        # Save results
        df.to_excel(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
        # Print enhanced statistics
        aduan_count = len(df[df['aduan_type'] == 'aduan_text'])
        review_count = len(df[df['aduan_type'] == 'review_needed'])
        total_count = len(df)
        
        print(f"\nClassification Results:")
        print(f"Total texts: {total_count}")
        print(f"Confident aduan texts: {aduan_count}")
        print(f"Need manual review: {review_count}")
        print(f"Not aduan texts: {total_count - aduan_count - review_count}")
        
        if total_count > 0:
            print(f"Confident aduan percentage: {(aduan_count/total_count)*100:.2f}%")
            print(f"Review needed percentage: {(review_count/total_count)*100:.2f}%")
        
        # Return only confident aduan texts for next step
        confident_aduan = df[df['aduan_type'] == 'aduan_text']
        print(f"\nPassing {len(confident_aduan)} confident aduan texts to next stage")
        
        return confident_aduan

    except Exception as e:
        print(f"Error during aduan classification: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

def get_filtered_dataset(file_path):
    """Get only aduan texts for further classification (keep existing)"""
    try:
        aduan_df = classify_dataset(file_path)
        return aduan_df
    except Exception as e:
        print(f"Error getting filtered dataset: {e}")
        raise