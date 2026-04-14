import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import json

class DatasetComparator:
    def __init__(self):
        self.le = LabelEncoder()
    
    def load_datasets(self, approved_path, not_approved_path):
        """Load both datasets and prepare for comparison"""
        try:
            print(f"Loading approved dataset from: {approved_path}")
            print(f"Loading not-approved dataset from: {not_approved_path}")
            
            # Deteksi format file berdasarkan ekstensi
            approved_ext = os.path.splitext(approved_path)[1].lower()
            not_approved_ext = os.path.splitext(not_approved_path)[1].lower()
            
            # Load approved dataset
            if approved_ext in ['.xlsx', '.xls']:
                # Coba berbagai engine untuk Excel
                engines = ['openpyxl', 'xlrd']
                for engine in engines:
                    try:
                        approved_df = pd.read_excel(approved_path, engine=engine)
                        print(f"Approved dataset loaded with {engine} engine")
                        break
                    except Exception as e:
                        print(f"{engine} failed: {e}")
                        continue
                else:
                    raise Exception("All Excel engines failed for approved dataset")
            else:
                # Untuk CSV, coba berbagai encoding
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        approved_df = pd.read_csv(approved_path, encoding=encoding)
                        print(f"Approved dataset loaded with {encoding} encoding")
                        break
                    except Exception as e:
                        print(f"{encoding} encoding failed: {e}")
                        continue
                else:
                    raise Exception("All encodings failed for approved dataset")
            
            # Load not-approved dataset
            if not_approved_ext in ['.xlsx', '.xls']:
                # Coba berbagai engine untuk Excel
                engines = ['openpyxl', 'xlrd']
                for engine in engines:
                    try:
                        not_approved_df = pd.read_excel(not_approved_path, engine=engine)
                        print(f"Not-approved dataset loaded with {engine} engine")
                        break
                    except Exception as e:
                        print(f"{engine} failed: {e}")
                        continue
                else:
                    raise Exception("All Excel engines failed for not-approved dataset")
            else:
                # Untuk CSV, coba berbagai encoding
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        not_approved_df = pd.read_csv(not_approved_path, encoding=encoding)
                        print(f"Not-approved dataset loaded with {encoding} encoding")
                        break
                    except Exception as e:
                        print(f"{encoding} encoding failed: {e}")
                        continue
                else:
                    raise Exception("All encodings failed for not-approved dataset")
            
            print(f"Approved dataset shape: {approved_df.shape}")
            print(f"Not-approved dataset shape: {not_approved_df.shape}")
            print(f"Approved dataset columns: {approved_df.columns.tolist()}")
            print(f"Not-approved dataset columns: {not_approved_df.columns.tolist()}")
            
            return approved_df, not_approved_df
            
        except Exception as e:
            raise Exception(f"Error loading datasets: {str(e)}")
    
    def preprocess_datasets(self, approved_df, not_approved_df):
        """Preprocess and align both datasets for comparison"""
        # Standardize column names (convert to lowercase)
        approved_df.columns = [col.lower().strip() for col in approved_df.columns]
        not_approved_df.columns = [col.lower().strip() for col in not_approved_df.columns]
        
        print("Columns in approved dataset:", approved_df.columns.tolist())
        print("Columns in not-approved dataset:", not_approved_df.columns.tolist())
        
        # Cari kolom yang berisi ground truth dan prediksi
        actual_direktorat_col = None
        predicted_direktorat_col = None
        
        # Untuk approved dataset - cari kolom ground truth
        for col in approved_df.columns:
            if 'actual' in col and 'direktorat' in col:
                actual_direktorat_col = col
                break
            elif 'true' in col and 'label' in col:
                actual_direktorat_col = col
                break
        
        # Jika tidak ditemukan, cari kolom 'direktorat' atau kolom pertama
        if not actual_direktorat_col:
            if 'direktorat' in approved_df.columns:
                actual_direktorat_col = 'direktorat'
            else:
                # Gunakan kolom ke-4 (index 3) berdasarkan struktur file Anda
                if len(approved_df.columns) >= 4:
                    actual_direktorat_col = approved_df.columns[3]
                else:
                    actual_direktorat_col = approved_df.columns[0]
        
        # Untuk not-approved dataset - cari kolom prediksi
        for col in not_approved_df.columns:
            if 'predicted' in col and 'direktorat' in col:
                predicted_direktorat_col = col
                break
            elif 'predicted' in col and 'label' in col:
                predicted_direktorat_col = col
                break
        
        # Jika tidak ditemukan, cari kolom 'direktorat' atau kolom pertama
        if not predicted_direktorat_col:
            if 'direktorat' in not_approved_df.columns:
                predicted_direktorat_col = 'direktorat'
            else:
                # Gunakan kolom ke-4 (index 3) berdasarkan struktur file Anda
                if len(not_approved_df.columns) >= 4:
                    predicted_direktorat_col = not_approved_df.columns[3]
                else:
                    predicted_direktorat_col = not_approved_df.columns[0]
        
        # Cari kolom text
        text_col_approved = 'text'
        text_col_not_approved = 'text'
        
        if 'text' not in approved_df.columns:
            for col in approved_df.columns:
                if 'text' in col:
                    text_col_approved = col
                    break
            else:
                text_col_approved = approved_df.columns[0]  # Gunakan kolom pertama sebagai fallback
        
        if 'text' not in not_approved_df.columns:
            for col in not_approved_df.columns:
                if 'text' in col:
                    text_col_not_approved = col
                    break
            else:
                text_col_not_approved = not_approved_df.columns[0]  # Gunakan kolom pertama sebagai fallback
        
        print(f"Using '{actual_direktorat_col}' as actual direktorat column")
        print(f"Using '{predicted_direktorat_col}' as predicted direktorat column")
        print(f"Using '{text_col_approved}' as text column in approved dataset")
        print(f"Using '{text_col_not_approved}' as text column in not-approved dataset")
        
        # Clean and standardize text - convert to string first to handle any non-string values
        approved_df['text_clean'] = approved_df[text_col_approved].astype(str).str.lower().str.strip()
        not_approved_df['text_clean'] = not_approved_df[text_col_not_approved].astype(str).str.lower().str.strip()
        
        # Clean Direktorat values - convert to string and handle NaN
        approved_df['direktorat_clean'] = approved_df[actual_direktorat_col].fillna('unknown').astype(str).str.lower().str.strip()
        not_approved_df['direktorat_clean'] = not_approved_df[predicted_direktorat_col].fillna('unknown').astype(str).str.lower().str.strip()
        
        return approved_df, not_approved_df
    
    def align_datasets(self, approved_df, not_approved_df):
        """Align datasets based on text matching"""
        # Create mapping based on text similarity
        matched_data = []
        
        for idx, not_approved_row in not_approved_df.iterrows():
            not_approved_text = not_approved_row['text_clean']
            
            # Skip empty texts
            if not not_approved_text or not_approved_text == 'nan' or not_approved_text == 'none':
                continue
                
            # Find exact matching text in approved dataset
            matches = approved_df[approved_df['text_clean'] == not_approved_text]
            
            if len(matches) > 0:
                approved_row = matches.iloc[0]
                matched_data.append({
                    'text': str(not_approved_text),
                    'actual_direktorat': str(approved_row['direktorat_clean']),
                    'predicted_direktorat': str(not_approved_row['direktorat_clean']),
                    'actual_keyword': str(approved_row.get('keyword', approved_row.get('kategori', ''))),
                    'predicted_keyword': str(not_approved_row.get('keyword', not_approved_row.get('kategori', '')))
                })
            else:
                # Try partial matching for texts that don't match exactly
                if len(not_approved_text) > 10:
                    # Try different substring lengths
                    for substr_length in [30, 20, 15, 10]:
                        substring = not_approved_text[:substr_length]
                        similar_matches = approved_df[
                            approved_df['text_clean'].str.contains(substring, na=False, regex=False)
                        ]
                        
                        if len(similar_matches) > 0:
                            approved_row = similar_matches.iloc[0]
                            matched_data.append({
                                'text': str(not_approved_text),
                                'actual_direktorat': str(approved_row['direktorat_clean']),
                                'predicted_direktorat': str(not_approved_row['direktorat_clean']),
                                'actual_keyword': str(approved_row.get('keyword', approved_row.get('kategori', ''))),
                                'predicted_keyword': str(not_approved_row.get('keyword', not_approved_row.get('kategori', '')))
                            })
                            break
        
        matched_df = pd.DataFrame(matched_data)
        print(f"Successfully matched {len(matched_df)} out of {len(not_approved_df)} records")
        
        if len(matched_df) == 0:
            print("No matches found. Sample of texts from both datasets:")
            print("Approved dataset sample texts:")
            print(approved_df['text_clean'].head(3).tolist())
            print("Not-approved dataset sample texts:")
            print(not_approved_df['text_clean'].head(3).tolist())
        
        return matched_df
    
    def calculate_metrics(self, matched_df):
        """Calculate evaluation metrics with proper serialization"""
        if len(matched_df) == 0:
            return {
                'overall_metrics': {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'total_samples': 0
                },
                'class_metrics': {},
                'confusion_matrix': [],
                'matched_data': []
            }
        
        # Ensure all data is string type for serialization
        matched_df = matched_df.copy()
        matched_df['actual_direktorat'] = matched_df['actual_direktorat'].astype(str)
        matched_df['predicted_direktorat'] = matched_df['predicted_direktorat'].astype(str)
        
        y_true = matched_df['actual_direktorat']
        y_pred = matched_df['predicted_direktorat']
        
        # Get unique labels and create mapping
        all_labels = sorted(set(y_true) | set(y_pred))
        label_mapping = {label: idx for idx, label in enumerate(all_labels)}
        
        # Convert to encoded values
        y_true_encoded = np.array([label_mapping[label] for label in y_true])
        y_pred_encoded = np.array([label_mapping[label] for label in y_pred])
        
        # Calculate metrics
        accuracy = float(accuracy_score(y_true_encoded, y_pred_encoded))
        precision = float(precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0))
        recall = float(recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0))
        f1 = float(f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0))
        
        # Create confusion matrix data
        confusion_data = []
        for actual_label in all_labels:
            for predicted_label in all_labels:
                count = int(len(matched_df[
                    (matched_df['actual_direktorat'] == actual_label) & 
                    (matched_df['predicted_direktorat'] == predicted_label)
                ]))
                confusion_data.append({
                    'actual': str(actual_label),
                    'predicted': str(predicted_label),
                    'count': count
                })
        
        # Calculate per-class metrics
        class_metrics = {}
        for label in all_labels:
            true_positives = int(len(matched_df[
                (matched_df['actual_direktorat'] == label) & 
                (matched_df['predicted_direktorat'] == label)
            ]))
            false_positives = int(len(matched_df[
                (matched_df['actual_direktorat'] != label) & 
                (matched_df['predicted_direktorat'] == label)
            ]))
            false_negatives = int(len(matched_df[
                (matched_df['actual_direktorat'] == label) & 
                (matched_df['predicted_direktorat'] != label)
            ]))
            
            precision_class = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0.0
            recall_class = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
            f1_class = float(2 * (precision_class * recall_class) / (precision_class + recall_class)) if (precision_class + recall_class) > 0 else 0.0
            
            class_metrics[str(label)] = {
                'precision': round(precision_class, 4),
                'recall': round(recall_class, 4),
                'f1_score': round(f1_class, 4),
                'support': int(true_positives + false_negatives)
            }
        
        # Prepare matched data for serialization
        serializable_matched_data = []
        for _, row in matched_df.iterrows():
            serializable_matched_data.append({
                'text': str(row['text']),
                'actual_direktorat': str(row['actual_direktorat']),
                'predicted_direktorat': str(row['predicted_direktorat']),
                'actual_keyword': str(row.get('actual_keyword', '')),
                'predicted_keyword': str(row.get('predicted_keyword', ''))
            })
        
        return {
            'overall_metrics': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'total_samples': int(len(matched_df))
            },
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_data,
            'matched_data': serializable_matched_data
        }
    
    def evaluate(self, approved_path, not_approved_path):
        """Main evaluation function"""
        try:
            print("Loading datasets...")
            approved_df, not_approved_df = self.load_datasets(approved_path, not_approved_path)
            
            print("Preprocessing datasets...")
            approved_df, not_approved_df = self.preprocess_datasets(approved_df, not_approved_df)
            
            print("Aligning datasets...")
            matched_df = self.align_datasets(approved_df, not_approved_df)
            
            print("Calculating metrics...")
            metrics = self.calculate_metrics(matched_df)
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Evaluation failed: {str(e)}")

def compare_datasets(approved_path, not_approved_path):
    """
    Compare approved dataset with not-approved dataset and return evaluation metrics
    """
    comparator = DatasetComparator()
    return comparator.evaluate(approved_path, not_approved_path)