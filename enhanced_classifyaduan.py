# enhanced_classifyaduan.py - FIXED VERSION
import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class EnhancedAduanClassifier:
    def __init__(self):
        self.ml_model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = 'ml_models/aduan_classifier_model.pkl'
        self.vectorizer_path = 'ml_models/aduan_vectorizer.pkl'
        
        # Create models directory if not exists
        os.makedirs('ml_models', exist_ok=True)
        
    # Enhanced keyword rules dengan context awareness
    STRONG_ADUAN_INDICATORS = {
        # Academic complaints
        'bpp telat': 2, 'ukt masalah': 2, 'nilai salah': 2, 'krs error': 2,
        'khs belum': 2, 'sks kurang': 2, 'magang masalah': 2, 'nilai belum': 1.5,
        'ipk salah': 2, 'wisuda masalah': 2,
        
        # System/IT complaints
        'igracias error': 2, 'lms down': 2, 'lms error': 2, 'sistem bermasalah': 2, 
        'login gagal': 2, 'wifi lemot': 1.5, 'wifi down': 1.5, 'error sistem': 2,
        'sistem down': 2, 'gabisa login': 1.5, 'tidak bisa akses': 1.5,
        
        # Facility complaints
        'toilet rusak': 2, 'parkir penuh': 1, 'gedung rusak': 2,
        'ruangan kotor': 1.5, 'fasilitas rusak': 2, 'ac rusak': 1.5,
        'listrik mati': 1.5, 'air mati': 1.5,
        
        # Service complaints
        'pelayanan buruk': 2, 'respon lambat': 1.5, 'tidak profesional': 2,
        'staff kasar': 2, 'administrasi lama': 1.5,
        
        # Financial complaints
        'tagihan salah': 2, 'pembayaran masalah': 2, 'biaya keliru': 2,
        'tunggakan': 1.5, 'denda': 1.5,
        # Emotional support requests
        'kasih semangat': 1.2, 'butuh semangat': 1.2, 'butuh motivasi': 1.0,
        'butuh dukungan': 1.0, 'butuh support': 1.0,
        
        # Social isolation
        'butuh temen': 1.0, 'butuh teman': 1.0, 'butuh ngobrol': 0.8,
        'sendiri': 0.6, 'kesepian': 0.8, 'sepi': 0.6,
        
        # Academic stress
        'cape banget': 1.0, 'lelah banget': 1.0, 'penat banget': 0.8,
        'stress banget': 1.2, 'burnout': 1.2, 'kewalahan': 0.8,
        
        # Career anxiety  
        'bingung karir': 0.8, 'takut masa depan': 0.8, 'khawatir lulus': 0.8,
        
        
    }
    
    WEAK_INDICATORS = {
        'tolong': 0.3, 'mohon': 0.3, 'kenapa': 0.5, 'gimana': 0.5,
        'bagaimana': 0.5, 'bisa': 0.2, 'masalah': 1.0, 'error': 1.0,
        'rusak': 1.0, 'belum': 0.6, 'kapan': 0.4, 'salah': 1.0,
        'lemot': 0.8, 'down': 0.8, 'gagal': 0.8, 'kotor': 0.7,
        'telat': 0.7, 'lambat': 0.6, 'buruk': 0.8, 'frustasi': 0.7, 'bantu': 0.4, 'bantuan': 0.4, 
        'semangat': 0.5, 'motivasi': 0.5, 'dukungan': 0.5, 'support': 0.4,
        'temen': 0.4, 'teman': 0.4, 'ngobrol': 0.4, 'curhat': 0.4,
        'sendiri': 0.4, 'sepi': 0.4, 'kesepian': 0.5,
        'cape': 0.5, 'lelah': 0.5, 'penat': 0.4, 'stress': 0.6,
        'bingung': 0.4, 'takut': 0.4, 'khawatir': 0.4,
    }
    
    # FIXED: Remove byte strings (b'pattern') and use normal strings
    NON_ADUAN_PATTERNS = [
        # Casual conversation starters
        'ini kejadian', 'derita kuliah', 'wdyt', 'what do you think',
        'mau tanya', 'penasaran', 'kepo', 'curious',
        
        # Social/relationship topics
        'jatuh cinta', 'cowok', 'cewek', 'pacaran', 'jomblo', 
        'selingkuh', 'gebetan', 'pdkt', 'mantan',
        
        # General campus life (bukan keluhan)
        'kuliah di', 'kampus', 'mahasiswa', 'kegiatan kampus',
        'organisasi', 'ukm', 'ekstrakurikuler',
        
        # Personal stories/curhat
        'pengalaman', 'cerita', 'curhat', 'sharing',
        'sedih', 'senang', 'bahagia', 'galau', 'rindu', 'kangen',
        
        # Questions about general topics
        'gimana pendapat', 'bagaimana menurut', 'apa pendapat',
        'setuju ga', 'percaya ga', 'yakin ga',
        
        # Recommendations & information seeking
        'rekomendasi', 'rekom', 'spill', 'saran', 'saranin',
        'bagi nomor', 'info loker', 'tempat.*enak', 'cafe.*bagus',
        'makanan.*recommend', 'jam buka', 'berapa harga',
        
        # Social networking
        'ada yang', 'mau kenalan', 'cari teman', 'cari pacar',
        'yang punya pengalaman', 'share pengalaman', 'bagi pengalaman',
        
        # General questions
        'gimana cara', 'bagaimana.*caranya', 'cara daftar',
        'boleh ga', 'bisa ga', 'apa bisa',
        
        # Casual greetings
        'selamat pagi', 'selamat siang', 'selamat malam', 'semangat',
        'makasih', 'thanks', 'terima kasih',
        
        # Personal preferences/opinions
        'suka', 'seneng', 'setuju', 'percaya', 'yakin',
        'gembira', 'senang', 'bahagia',
        
        # Future plans/hopes
        'pengen', 'mau', 'ingin', 'rencana', 'harapan',
        'semoga', 'mudah-mudahan', 'insyaallah',
        
        # General discussions
        'menurut kalian', 'menurut kamu', 'apa pendapat',
        'gimana menurut', 'bagaimana pendapat',
        
        # Strong greeting patterns
        r'^selamat pagi', r'^selamat siang', r'^selamat malam', 
        r'^selamat sore', r'^good morning', r'^hai', r'^halo',
        r'semangat pagi', r'semangat semua', r'semangat guys',
    ]

    # Enhanced STRONG non-aduan indicators
    STRONG_NON_ADUAN_KEYWORDS = {
        'spill': -2, 'wdyt': -2, 'curhat': -1, 'sharing': -1, 'cerita': -1,
        'pengalaman': -1, 'kenalan': -1.5, 'temen': -1, 'pacaran': -1,
        'cowok': -1,  'selingkuh': -1, 'gebetan': -1,
        'mantan': -1, 'jomblo': -1, 'pdkt': -1,'cafe': -1.5, 'wifi': -1.0, 'nugas': -1.0, 'tempat': -0.8,
        'rekomendasi': -2.0, 'rekom': -2.0, 'saran': -1.5, 'tips': -1.0,
        'drop': -1.5, 'nomor': -1.0, 'bagi': -1.0, 'info': -0.8,
        'cowo': -1.0, 'cewek': -1.0, 'ganteng': -1.0, 'baik': -0.5,
        '2024': -1.0, 'usaha': -0.5, 'fif': -0.5,
    }

    def preprocess_text(self, text):
        """Enhanced preprocessing"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Normalize common variations
        normalization_map = {
            'gak': 'tidak', 'nggak': 'tidak', 'ga': 'tidak', 'ngga': 'tidak',
            'gmn': 'bagaimana', 'gimana': 'bagaimana', 
            'knp': 'kenapa', 'kenape': 'kenapa',
            'bgtt': 'banget', 'bgt': 'banget',
            'plis': 'tolong', 'pls': 'tolong',
            'telyu': '', 'telu': '', 'tel-u': ''
        }
        
        for old, new in normalization_map.items():
            text = re.sub(rf'\b{old}\b', new, text)
            
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text

    def calculate_rule_score(self, text):
        """More conservative untuk reduce false positives"""
        score = 0
        processed_text = self.preprocess_text(text)
        
        # SPECIAL CASE: Strong non-aduan patterns
        strong_non_aduan = [
            'cafe.*enak', 'rekomendasi.*cafe', 'tempat.*nugas', 
            'wifi.*kenceng', 'drop nomor', 'bagi nomor',
            '2024.*cowo', 'cowo.*fif', 'ganteng.*baik',
            'minta saran', 'saran.*dong', 'tips.*dong'
        ]
        
        for pattern in strong_non_aduan:
            if re.search(pattern, processed_text):
                return -4.0  # Very strong non-aduan
        
        # Regular non-aduan patterns
        non_aduan_penalty = 0
        for pattern in self.NON_ADUAN_PATTERNS:
            if re.search(pattern, processed_text):
                non_aduan_penalty -= 1
        
        score += max(non_aduan_penalty, -2)
        
        # Check strong non-aduan keywords
        for keyword, weight in self.STRONG_NON_ADUAN_KEYWORDS.items():
            if re.search(rf'\b{keyword}\b', processed_text):
                score += weight
        
        # Check strong indicators
        for indicator, weight in self.STRONG_ADUAN_INDICATORS.items():
            if indicator in processed_text:
                score += weight
                    
        # Check weak indicators - REDUCE SLIGHTLY
        for indicator, weight in self.WEAK_INDICATORS.items():
            if re.search(rf'\b{indicator}\b', processed_text):
                score += weight * 1.2  # Reduced from 1.5
                    
        # Length-based adjustment
        word_count = len(processed_text.split())
        if word_count < 5:
            score -= 0.5
        elif word_count > 30:
            score += 0.2
                
        return score

    def train_ml_model(self, labeled_data_path):
        """Train ML model dengan data yang sudah di-label"""
        print("Training ML model dengan data:", labeled_data_path)
        
        df = pd.read_csv(labeled_data_path)
        
        # Prepare features - use corrected_type if available, otherwise use aduan_type
        texts = df['text'].apply(self.preprocess_text)
        
        # Determine labels: prefer corrected_type, fallback to aduan_type
        if 'corrected_type' in df.columns:
            labels = df['corrected_type'].fillna(df['aduan_type'])
        else:
            labels = df['aduan_type']
        
        # Convert labels to binary (1 for aduan_text, 0 for not_aduan_text)
        y = (labels == 'aduan_text').astype(int)
        
        print(f"Label distribution: {y.value_counts().to_dict()}")
        
        # Create features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # Train model
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.ml_model.fit(X, y)
        self.is_trained = True
        
        # Save model
        joblib.dump(self.ml_model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        print("ML model trained and saved successfully!")
        
        # Show feature importance
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.ml_model.feature_importances_
        top_features = sorted(zip(feature_names, importances), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        print("Top 10 important features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")

    def load_ml_model(self):
        """Load pre-trained ML model - prefer balanced model"""
        try:
            # Try to load balanced model first
            balanced_model_path = 'ml_models/balanced_aduan_classifier_model.pkl'
            balanced_vectorizer_path = 'ml_models/balanced_aduan_vectorizer.pkl'
            
            if os.path.exists(balanced_model_path) and os.path.exists(balanced_vectorizer_path):
                self.ml_model = joblib.load(balanced_model_path)
                self.vectorizer = joblib.load(balanced_vectorizer_path)
                self.model_path = balanced_model_path
                self.vectorizer_path = balanced_vectorizer_path
                self.is_trained = True
                print("Balanced ML model loaded successfully!")
            elif os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                # Fallback to original model
                self.ml_model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.is_trained = True
                print("Original ML model loaded successfully!")
            else:
                print("No ML model found, using rule-based only")
                self.is_trained = False
        except Exception as e:
            print(f"Error loading ML model: {e}")
            self.is_trained = False

    def ml_predict(self, text):
        """ML prediction jika model tersedia"""
        if not self.is_trained:
            return None
            
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        proba = self.ml_model.predict_proba(X)[0]
        
        return proba[1]  # Probability of being aduan

    def classify_text(self, text):
        """Balanced classification - moderate approach"""
        # Rule-based scoring
        rule_score = self.calculate_rule_score(text)
        
        # ML prediction jika available
        ml_prob = self.ml_predict(text)
        
        # MODERATE DECISION LOGIC - Balanced approach
        if rule_score <= -2.5:  # Clear non-aduan
            return 'not_aduan_text', rule_score, ml_prob
            
        if ml_prob is not None:
            # Balanced thresholds
            if ml_prob > 0.55 and rule_score > 0.2:  # Good confidence both
                return 'aduan_text', rule_score + ml_prob, ml_prob
            elif ml_prob > 0.7:  # Very high ML confidence
                return 'aduan_text', ml_prob * 10, ml_prob
            elif rule_score > 1.2:  # Strong rule confidence
                return 'aduan_text', rule_score, ml_prob
            elif ml_prob > 0.35 or rule_score > 0.0:  # Weak signals -> review
                return 'review_needed', (rule_score + ml_prob) / 2, ml_prob
            else:
                return 'not_aduan_text', rule_score, ml_prob
        else:
            # Rule-based only - moderate
            if rule_score > 0.8:
                return 'aduan_text', rule_score, ml_prob
            elif rule_score > 0.1:
                return 'review_needed', rule_score, ml_prob
            else:
                return 'not_aduan_text', rule_score, ml_prob
# Compatibility function untuk existing code
def enhanced_classify_aduan(text):
    """Wrapper function untuk compatibility dengan existing code"""
    classifier = EnhancedAduanClassifier()
    classifier.load_ml_model()  # Try to load ML model
    classification, score, ml_prob = classifier.classify_text(text)
    return classification