import os
import re
import pandas as pd
import time
import builtins
from tqdm import tqdm as original_tqdm
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Backward compatibility for older LangChain versions.
    from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from prompt_manager import prompt_manager

# ✅ P0 FIX: Import improvements dari suggested_improvements module
try:
    from suggested_improvements import (
        DIREKTORAT_STANDARDIZATION,
        VALID_DIREKTORATS,
        ABBREVIATION_EXPANSIONS,
        PARSING_PATTERNS_COMPILED,
        parse_llm_output_improved,
        normalize_direktorat_name_optimized,
        validate_keyword_against_text,
        is_generic_pattern,
        extract_direktorat_by_mention,
        PerformanceMetrics,
    )
    IMPROVEMENTS_AVAILABLE = True
    print("✅ Improvements module loaded successfully!")
except ImportError as e:
    print(f"⚠️ Improvements module not available: {e}")
    IMPROVEMENTS_AVAILABLE = False


def _env_int(name, default):
    """Read integer env var safely with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


CLASSIFY_MAX_OUTPUT_TOKENS = _env_int('CLASSIFY_MAX_OUTPUT_TOKENS', 70)
CLASSIFY_PROGRESS_UPDATE_EVERY = max(1, _env_int('CLASSIFY_PROGRESS_UPDATE_EVERY', 25))
CLASSIFY_CHECKPOINT_EVERY = max(1, _env_int('CLASSIFY_CHECKPOINT_EVERY', 200))
CLASSIFY_VERBOSE_PARSING = os.getenv('CLASSIFY_VERBOSE_PARSING', '0') == '1'
CLASSIFY_DEFAULT_MODE = os.getenv('CLASSIFY_DEFAULT_MODE', 'pure_llm')
CLASSIFY_USE_STOP_SEQUENCES = os.getenv('CLASSIFY_USE_STOP_SEQUENCES', '1') == '1'
CLASSIFY_TEMPERATURE = float(os.getenv('CLASSIFY_TEMPERATURE', '0.0'))
CLASSIFY_EMPTY_RETRY_STRATEGY = os.getenv('CLASSIFY_EMPTY_RETRY_STRATEGY', 'db').strip().lower()
CLASSIFY_STRICT_KEYWORD_DEFAULT = os.getenv('CLASSIFY_STRICT_KEYWORD_DEFAULT', '1') == '1'

FAST_MODE_RULES = [
    ('Keuangan', ['bpp', 'ukt', 'bayar', 'pembayaran', 'biaya', 'semesteran']),
    ('Akademik', ['krs', 'sks', 'semester', 'nilai', 'ipk', 'tpa', 'wisuda', 'sempro', 'cuti']),
    ('Pusat Teknologi Informasi', ['igracias', 'lms', 'sso', 'wifi', 'ktm', 'sistem', 'notif']),
    ('Sumber Daya Manusia', ['dosen', 'edom', 'bimbingan', 'wal dosen']),
    ('Kemahasiswaan, Pengembangan Karir, Alumni', ['loker', 'magang', 'kp', 'bangkit', 'parkir', 'organisasi']),
]


def _safe_print(*args, **kwargs):
    """Print wrapper that avoids Windows charmap crashes for Unicode logs."""
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for arg in args:
            try:
                text = str(arg)
            except Exception:
                text = repr(arg)
            safe_args.append(text.encode('ascii', errors='replace').decode('ascii'))
        builtins.print(*safe_args, **kwargs)


print = _safe_print

# ✅ PERBAIKAN: Custom tqdm yang aman untuk Flask
def safe_tqdm(*args, **kwargs):
    """Safe tqdm untuk environment Flask"""
    if 'WERKZEUG_RUN_MAIN' in os.environ or 'FLASK_APP' in os.environ:
        # Nonaktifkan progress bar di web environment
        kwargs['disable'] = True
    return original_tqdm(*args, **kwargs)

# Ganti tqdm dengan safe_tqdm
tqdm = safe_tqdm

# ==================== OPTIMIZER CLASSES ====================

class UniversalPromptOptimizer:
    def __init__(self):
        self.component_cache = {}
        self.model_profiles = {
            # Qwen series
            'qwen': {'optimal_context': 16000, 'preferred_format': 'chatml'},
            'qwen2': {'optimal_context': 16000, 'preferred_format': 'chatml'},
            'qwen3': {'optimal_context': 16000, 'preferred_format': 'chatml'},
            
            # Llama series
            'llama': {'optimal_context': 4096, 'preferred_format': 'llama'},
            'llama2': {'optimal_context': 4096, 'preferred_format': 'llama'},
            'llama3': {'optimal_context': 8192, 'preferred_format': 'llama'},
            
            # Mistral series
            'mistral': {'optimal_context': 8192, 'preferred_format': 'mistral'},
            'mixtral': {'optimal_context': 32768, 'preferred_format': 'mistral'},
            
            # Phi series
            'phi': {'optimal_context': 2048, 'preferred_format': 'chatml'},
            
            # Gemma series
            'gemma': {'optimal_context': 8192, 'preferred_format': 'chatml'},
            
            # Default profile
            'default': {'optimal_context': 4096, 'preferred_format': 'chatml'}
        }
    
    def detect_model_family(self, model_name):
        """Detect model family dari nama model"""
        model_lower = model_name.lower()
        
        for family in self.model_profiles:
            if family in model_lower and family != 'default':
                return family
        
        return 'default'
    
    def get_model_profile(self, model_name):
        """Dapatkan profile optimasi untuk model tertentu"""
        family = self.detect_model_family(model_name)
        return self.model_profiles.get(family, self.model_profiles['default'])
    
    def optimize_prompt_for_model(self, text, config_id=None, model_name="default", max_tokens_override=None):
        """Optimasi prompt untuk model spesifik"""
        
        # Dapatkan model profile
        profile = self.get_model_profile(model_name)
        optimal_context = max_tokens_override or profile['optimal_context']
        max_prompt_tokens = int(optimal_context * 0.7)  # 70% untuk prompt, 30% untuk output
        
        print(f"🎯 Optimizing for {model_name} | Context: {optimal_context} | Family: {self.detect_model_family(model_name)}")
        
        # 1. Dapatkan base prompt dari database
        if config_id not in self.component_cache:
            self.component_cache[config_id] = self.load_prompt_components(config_id)
        
        components = self.component_cache[config_id]
        
        # 2. Bangun optimized prompt berdasarkan model family
        optimized_prompt = self.build_model_specific_prompt(
            components, text, model_name, max_prompt_tokens
        )
        
        estimated_tokens = len(optimized_prompt) // 4
        print(f"🔧 {model_name} Optimized: {len(optimized_prompt)} chars (~{estimated_tokens} tokens)")
        
        return optimized_prompt
    
    def build_model_specific_prompt(self, components, text, model_name, max_tokens):
        """Bangun prompt spesifik untuk model"""
        
        model_family = self.detect_model_family(model_name)
        
        # Template berdasarkan model family
        templates = {
            'qwen': self.build_chatml_prompt,
            'qwen2': self.build_chatml_prompt,
            'qwen3': self.build_chatml_prompt,
            'llama': self.build_llama_prompt,
            'llama2': self.build_llama_prompt, 
            'llama3': self.build_llama_prompt,
            'mistral': self.build_mistral_prompt,
            'mixtral': self.build_mistral_prompt,
            'komodo': self.build_mistral_prompt,
            'phi': self.build_chatml_prompt,
            'gemma': self.build_chatml_prompt,
            'default': self.build_chatml_prompt
        }
        
        builder = templates.get(model_family, self.build_chatml_prompt)
        prompt = builder(components, text, max_tokens)
        
        # Jika masih terlalu panjang, gunakan aggressive compression
        if len(prompt) > max_tokens:
            prompt = self.build_aggressive_prompt(text, model_family)
            
        return prompt
    
    def build_chatml_prompt(self, components, text, max_tokens):
        """ChatML format untuk Qwen, Phi, Gemma, dll"""
        prompt_parts = []
        
        # System message
        system_content = "Anda adalah sistem klasifikasi untuk keluhan mahasiswa. "
        if 'base' in components:
            system_content += self.compress_instructions(components['base'].content)
        prompt_parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")
        
        # Format instruction
        prompt_parts.append("<|im_start|>system\nFORMAT: Direktorat: [nama], Keyword: [kata_kunci]<|im_end|>")
        
        # Direktorat list
        if 'keywords' in components:
            direktorat_section = self.extract_direktorat_list(components['keywords'].content)
            prompt_parts.append(f"<|im_start|>system\n{direktorat_section}<|im_end|>")
        
        # Examples
        if 'examples' in components:
            best_examples = self.select_best_examples(components['examples'].content, count=2)
            prompt_parts.append(f"<|im_start|>system\nCONTOH:\n{best_examples}<|im_end|>")
        
        # User input
        prompt_parts.append(f"<|im_start|>user\nTeks: {text}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant")
        
        return '\n'.join(prompt_parts)
    
    def build_llama_prompt(self, components, text, max_tokens):
        """Llama format"""
        prompt_parts = ["<s>[INST] <<SYS>>"]
        
        # System instruction
        if 'base' in components:
            system_content = self.compress_instructions(components['base'].content)
            prompt_parts.append(system_content)
        
        prompt_parts.append("Format: Direktorat: [nama], Keyword: [kata_kunci]")
        
        # Direktorat list
        if 'keywords' in components:
            direktorat_section = self.extract_direktorat_list(components['keywords'].content)
            prompt_parts.append(direktorat_section)
        
        # Examples
        if 'examples' in components:
            best_examples = self.select_best_examples(components['examples'].content, count=2)
            prompt_parts.append(f"Contoh:\n{best_examples}")
        
        prompt_parts.append("<</SYS>>")
        
        # User input
        prompt_parts.append(f"Klasifikasi: {text} [/INST]")
        
        return '\n'.join(prompt_parts)
    
    def build_mistral_prompt(self, components, text, max_tokens):
        """Mistral format"""
        prompt_parts = ["[INST]"]
        
        # System instruction
        if 'base' in components:
            system_content = self.compress_instructions(components['base'].content)
            prompt_parts.append(system_content)
        
        prompt_parts.append("Format: Direktorat: [nama], Keyword: [kata_kunci]")
        
        # Direktorat list
        if 'keywords' in components:
            direktorat_section = self.extract_direktorat_list(components['keywords'].content)
            prompt_parts.append(direktorat_section)
        
        # Examples  
        if 'examples' in components:
            best_examples = self.select_best_examples(components['examples'].content, count=2)
            prompt_parts.append(f"Contoh:\n{best_examples}")
        
        # User input
        prompt_parts.append(f"Teks: {text} [/INST]")
        
        return '\n'.join(prompt_parts)
    
    def build_aggressive_prompt(self, text, model_family):
        """Prompt sangat minimal"""
        base_prompt = f"Klasifikasi: {text}\nFormat: Direktorat: [nama], Keyword: [kata_kunci]"
        
        if model_family in ['qwen', 'qwen2', 'qwen3']:
            return f"<|im_start|>user\n{base_prompt}<|im_end|>\n<|im_start|>assistant"
        elif model_family in ['llama', 'llama2', 'llama3']:
            return f"<s>[INST] {base_prompt} [/INST]"
        elif model_family in ['mistral', 'mixtral', 'komodo']:
            return f"[INST] {base_prompt} [/INST]"
        else:
            return base_prompt

    def load_prompt_components(self, config_id):
        """Load komponen prompt dari database"""
        components = prompt_manager.get_config_components(config_id)
        return {comp.component_type: comp for comp in components if comp.is_enabled}
    
    def compress_instructions(self, instructions, max_lines=5):
        """Kompres instructions menjadi esensial saja"""
        lines = instructions.split('\n')
        essential_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(keyword in line.lower() for keyword in ['klasifikasi', 'format', 'output', 'direktorat']):
                essential_lines.append(line)
            elif len(essential_lines) < max_lines:
                essential_lines.append(line)
        
        return '\n'.join(essential_lines)
    
    def extract_direktorat_list(self, keywords_content):
        """Extract list direktorat yang simpel"""
        lines = keywords_content.split('\n')
        direktorat_lines = []
        
        for line in lines:
            if ':' in line and any(str(i) in line for i in range(1, 8)):
                # Format: "1. Akademik: krs, nilai, ipk"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    direktorat = parts[0].strip()
                    # Hanya ambil 2-3 keyword utama
                    main_keywords = ', '.join(parts[1].strip().split(',')[:3])
                    direktorat_lines.append(f"{direktorat}: {main_keywords}")
        
        return "DIREKTORAT:\n" + '\n'.join(direktorat_lines[:7])  # Maks 7 direktorat
    
    def select_best_examples(self, examples_content, count=3):
        """Pilih contoh terbaik yang paling representatif"""
        examples = examples_content.split('\n\n')
        selected = []
        
        # Prioritaskan contoh dengan format yang konsisten
        for example in examples:
            if '→' in example and 'Direktorat:' in example:
                selected.append(example)
                if len(selected) >= count:
                    break
        
        return '\n\n'.join(selected)

class UniversalMemoryManager:
    def __init__(self, llm_instance):
        self.llm = llm_instance
        self.processed_count = 0
        self.reset_interval = 100  # Reset setiap 100 teks
    
    def process_with_memory_management(self, prompt):
        """Process dengan memory management"""
        output = self.llm.invoke(prompt)
        
        self.processed_count += 1
        
        # Reset context setiap interval tertentu untuk hindari memory leak
        if self.processed_count % self.reset_interval == 0:
            self.llm.reset()
            print("🔄 LLM context reset untuk memory management")
        
        return output

# ==================== MAIN CLASSIFICATION SYSTEM ====================

class ClassificationSystem:
    def __init__(self):
        self.optimizer = UniversalPromptOptimizer()
        self.memory_manager = None

    def get_fast_mode_hint(self, processed_text):
        """Return a lightweight heuristic result only for fast mode."""
        text = (processed_text or '').lower()
        if not text:
            return None

        for direktorat, keywords in FAST_MODE_RULES:
            for keyword in keywords:
                if keyword_present_in_text(text, keyword):
                    return direktorat, keyword

        return None

    def get_strong_keyword_directorat(self, processed_text):
        """Return a strong deterministic direktorat hint from explicit keywords."""
        text = (processed_text or '').lower()
        if not text:
            return None

        strong_rules = [
            ('Pusat Teknologi Informasi', ['aplikasi', 'login', 'akun', 'email', 'igracias', 'lms', 'sso', 'wifi', 'sistem', 'igrcis']),
            ('Aset & Sustainability', ['parkir', 'parkiran', 'gedung', 'fasilitas', 'spanduk', 'tult', 'kampus fisik', 'aset', 'sustainability']),
            ('Keuangan', ['pembayaran', 'bpp', 'ukt', 'biaya', 'tagihan', 'uang kuliah', 'bayar', 'semesteran']),
            ('Kemahasiswaan, Pengembangan Karir, Alumni', ['magang', 'kp', 'loker', 'karir', 'alumni', 'pembinaan', 'beasiswa', 'organisasi mahasiswa']),
            ('Pasca Sarjana dan Advance Learning', ['pendaftaran s2', 'pendaftaran s3', 's2', 's3', 'tpa', 'program pasca', 'beasiswa pasca', 'pasca sarjana', 'advance learning']),
            ('Akademik', ['krs', 'sks', 'nilai', 'sidang', 'yudisium', 'tugas akhir', 'ta', 'registrasi akademik', 'libur akademik', 'transkrip', 'perwalian']),
            ('Sumber Daya Manusia', ['dosen', 'edom', 'bimbingan', 'wal dosen', 'sdm', 'kepegawaian', 'pegawai', 'staff'])
        ]

        for direktorat, keywords in strong_rules:
            for keyword in keywords:
                if keyword_present_in_text(text, keyword):
                    return direktorat, keyword

        return None

    def infer_direktorat_from_keyword(self, keyword_text):
        """Infer direktorate from a predicted keyword string."""
        if not keyword_text or keyword_text.strip().lower() in ['-', 'none', 'null', 'nan']:
            return None

        keyword_text = keyword_text.lower()
        keyword_map = {
            'aplikasi': 'Pusat Teknologi Informasi',
            'login': 'Pusat Teknologi Informasi',
            'akun': 'Pusat Teknologi Informasi',
            'lms': 'Pusat Teknologi Informasi',
            'sso': 'Pusat Teknologi Informasi',
            'wifi': 'Pusat Teknologi Informasi',
            'bpp': 'Keuangan',
            'ukt': 'Keuangan',
            'biaya': 'Keuangan',
            'bayar': 'Keuangan',
            'tagihan': 'Keuangan',
            'pembayaran': 'Keuangan',
            'magang': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            'kp': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            'loker': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            'alumni': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            'beasiswa': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            'karir': 'Kemahasiswaan, Pengembangan Karir, Alumni',
            's2': 'Pasca Sarjana dan Advance Learning',
            's3': 'Pasca Sarjana dan Advance Learning',
            'tpa': 'Pasca Sarjana dan Advance Learning',
            'krs': 'Akademik',
            'sks': 'Akademik',
            'nilai': 'Akademik',
            'sidang': 'Akademik',
            'yudisium': 'Akademik',
            'tugas akhir': 'Akademik',
            'transkrip': 'Akademik',
            'perwalian': 'Akademik',
            'dosen': 'Sumber Daya Manusia',
            'edom': 'Sumber Daya Manusia',
            'sdm': 'Sumber Daya Manusia',
            'bimbingan': 'Sumber Daya Manusia',
            'pegawai': 'Sumber Daya Manusia'
        }

        for alias, direktorat in keyword_map.items():
            if alias in keyword_text:
                return direktorat, alias

        return None

    def validate_prediction_with_text(self, direktorat, keyword, processed_text):
        """Validate or correct LLM prediction using explicit keyword signals from the text."""
        if not processed_text:
            return direktorat, keyword

        strong_hint = self.get_strong_keyword_directorat(processed_text)

        if direktorat == 'Uncategorized' and strong_hint:
            self.log_message(
                'LLM output was Uncategorized; applying strong keyword heuristic override.',
                'INFO'
            )
            return strong_hint

        inferred_from_keyword = self.infer_direktorat_from_keyword(keyword) if keyword else None
        if direktorat == 'Uncategorized' and inferred_from_keyword:
            self.log_message(
                'LLM output was Uncategorized; inferring direktorat from keyword.',
                'INFO'
            )
            return inferred_from_keyword

        if strong_hint and strong_hint[0] != direktorat:
            if not keyword or keyword.strip().lower() in ['-', 'none', 'null', 'nan']:
                self.log_message(
                    'LLM prediction keyword invalid; overriding prediction with strong keyword heuristic.',
                    'INFO'
                )
                return strong_hint
            if not keyword_present_in_text(processed_text, keyword):
                self.log_message(
                    'LLM predicted keyword not found in text; overriding with strong heuristic.',
                    'INFO'
                )
                return strong_hint

        return direktorat, keyword

    def get_stop_sequences(self, model_name):
        """Stop generation before model starts echoing examples or next chat turns."""
        base_stops = [
            "\n---",
            "\nJAWABAN:",
            "\nCONTOH",
        ]

        lowered = (model_name or '').lower()
        if 'qwen' in lowered:
            # Enforce one-line response for Qwen-based ChatML prompts.
            return base_stops + ["<|im_end|>", "\n"]
        if 'llama' in lowered:
            return base_stops + ["<|eot_id|>"]
        return base_stops

    def build_compact_retry_prompt(self, text):
        """Compact retry prompt used when first generation is empty."""
        return (
            "Anda adalah classifier direktorat Telkom University. "
            "Pilih SATU direktorat paling relevan dari: Akademik; Pasca Sarjana dan Advance Learning; "
            "Aset & Sustainability; Keuangan; Sumber Daya Manusia; Pusat Teknologi Informasi; "
            "Kemahasiswaan, Pengembangan Karir, Alumni.\n"
            "Jawab tepat satu baris dan jangan menambahkan teks lain.\n"
            "Format wajib: Direktorat: [nama_direktorat], Keyword: [keyword_terkait]\n"
            f"Teks: {text}"
        )

    def build_strict_answer_suffix(self):
        """Strict single-line suffix appended to the main prompt."""
        return (
            "\n\nJAWABAN AKHIR WAJIB SATU BARIS SAJA. "
            "JANGAN menulis ulang 'Teks:'. "
            "JANGAN beri penjelasan. "
            "Format tepat: Direktorat: [nama_direktorat], Keyword: [keyword_terkait]"
        )

    def build_fallback_prompt(self, text):
        """Fallback prompt saat DB/app context prompt manager tidak tersedia."""
        return (
            "Anda adalah sistem klasifikasi direktorat Telkom University. "
            "Klasifikasikan teks berikut ke SATU direktorat paling relevan dari daftar ini: "
            "Akademik; Pasca Sarjana dan Advance Learning; Aset & Sustainability; Keuangan; "
            "Sumber Daya Manusia; Pusat Teknologi Informasi; Kemahasiswaan, Pengembangan Karir, Alumni.\n\n"
            "Output WAJIB satu baris dengan format tepat: "
            "Direktorat: [nama_direktorat], Keyword: [keyword_terkait].\n\n"
            f"Teks: {text}"
        )

    def insert_strict_suffix(self, prompt):
        """Insert strict answer suffix inside the final assistant/instruction block."""
        suffix = self.build_strict_answer_suffix()
        prompt = prompt.rstrip()

        if prompt.endswith("<|im_end|>"):
            return prompt[:-len("<|im_end|>")] + suffix + "<|im_end|>"
        if prompt.endswith("[/INST]"):
            return prompt[:-len("[/INST]")] + suffix + " [/INST]"
        if prompt.endswith("</s>"):
            return prompt[:-len("</s>")] + suffix + "</s>"

        return prompt + suffix

    def estimate_token_count(self, text):
        """Estimate token count from prompt length."""
        return max(1, len(text) // 4)

    def get_model_context_limit(self, model_name):
        if model_name:
            profile = self.optimizer.get_model_profile(model_name)
            return profile.get('optimal_context', 4096)
        return 4096

    def log_message(self, msg, level="INFO", socketio=None):
        """Centralized logging function"""
        full_msg = f"[{level}] {msg}"
        print(full_msg)
        if socketio:
            try:
                socketio.emit('progress_update', {
                    'status': msg,
                    'level': level
                }, namespace='/')
            except Exception as e:
                print(f"Socket emit error: {e}")

    def validate_file(self, file_path):
        """Validate input file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not file_path.endswith(('.csv', '.xlsx')):
            raise ValueError("Only CSV and XLSX files are supported")
            
        return True

    def read_dataframe(self, file_path):
        """Read dataframe with multiple fallback strategies"""
        if file_path.endswith('.csv'):
            return self._read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, engine='openpyxl')
    
    def _read_csv(self, file_path):
        """Read CSV with multiple attempts"""
        attempts = [
            {"name": "Default", "kwargs": {}},
            {"name": "Python engine", "kwargs": {"engine": "python", "encoding": "utf-8"}},
            {"name": "UTF-8-sig", "kwargs": {"engine": "python", "encoding": "utf-8-sig"}},
            {"name": "Permissive", "kwargs": {
                "engine": "python", "encoding": "utf-8-sig",
                "on_bad_lines": "skip", "sep": None
            }}
        ]
        
        for attempt in attempts:
            try:
                df = pd.read_csv(file_path, **attempt['kwargs'])
                self.log_message(f"CSV read successful with {attempt['name']}")
                return df
            except Exception as e:
                self.log_message(f"Attempt {attempt['name']} failed: {e}", "WARNING")
                
        raise RuntimeError("All CSV reading attempts failed")

    def find_text_column(self, df):
        """Find appropriate text column in dataframe"""
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        if text_columns:
            return text_columns[0]
        elif 'Text' in df.columns:
            return 'Text'
        else:
            raise ValueError(f"No text column found. Available: {df.columns.tolist()}")

    def load_checkpoint(self, checkpoint_path, total_rows):
        """Load existing checkpoint if available"""
        if not os.path.exists(checkpoint_path):
            return [None] * total_rows, 0
            
        try:
            checkpoint_df = pd.read_excel(checkpoint_path)
            results = [None] * total_rows
            
            for idx, row in checkpoint_df.iterrows():
                if idx >= total_rows:
                    continue

                raw_direktorat = row.get('Direktorat')
                raw_keyword = row.get('Keyword')

                if pd.isna(raw_direktorat) or pd.isna(raw_keyword):
                    continue

                normalized_direktorat = normalize_direktorat_name(str(raw_direktorat).strip())
                if normalized_direktorat == "Uncategorized":
                    # Abaikan label checkpoint yang tidak valid (contoh: "Error").
                    continue

                results[idx] = (normalized_direktorat, str(raw_keyword).strip())
                    
            start_idx = next((i for i, r in enumerate(results) if r is None), 0)
            return results, start_idx
            
        except Exception as e:
            self.log_message(f"Checkpoint load failed: {e}", "WARNING")
            return [None] * total_rows, 0

    def save_checkpoint(self, df, results, checkpoint_path):
        """Save progress checkpoint"""
        try:
            checkpoint_df = df.copy()
            checkpoint_df['Direktorat'] = [r[0] if r else None for r in results]
            checkpoint_df['Keyword'] = [r[1] if r else None for r in results]
            checkpoint_df.to_excel(checkpoint_path, index=False)
            return True
        except Exception as e:
            self.log_message(f"Checkpoint save failed: {e}", "WARNING")
            return False

    def process_single_text(self, text, llm, config_id, classification_mode='pure_llm', model_name=None):
        """Process dengan parser yang diperbaiki"""
        if not text or not text.strip():
            return "Uncategorized", "-"

        if classification_mode == 'fast_mode':
            fast_result = self.get_fast_mode_hint(text)
            if fast_result:
                self.log_message(
                    f"Fast mode heuristic match: {fast_result[0]} ({fast_result[1]})",
                    "INFO"
                )
                return fast_result
            
        try:
            try:
                prompt = prompt_manager.build_prompt(text, config_id)
                print(f"🎯 Using YOUR custom prompt (config_id: {config_id})")
            except RuntimeError as e:
                if 'Working outside of application context' in str(e):
                    # Jalankan fallback supaya proses tidak menghasilkan label Error massal.
                    self.log_message(
                        "Prompt manager app context tidak tersedia, menggunakan fallback prompt.",
                        "WARNING"
                    )
                    prompt = self.build_fallback_prompt(text)
                else:
                    raise

            # Penutup tegas agar model output konsisten dari first-pass.
            prompt = self.insert_strict_suffix(prompt)

            model_context = self.get_model_context_limit(model_name)
            max_prompt_tokens = max(64, model_context - CLASSIFY_MAX_OUTPUT_TOKENS)
            prompt_tokens = self.estimate_token_count(prompt)

            if prompt_tokens > max_prompt_tokens:
                self.log_message(
                    f"Prompt estimated at {prompt_tokens} tokens exceeds model budget {max_prompt_tokens}. "
                    f"Optimizing prompt for model '{model_name or 'default'}'.",
                    "WARNING"
                )
                prompt = self.optimizer.optimize_prompt_for_model(
                    text,
                    config_id,
                    model_name or 'default',
                    max_tokens_override=max_prompt_tokens,
                )
                prompt += self.build_strict_answer_suffix()
                prompt_tokens = self.estimate_token_count(prompt)

                if prompt_tokens > max_prompt_tokens:
                    self.log_message(
                        f"Optimized prompt still too large ({prompt_tokens} tokens). "
                        "Switching to compact retry prompt.",
                        "WARNING"
                    )
                    prompt = self.build_compact_retry_prompt(text)

            print(f"📝 Text: {text[:80]}...")

            invoke_kwargs = {
                'max_tokens': CLASSIFY_MAX_OUTPUT_TOKENS,
                'temperature': CLASSIFY_TEMPERATURE,
            }
            if CLASSIFY_USE_STOP_SEQUENCES:
                invoke_kwargs['stop'] = self.get_stop_sequences(model_name)

            try:
                output = llm.invoke(prompt, **invoke_kwargs)
            except Exception as e:
                if 'exceed context window' in str(e).lower() or 'requested tokens' in str(e).lower():
                    self.log_message(
                        "LLM rejected prompt due context window. Retrying with compact fallback prompt.",
                        "WARNING"
                    )
                    prompt = self.build_compact_retry_prompt(text)
                    output = llm.invoke(prompt, **invoke_kwargs)
                else:
                    raise

            if not str(output).strip():
                strategy = CLASSIFY_EMPTY_RETRY_STRATEGY
                self.log_message(
                    f"LLM produced empty output. retry_strategy={strategy}",
                    "WARNING"
                )

                if strategy == 'db':
                    # Retry with the same DB/fallback prompt but without stop sequences.
                    retry_kwargs = {
                        'max_tokens': max(CLASSIFY_MAX_OUTPUT_TOKENS, 96),
                        'temperature': 0.0,
                    }
                    output = llm.invoke(prompt, **retry_kwargs)
                elif strategy == 'compact':
                    retry_prompt = self.build_compact_retry_prompt(text)
                    retry_kwargs = {
                        'max_tokens': max(CLASSIFY_MAX_OUTPUT_TOKENS, 96),
                        'temperature': 0.0,
                    }
                    output = llm.invoke(retry_prompt, **retry_kwargs)
                elif strategy == 'off':
                    output = ""
                else:
                    retry_kwargs = {
                        'max_tokens': max(CLASSIFY_MAX_OUTPUT_TOKENS, 96),
                        'temperature': 0.0,
                    }
                    output = llm.invoke(prompt, **retry_kwargs)

            output = str(output).strip()
            print(f"📨 LLM Output: {output}")
            
            # ✅ P0 FIX: Pass original text for keyword validation
            direktorat, keyword = parse_llm_output(output, text)
            direktorat, keyword = self.validate_prediction_with_text(direktorat, keyword, text)
            return direktorat, keyword
            
        except Exception as e:
            self.log_message(f"Text processing error: {e}", "ERROR")
            # Gunakan fallback label aman agar output tetap bisa dipakai.
            return "Uncategorized", "-"

    def calculate_stats(self, start_time, processed_count, total_rows):
        """Calculate processing statistics"""
        elapsed = time.time() - start_time
        texts_per_second = processed_count / elapsed if elapsed > 0 else 0
        remaining = (total_rows - processed_count) / texts_per_second if texts_per_second > 0 else 0
        
        return {
            'processed': processed_count,
            'total': total_rows,
            'speed': f'{texts_per_second:.2f} texts/sec',
            'estimated_remaining': f'{remaining/60:.1f} minutes'
        }

    def classify_universal(self, file_path, socketio=None, config_id=None, 
                          model_name=None, ctx_size_override=None, 
                          benchmark_mode=False, strict_keyword_match=None,
                          classification_mode=CLASSIFY_DEFAULT_MODE):
        """Main universal classification function"""

        if classification_mode not in ['pure_llm', 'fast_mode']:
            classification_mode = CLASSIFY_DEFAULT_MODE

        if strict_keyword_match is None:
            # Default: always validate keyword against processed text for all modes.
            strict_keyword_match = CLASSIFY_STRICT_KEYWORD_DEFAULT
        
        # Initialization
        self.log_message(
            f"Starting classification: {file_path} | mode={classification_mode}",
            socketio=socketio
        )
        start_time = time.time()
        
        # Validate and read data
        self.validate_file(file_path)
        df = self.read_dataframe(file_path)
        text_column = self.find_text_column(df)
        total_rows = len(df)
        
        # Preprocessing
        self.log_message("Preprocessing texts...", socketio=socketio)
        preprocess_update_every = max(1, min(200, total_rows // 20 if total_rows > 20 else 1))
        processed_texts = []
        for i, raw_text in enumerate(df[text_column].tolist()):
            processed_texts.append(preprocess_text(raw_text))

            if socketio and ((i + 1) % preprocess_update_every == 0 or i == total_rows - 1):
                socketio.emit('progress_update', {
                    'status': f'Preprocessing {i + 1}/{total_rows}',
                    'percent': int(((i + 1) / total_rows) * 20),
                    'stage': 'Preprocessing'
                }, namespace='/')

        df['processed_text'] = processed_texts
        
        # Initialize LLM
        llm = init_llm_universal(model_name, ctx_size_override)
        self.memory_manager = UniversalMemoryManager(llm)
        
        # Checkpoint setup
        checkpoint_path = f"{os.path.splitext(file_path)[0]}_checkpoint.xlsx"
        results, start_idx = self.load_checkpoint(checkpoint_path, len(df))

        progress_update_every = max(1, min(CLASSIFY_PROGRESS_UPDATE_EVERY, total_rows // 20 if total_rows > 20 else 1))
        
        # ✅ PERBAIKAN: Gunakan safe_tqdm yang sudah didefinisikan di atas
        for idx, row in tqdm(list(df.iterrows())[start_idx:], 
                           initial=start_idx, total=total_rows,
                           desc="Classifying"):
            
            text = row['processed_text']
            direktorat, keyword = self.process_single_text(
                text,
                llm,
                config_id,
                classification_mode=classification_mode,
                model_name=model_name,
            )
            results[idx] = (direktorat, keyword)
            
            # Progress reporting
            if socketio and (idx % progress_update_every == 0 or idx == total_rows - 1):
                stats = self.calculate_stats(start_time, idx + 1, total_rows)
                socketio.emit('progress_update', {
                    'status': f'Processing {idx + 1}/{total_rows}',
                    'percent': 20 + int((idx + 1) / total_rows * 80),
                    'stats': stats
                }, namespace='/')
            
            # Checkpoint saving
            if idx % CLASSIFY_CHECKPOINT_EVERY == 0:
                self.save_checkpoint(df, results, checkpoint_path)
        
        # Final processing
        df['Direktorat'] = [r[0] if r else "Uncategorized" for r in results]
        df['Keyword'] = [r[1] if r else "-" for r in results]
        
        # Apply strict keyword matching if enabled
        if strict_keyword_match:
            df = self.apply_strict_keyword_matching(df)
        
        # Save results
        output_path = f"{os.path.splitext(file_path)[0]}_classified.xlsx"
        df.to_excel(output_path, index=False)
        
        self.log_message(f"Classification completed: {output_path}", socketio=socketio)
        return output_path

    def apply_strict_keyword_matching(self, df):
        """Apply strict keyword validation"""
        final_keywords = []
        for _, row in df.iterrows():
            processed = row.get('processed_text', '')
            raw_kw = row.get('Keyword', '-')
            
            if not raw_kw or str(raw_kw).strip() in ['-', 'None', 'none', 'NULL']:
                final_keywords.append('-')
                continue
                
            candidates = normalize_candidate_keywords(str(raw_kw))
            chosen = None
            
            for cand in candidates:
                if keyword_present_in_text(processed, cand):
                    chosen = cand
                    break
                    
            final_keywords.append(chosen or '-')
            
        df['Keyword'] = final_keywords
        return df

# ==================== UTILITY FUNCTIONS ====================

def preprocess_text(text):
    """
    ✅ IMPROVED: Gunakan centralized abbreviation expansions dari suggested_improvements
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    # Remove URLs but preserve important keywords
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Convert to lowercase but preserve common acronyms
    common_acronyms = ['sks', 'krs', 'bpp', 'ukt', 'khs', 'sp2', 'sp3', 'ssw', 'sso', 'lms', 'ktm', 'msib', 'kp','puti','kipk','feb','fkb','ksm','pkkmb','cbt','skpi','eprt', 'sirama', 'tult', 'prs','bap', 'lak', 'mbkm', 'fri', 'mbti', 'nik', 'sk', 'fik', 'ftif', 'toss']
    text_lower = text.lower()
    for acronym in common_acronyms:
        if acronym in text_lower:
            text_lower = text_lower.replace(acronym, acronym.upper())
    text = text_lower
    
    # Use centralized abbreviation expansions jika available
    if IMPROVEMENTS_AVAILABLE:
        replacements = ABBREVIATION_EXPANSIONS
    else:
        # Fallback ke old hardcoded replacements
        replacements = {
            'igracias': 'igracias',
            'i-gracias': 'igracias',
            'tel-u': 'telyu',
            'telu': 'telyu',
            'tel u': 'telyu',
            'skrg': 'sekarang',
            'semester': 'semester',
            'smstr': 'semester',
            'sem': 'semester',
            'ttp': 'tetap',
            'tp': 'tapi',
            'twt': 'twitter',
            'kk': 'kakak',
            'pen': 'pengen',
            'abg': 'abang',
            'ga': 'tidak',
            'gt': 'gitu',
            'ak': 'aku',
            'trs': 'terus',
            'krn': 'karena',
            'kpn': 'kapan',
            'sm': 'sama',
            'gmn': 'gimana',
            'dgn': 'dengan',
            'sy': 'saya',
            'dlm': 'dalam',
            'org': 'orang',
            'bg': 'bang',
            'diacc': 'di accept',
            'bukber': 'buka bersama',
            'bsk': 'besok',
            'bs': 'bisa',
            'bgt': 'banget',
            'pls': 'please',
            'tbtb': 'tiba tiba',
            'jgn': 'jangan',
            'omg': 'oh my god',
            'ta': 'tugas akhir',
            'gw': 'saya',
            'hp': 'handphone',
            'wkt': 'waktu',
            'sertif': 'sertifikat',
            'gasi': 'nggak sih',
            'yudis': 'yudisium',
            'proker': 'program kerja',
            'matkul': 'mata kuliah',
            'sk': 'surat keterangan',
            'mk': 'mata kuliah',
            'waldos': 'wali dosen',
            'dospen': 'dosen pembimbing',
            'fri': 'fakultas rekaraya industri',
            'feb': 'fakultas ekonomi bisnis',
            'fik': 'fakultas ilmu komputer',
            'fkb': 'fakultas komunikasi bisnis',
            'nik': 'nomor induk kependudukan'
        }
    
    for old, new in replacements.items():
        text = re.sub(r'\b' + re.escape(old) + r'\b', new, text)
    
    # Clean punctuation but preserve sentence structure
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text

def parse_llm_output(output, original_text=None):
    """
    ✅ IMPROVED Parser - menggunakan improved version dengan:
    - Pre-compiled regex patterns (5x faster!)
    - Keyword validation (P0 fix)
    - Confidence scoring (P1 improvement)
    - Fallback ke improved version jika available, else gunakan old version
    """
    
    # Gunakan improved parser jika available
    if IMPROVEMENTS_AVAILABLE:
        result = parse_llm_output_improved(output, original_text, verbose=CLASSIFY_VERBOSE_PARSING)
        # Return tuple untuk backward compatibility
        return result['direktorat'], result['keyword']
    
    # Fallback ke old parser (backward compatible)
    if CLASSIFY_VERBOSE_PARSING:
        print("=" * 60)
        print("🔍 PARSING LLM OUTPUT (FALLBACK VERSION)")
        print("=" * 60)
        print(f"RAW OUTPUT:\n{output}")
        print("=" * 60)
    
    # Use centralized VALID_DIREKTORATS jika available
    valid_direktorats = VALID_DIREKTORATS if IMPROVEMENTS_AVAILABLE else [
        "Akademik",
        "Pasca Sarjana dan Advance Learning", 
        "Aset & Sustainability",
        "Keuangan",
        "Sumber Daya Manusia",
        "Pusat Teknologi Informasi",
        "Kemahasiswaan, Pengembangan Karir, Alumni"
    ]
    
    # Use pre-compiled patterns jika available
    if IMPROVEMENTS_AVAILABLE and PARSING_PATTERNS_COMPILED:
        direktorat = "Uncategorized"
        keyword = "-"
        
        for pattern_name, regex in PARSING_PATTERNS_COMPILED.items():
            matches = regex.findall(output)
            if matches:
                if CLASSIFY_VERBOSE_PARSING:
                    print(f"✅ Found {len(matches)} matches with pattern: {pattern_name}")
                
                for raw_direktorat, raw_keyword in matches:
                    raw_direktorat = raw_direktorat.strip()
                    raw_keyword = raw_keyword.strip()
                    
                    if re.match(r'\[.*\]', raw_direktorat) or re.match(r'\[.*\]', raw_keyword):
                        continue
                    
                    if is_generic_pattern(raw_direktorat) if IMPROVEMENTS_AVAILABLE else (
                        len(raw_direktorat) < 3 or raw_direktorat.lower() in ['nama', 'direktorat', 'error', 'invalid']
                    ):
                        continue
                    
                    normalized = normalize_direktorat_name(raw_direktorat)
                    if normalized != "Uncategorized":
                        direktorat = normalized
                        keyword = raw_keyword
                        if CLASSIFY_VERBOSE_PARSING:
                            print(f"🎯 Selected: '{direktorat}' with keyword: '{keyword}'")
                        break
                
                if direktorat != "Uncategorized":
                    break
    else:
        # Old pattern matching code (fallback)
        patterns = [
            r'JAWABAN:.*?→\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'jawaban:.*?→\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'Jawaban:.*?→\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'JAWABAN:\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'jawaban:\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'Jawaban:\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'→\s*Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
            r'Direktorat:\s*([^,\n]+?)\s*(?:,|\n)\s*Keyword:\s*([^\n]+)',
        ]
        
        direktorat = "Uncategorized"
        keyword = "-"
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                for match in matches:
                    raw_direktorat = match[0].strip()
                    current_keyword = match[1].strip()
                    
                    if re.match(r'\[.*\]', raw_direktorat) or re.match(r'\[.*\]', current_keyword):
                        continue
                    
                    if len(raw_direktorat) < 3 or raw_direktorat.lower() in [
                        'nama', 'direktorat', 'error', 'invalid', 'unknown', 'uncategorized'
                    ]:
                        continue
                    
                    normalized = normalize_direktorat_name(raw_direktorat)
                    if normalized != "Uncategorized":
                        direktorat = normalized
                        keyword = current_keyword
                        break
                
                if direktorat != "Uncategorized":
                    break
    
    # Fallback: cari direktorat mention
    if direktorat == "Uncategorized":
        if IMPROVEMENTS_AVAILABLE:
            direktorat, keyword, _ = extract_direktorat_by_mention(output, valid_direktorats)
        else:
            for valid_dir in valid_direktorats:
                if valid_dir.lower() in output.lower():
                    direktorat = valid_dir
                    keyword = extract_keyword_near_mention(output, valid_dir)
                    break
    
    # P0 FIX: Validate keyword terhadap original text jika available
    if original_text and IMPROVEMENTS_AVAILABLE:
        keyword = validate_keyword_against_text(keyword, original_text)
    
    keyword = clean_keyword(keyword)
    
    if CLASSIFY_VERBOSE_PARSING:
        print(f"🎯 FINAL RESULT: Direktorat='{direktorat}', Keyword='{keyword}'")
        print("=" * 60)
    
    return direktorat, keyword

def extract_keyword_near_mention(output, direktorat_mentioned):
    """Extract keyword yang dekat dengan mention direktorat"""
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if direktorat_mentioned.lower() in line.lower():
            # Cari line berikutnya yang mungkin mengandung keyword
            for j in range(i, min(i+3, len(lines))):
                if 'keyword' in lines[j].lower() or 'kata kunci' in lines[j].lower():
                    # Extract keyword setelah "Keyword:"
                    keyword_match = re.search(r'Keyword:\s*([^\n,]+)', lines[j], re.IGNORECASE)
                    if keyword_match:
                        return keyword_match.group(1).strip()
    
    return "-"

def normalize_direktorat_name(raw_direktorat):
    """
    ✅ OPTIMIZED: Normalize nama direktorat ke format standar
    Menggunakan centralized mapping dari suggested_improvements
    """
    
    # Gunakan optimized version jika available
    if IMPROVEMENTS_AVAILABLE:
        return normalize_direktorat_name_optimized(raw_direktorat)
    
    # Fallback ke old implementation
    # Mapping untuk variasi penulisan yang mungkin dihasilkan LLM
    normalization_map = {
        'akademik': 'Akademik',
        'pasca sarjana': 'Pasca Sarjana dan Advance Learning',
        'pasca': 'Pasca Sarjana dan Advance Learning',
        'pasca sarjana dan advance learning': 'Pasca Sarjana dan Advance Learning',
        'pascasarjana': 'Pasca Sarjana dan Advance Learning',
        'aset': 'Aset & Sustainability',
        'asset': 'Aset & Sustainability',
        'aset sustainability': 'Aset & Sustainability',
        'aset & sustainability': 'Aset & Sustainability',
        'keuangan': 'Keuangan',
        'finance': 'Keuangan',
        'sdm': 'Sumber Daya Manusia',
        'sumber daya manusia': 'Sumber Daya Manusia',
        'pti': 'Pusat Teknologi Informasi',
        'puti': 'Pusat Teknologi Informasi',
        'teknologi informasi': 'Pusat Teknologi Informasi',
        'pusat teknologi informasi': 'Pusat Teknologi Informasi',
        'teknologi': 'Pusat Teknologi Informasi',
        'kemahasiswaan': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'kemahasiswaan pengembangan karir alumni': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'kemahasiswaan, pengembangan karir, alumni': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'kemahasiswaan dan pengembangan karir': 'Kemahasiswaan, Pengembangan Karir, Alumni',
    }
    
    normalized = raw_direktorat.lower().strip()
    
    # Cek exact match di mapping
    if normalized in normalization_map:
        return normalization_map[normalized]
    
    # Cek partial match
    for variant, standard in normalization_map.items():
        if variant in normalized or normalized in variant:
            return standard
    
    # Unknown labels (mis. "Error") harus dianggap tidak valid.
    return "Uncategorized"

def clean_keyword(keyword):
    """✅ Bersihkan keyword dari karakter yang tidak diinginkan"""
    # Gunakan improved version jika available
    if IMPROVEMENTS_AVAILABLE:
        from suggested_improvements import clean_keyword as clean_keyword_improved
        return clean_keyword_improved(keyword)
    
    # Fallback ke old implementation
    if not keyword or keyword.lower() in ['-', 'none', 'null', 'nan']:
        return "-"
    
    # Remove punctuation dan extra spaces
    keyword = re.sub(r'[^\w\s-]', '', keyword)
    keyword = ' '.join(keyword.split())
    
    return keyword

def get_valid_keywords():
    keywords = set()
    try:
        # Try to parse the keywords component content
        keyword_comp = None
        try:
            # try to get existing active config components
            active = prompt_manager.get_active_config()
            if active:
                comps = prompt_manager.get_config_components(active.id)
                for c in comps:
                    if c.component_type == 'keywords':
                        keyword_comp = c.content
                        break
        except Exception:
            # fallback to default_components dictionary if present
            if hasattr(prompt_manager, 'default_components') and 'keywords' in prompt_manager.default_components:
                keyword_comp = prompt_manager.default_components['keywords']['default_content']
        
        if keyword_comp:
            # Find lines after ":" that contain keywords
            parts = re.split(r'\n|\r', keyword_comp)
            for line in parts:
                if ':' in line:
                    # take substring after ':'
                    _, after = line.split(':', 1)
                    # split by commas
                    pieces = re.split(r',|;', after)
                    for p in pieces:
                        w = p.strip()
                        # filter short empty tokens
                        if len(w) >= 2:
                            # remove words like "Keywords" if present
                            w = re.sub(r'Keywords|Primary Keywords|Secondary Keywords|Direktorat', '', w, flags=re.IGNORECASE).strip()
                            if w:
                                # some tokens are phrases; include them lowercased
                                keywords.add(w.lower())
        # fallback manual list if still empty
        if not keywords:
            manual = [
                'krs','nilai','transkrip','perkuliahan','mata kuliah','sks','cuti akademik','rps','perwalian',
                'ipk','jadwal','semester','s2','s3','wisuda','wifi','password','parkiran','parkir','gedung',
                'bpp','biaya','bayar','dosen','mahasiswa','staff','lms','puti','igracias','ktm',
                'magang','msib','kp','puti','igracias','tugas akhir','registrasi akademik'
            ]
            keywords.update([k.lower() for k in manual])
    except Exception as e:
        # worst-case fallback
        keywords = {
            'krs','nilai','transkrip','perkuliahan','sks','cuti akademik','rps','perwalian',
            'ipk','jadwal','semester','s2','s3','wisuda','wifi','password','parkiran','parkir','gedung',
            'bpp','biaya','bayar','dosen','mahasiswa','staff','lms','puti','igracias','ktm',
            'magang','msib','kp'
        }
    return set(k.lower() for k in keywords)

VALID_KEYWORDS = get_valid_keywords()

def normalize_candidate_keywords(keyword_str):
    """✅ Split keyword string returned by LLM into candidate keywords."""
    # Gunakan improved version jika available
    if IMPROVEMENTS_AVAILABLE:
        from suggested_improvements import normalize_candidate_keywords as normalize_candidate_keywords_improved
        return normalize_candidate_keywords_improved(keyword_str)
    
    # Fallback ke old implementation
    if not keyword_str or keyword_str.strip() in ['-', 'none', 'None', 'NULL']:
        return []
    # Remove surrounding quotes
    kw = keyword_str.strip().strip('"').strip("'")
    # split
    parts = re.split(r',|;|/|\||\band\b', kw)
    candidates = [p.strip() for p in parts if p and p.strip() != '']
    return candidates

def keyword_present_in_text(processed_text, candidate):
    """✅ Check if keyword present in text dengan word-boundary support."""
    # Gunakan improved version jika available
    if IMPROVEMENTS_AVAILABLE:
        from suggested_improvements import keyword_present_in_text as keyword_present_in_text_improved
        return keyword_present_in_text_improved(processed_text, candidate)
    
    # Fallback ke old implementation
    if not candidate:
        return False
    t = processed_text.lower()
    c = candidate.lower()
    # exact phrase
    if ' ' in c:
        return c in t
    # for single token, use word boundary
    return re.search(r'\b' + re.escape(c) + r'\b', t) is not None

def init_llm_universal(model_name=None, ctx_size_override=None, performance_mode=True):
    """Universal model initializer untuk semua model
    
    Args:
        model_name: Model yang diinginkan
        ctx_size_override: Override context window size
        performance_mode: Jika True, optimize untuk RTX 4060 (faster inference)
    """
    
    if not model_name:
        model_name = "Qwen3-4B-Instruct"
    
    # Deteksi model family untuk default settings
    optimizer = UniversalPromptOptimizer()
    profile = optimizer.get_model_profile(model_name)
    
    # Context size: override > model profile > default
    # OPTIMIZATION: Reduce context window untuk RTX 4060
    if performance_mode:
        n_ctx = ctx_size_override or min(profile['optimal_context'] or 4096, 2048)
        print("[PERF] Performance mode: context reduced to", n_ctx)
    else:
        n_ctx = ctx_size_override or profile['optimal_context'] or 4096
    
    def resolve_model_path(name):
        model_root = 'model'
        raw_name = str(name).strip()
        normalized_name = os.path.splitext(os.path.basename(raw_name))[0].lower()

        # Fast-path: direct candidates.
        direct_candidates = [
            os.path.join(model_root, raw_name, f'{raw_name}.gguf'),
            os.path.join(model_root, f'{raw_name}.gguf'),
            os.path.join(model_root, raw_name),
        ]
        for candidate in direct_candidates:
            if os.path.isfile(candidate) and candidate.lower().endswith('.gguf'):
                return candidate

        # Robust path: scan nested gguf files inside model/.
        if not os.path.isdir(model_root):
            return None

        gguf_files = []
        for root, _, files in os.walk(model_root):
            for filename in files:
                if filename.lower().endswith('.gguf'):
                    full_path = os.path.join(root, filename)
                    stem = os.path.splitext(filename)[0].lower()
                    parent = os.path.basename(root).lower()
                    gguf_files.append((full_path, stem, parent))

        if not gguf_files:
            return None

        # Priority 1: exact filename stem match.
        for full_path, stem, _ in gguf_files:
            if stem == normalized_name:
                return full_path

        # Priority 2: exact parent folder match.
        for full_path, _, parent in gguf_files:
            if parent == normalized_name:
                return full_path

        # Priority 3: relaxed contains match for slightly different naming.
        for full_path, stem, parent in gguf_files:
            if normalized_name in stem or stem in normalized_name:
                return full_path
            if normalized_name in parent or parent in normalized_name:
                return full_path

        return None

    selected_model_path = resolve_model_path(model_name)
    if not selected_model_path:
        raise ValueError(f"Model {model_name} not found")

    is_windows = os.name == 'nt'
    default_threads = max(4, min(8, os.cpu_count() or 8))
    gpu_layers_override = _env_int('CLASSIFY_GPU_LAYERS', -1)

    # Windows default: coba GPU dulu untuk mode performa, fallback CPU otomatis jika gagal.
    windows_gpu_layers = 0
    if is_windows and performance_mode:
        try:
            windows_gpu_layers = int(os.getenv('LLM_WINDOWS_GPU_LAYERS', '28'))
        except ValueError:
            windows_gpu_layers = 28

    # Common parameters untuk semua model
    common_params = {
        'model_path': selected_model_path,
        'n_ctx': int(n_ctx),
        'n_batch': 512,
        # Di Windows mode performa, coba GPU terlebih dulu agar inferensi lebih cepat.
        'n_gpu_layers': windows_gpu_layers if is_windows else 25,
        'f16_kv': True,
        'temperature': 0.0,
        'top_p': 0.9,
        'top_k': 40,
        'n_threads': default_threads,
        'verbose': False,
        'use_mlock': False if is_windows else True,
        'use_mmap': True,
        'seed': 42,
        'repeat_penalty': 1.1,
    }

    # Model-specific optimizations
    model_family = optimizer.detect_model_family(model_name)
    
    # Performance mode optimization untuk RTX 4060
    if performance_mode:
        common_params['n_batch'] = 256          # Reduce batch size
        common_params['f16_kv'] = False         # Disable fp16 
        common_params['temperature'] = 0.0      # Balanced (not too low to break format)
        common_params['top_p'] = 0.9            # Standard (keep format compliance)
        common_params['top_k'] = 40             # Standard
        common_params['repeat_penalty'] = 1.1   # Standard
        if n_ctx > 2048:
            common_params['n_ctx'] = 2048       # Cap context to 2K
        print(f"⚡ Performance mode ENABLED - Context: {common_params['n_ctx']}, Batch: {common_params['n_batch']}")

    # GPU layer strategy: aggressive in performance mode, fallback-safe on Windows
    if gpu_layers_override >= 0:
        common_params['n_gpu_layers'] = gpu_layers_override
    elif performance_mode and not is_windows:
        # Max GPU utilization for RTX 4060 in performance mode
        if model_family in ['qwen', 'qwen2', 'qwen3', 'phi', 'gemma']:
            common_params['n_gpu_layers'] = 40  # Aggressive for small models
        elif model_family in ['llama', 'llama2', 'llama3']:
            common_params['n_gpu_layers'] = 40
        elif model_family in ['mistral', 'mixtral', 'komodo']:
            common_params['n_gpu_layers'] = 45
        else:
            common_params['n_gpu_layers'] = 35
    else:
        # Standard or Windows CPU-first approach
        if model_family in ['qwen', 'qwen2', 'qwen3', 'phi', 'gemma']:
            common_params['n_gpu_layers'] = windows_gpu_layers if is_windows else 30
        elif model_family in ['llama', 'llama2', 'llama3']:
            common_params['n_gpu_layers'] = windows_gpu_layers if is_windows else 35
        elif model_family in ['mistral', 'mixtral', 'komodo']:
            common_params['n_gpu_layers'] = windows_gpu_layers if is_windows else 40
        else:
            common_params['n_gpu_layers'] = windows_gpu_layers if is_windows else 25
    
    common_params.update({
        'n_threads': default_threads,
        'rope_freq_base': 10000 if model_family not in ['llama', 'llama2', 'llama3'] else 1000000,
    })

    # Fallback model names (lebih ringan) jika model utama gagal di-load.
    fallback_names = [
        'Qwen3-4B-Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
        'komodo-7b-base.Q5_0',
    ]

    attempts = []

    if is_windows and performance_mode:
        # Auto-tuning GPU layers untuk RTX 4060/Windows.
        # Bisa override dengan env: LLM_WINDOWS_GPU_LAYERS_CANDIDATES="35,28,20,12"
        raw_candidates = os.getenv('LLM_WINDOWS_GPU_LAYERS_CANDIDATES', '35,28,20,12')
        parsed_candidates = []
        for token in raw_candidates.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                val = int(token)
            except ValueError:
                continue
            if val >= 0:
                parsed_candidates.append(val)

        gpu_candidates = [int(common_params.get('n_gpu_layers', 0))] + parsed_candidates
        unique_gpu_candidates = []
        for layer in gpu_candidates:
            if layer not in unique_gpu_candidates:
                unique_gpu_candidates.append(layer)

        print(f"[PERF] Windows GPU layer candidates: {unique_gpu_candidates}")
        for layer in unique_gpu_candidates:
            tuned_params = dict(common_params)
            tuned_params['n_gpu_layers'] = layer
            attempts.append((
                f"selected:{model_name}:optimized_gpu_{layer}",
                tuned_params,
            ))
    else:
        # Attempt 1: selected model optimized config
        attempts.append((
            f"selected:{model_name}:optimized",
            dict(common_params),
        ))

    # Attempt 2: selected model safe config (CPU, lower context)
    safe_selected = dict(common_params)
    safe_selected.update({
        'n_ctx': min(int(n_ctx), 2048 if performance_mode else 4096),
        'n_batch': 128 if not performance_mode else 256,
        'n_gpu_layers': 0,
        'n_threads': default_threads,
        'use_mlock': False,
        'f16_kv': False,
    })

    attempts.append((
        f"selected:{model_name}:safe",
        safe_selected,
    ))

    # Ultra-safe fallback khusus Windows untuk menghindari [WinError 1] pada beberapa runtime.
    if is_windows:
        windows_ultra_safe = dict(safe_selected)
        windows_ultra_safe.update({
            'use_mmap': False,
            'n_batch': 64,
            'temperature': 0.0,
        })
        attempts.append((
            f"selected:{model_name}:windows_ultra_safe",
            windows_ultra_safe,
        ))

    # Attempt 3+: lightweight fallback model(s) using safe config
    for fallback_name in fallback_names:
        if fallback_name == model_name:
            continue
        fallback_path = resolve_model_path(fallback_name)
        if not fallback_path:
            continue

        fallback_params = dict(safe_selected)
        fallback_params['model_path'] = fallback_path
        attempts.append((
            f"fallback:{fallback_name}:safe",
            fallback_params,
        ))

    print(f"🚀 Initializing {model_name} | Family: {model_family} | Context: {n_ctx}")

    load_errors = []
    for attempt_name, params in attempts:
        try:
            print(f"🔄 Loading LLM attempt: {attempt_name}")
            llm = LlamaCpp(**params)
            if attempt_name.startswith('fallback:'):
                print(f"⚠️ Requested model '{model_name}' gagal load. Menggunakan {attempt_name}.")
            return llm
        except Exception as e:
            err_detail = f"{type(e).__name__}: {e}"
            if hasattr(e, 'winerror') and getattr(e, 'winerror', None) is not None:
                err_detail += f" (winerror={e.winerror})"
            load_errors.append(f"{attempt_name} -> {err_detail}")
            print(f"❌ LLM load failed: {attempt_name} | {e}")

    raise ValueError(
        "Could not initialize any LlamaCpp model. Attempts: " + " | ".join(load_errors)
    )

def init_llm(model_name=None):
    """Legacy model initializer - untuk compatibility"""
    return init_llm_universal(model_name)

# ==================== COMPATIBILITY WRAPPERS ====================

# Initialize global instance
classification_system = ClassificationSystem()

# Primary universal function
def classify_universal(file_path, socketio=None, config_id=None, model_name=None, 
                      ctx_size_override=None, benchmark_mode=False, strict_keyword_match=None,
                      classification_mode=CLASSIFY_DEFAULT_MODE):
    return classification_system.classify_universal(
        file_path,
        socketio,
        config_id,
        model_name,
        ctx_size_override,
        benchmark_mode,
        strict_keyword_match,
        classification_mode,
    )

# Primary function - HANYA SATU DEFINISI
def classify_dataset(file_path, socketio=None, config_id=None, model_name=None, 
                   strict_keyword_match=None, classification_mode=CLASSIFY_DEFAULT_MODE):
    return classification_system.classify_universal(
        file_path,
        socketio,
        config_id,
        model_name,
        strict_keyword_match=strict_keyword_match,
        classification_mode=classification_mode,
    )

# Qwen-specific function
def classify_with_qwen_optimized(file_path, socketio=None, config_id=None, 
                               model_name="Qwen3-4B-Instruct"):
    return classify_universal(file_path, socketio, config_id, model_name)

# Legacy compatibility
def classify_dataset_legacy(file_path, socketio=None, prompt_id="classify_telkom"):
    return classify_dataset(file_path, socketio, strict_keyword_match=None, classification_mode='fast_mode')