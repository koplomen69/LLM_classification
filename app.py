from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import sys
import io
import builtins
import logging
import tempfile
from datetime import datetime
import tempfile
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from enhanced_classifyaduan import EnhancedAduanClassifier
from chat import ChatSession
from compare_evaluator import compare_datasets  # Import the comparison function
from models import db, PromptConfig, PromptComponent
from prompt_manager import prompt_manager
from datetime import datetime
from classify import classify_universal


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


class _SocketIOAccessFilter(logging.Filter):
    """Suppress noisy /socket.io polling access logs in terminal output."""

    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return '/socket.io/' not in msg


_werkzeug_logger = logging.getLogger('werkzeug')
_werkzeug_logger.addFilter(_SocketIOAccessFilter())

# ✅ Fix Windows encoding issue for emoji in debug output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # If wrapping fails, continue with default encoding

# Integrasi SVM + keyword dari folder eksperimen aduan.
try:
    import sys

    SVM_MODULE_DIR = os.path.join(os.path.dirname(__file__), 'kode_aduan_text_klasifikasi')
    if os.path.isdir(SVM_MODULE_DIR) and SVM_MODULE_DIR not in sys.path:
        sys.path.append(SVM_MODULE_DIR)

    from svm_aduan_classifier import (  # type: ignore
        preprocess_text as svm_preprocess_text,
        load_keyword_set as svm_load_keyword_set,
        keyword_rule_score as svm_keyword_rule_score,
        decide_label_with_rules as svm_decide_label_with_rules,
        DEFAULT_NON_ADUAN_KEYWORDS as SVM_DEFAULT_NON_ADUAN_KEYWORDS,
    )

    SVM_MODULE_AVAILABLE = True
except Exception as svm_import_error:
    print(f"SVM integration import warning: {svm_import_error}")
    SVM_MODULE_AVAILABLE = False


SVM_PRIMARY_MODEL_PATH = os.path.join('kode_aduan_text_klasifikasi', 'ml_models', 'svm_aduan_pipeline_active.joblib')
SVM_FALLBACK_MODEL_PATH = os.path.join('kode_aduan_text_klasifikasi', 'ml_models', 'svm_aduan_pipeline.joblib')
SVM_KEYWORD_PATH = os.path.join('kode_aduan_text_klasifikasi', 'list_kyword.csv')

_svm_runtime_cache = {
    'model': None,
    'model_path': None,
    'aduan_keywords': None,
}

DEFAULT_AVAILABLE_MODELS = [
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
    'Qwen2-7B-Instruct-Q4_K_M',
    'Mistral-7B-Instruct-v0.2-Q4_K_M',
]

PROMPT_COMPONENT_LABELS = {
    'base': 'Base Template',
    'instructions': 'Instructions & Rules',
    'keywords': 'Keywords List',
    'examples': 'Examples',
}

# Initialize Flask and SocketIO with CORS settings
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   allow_upgrades=False,
                   logger=False,
                   engineio_logger=False)

# Direktori untuk file upload dan output
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prompts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()
    prompt_manager.initialize_default_config()

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Store chat sessions
chat_sessions = {}

# ✅ PERBAIKAN: Import classify functions SEKALI SAJA
def import_classify_functions():
    """Import classify functions setelah app siap"""
    try:
        from classify import classify_dataset as classify_direktorat
        from classifyaduan import classify_dataset as classify_aduan
        return classify_direktorat, classify_aduan
    except ImportError as e:
        print(f"Import error: {e}")
        # Fallback functions
        def fallback_classify(*args, **kwargs):
            raise Exception("Classify function not available")
        return fallback_classify, fallback_classify

# Panggil fungsi import
classify_direktorat, classify_aduan = import_classify_functions()

# Route for root
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prompt Management Routes
@app.route('/prompt-manager')
def prompt_manager_page():
    """Main prompt management page"""
    configs = PromptConfig.query.all()
    active_config = prompt_manager.get_active_config()
    return render_template('prompt_manager.html', 
                         configs=configs, 
                         active_config=active_config)
    
# Route for classify (GET/POST)
@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available LLM models - fixed version"""
    try:
        # Scan untuk model yang benar-benar ada di sistem
        available_models = scan_actual_models()
        
        print(f"DEBUG: Found {len(available_models)} models: {available_models}")
        
        # Jika tidak ada model yang terdeteksi, berikan default options
        if not available_models:
            available_models = DEFAULT_AVAILABLE_MODELS.copy()
            print("DEBUG: Using fallback models")
        
        return jsonify({
            'models': available_models,
            'total_models': len(available_models),
            'status': 'success',
            'source': 'filesystem' if available_models and available_models != DEFAULT_AVAILABLE_MODELS else 'fallback'
        })
        
    except Exception as e:
        print(f"ERROR in get_available_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Failed to get available models: {str(e)}',
            'models': DEFAULT_AVAILABLE_MODELS.copy(),
            'total_models': len(DEFAULT_AVAILABLE_MODELS),
            'status': 'success',
            'source': 'fallback'
        })


@app.route('/api/resource_snapshot', methods=['GET'])
def api_resource_snapshot():
    """Return a lightweight resource snapshot (CPU/GPU/RAM) for UI polling/debug."""
    try:
        from classify import sample_resources
        snapshot = sample_resources()
        return jsonify({'status': 'success', 'snapshot': snapshot})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

def scan_actual_models():
    """Scan actual model files - improved version"""
    try:
        model_dirs = [
            'model',
            './model',
            '../model',
            os.path.join(os.getcwd(), 'model'),
            'models',
            './models',
            os.path.join(os.getcwd(), 'models'),
        ]
        
        found_models = []
        
        print("=== SCANNING FOR MODELS ===")
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                print(f"Scanning directory: {model_dir}")
                
                try:
                    # Scan for GGUF files in root directory
                    for item in os.listdir(model_dir):
                        item_path = os.path.join(model_dir, item)
                        
                        # Check if it's a GGUF file
                        if os.path.isfile(item_path) and item.endswith('.gguf'):
                            model_name = os.path.splitext(item)[0]
                            found_models.append(model_name)
                            print(f"  ✓ Found GGUF file: {item}")
                        
                        # Check if it's a directory that might contain GGUF files
                        elif os.path.isdir(item_path):
                            # Look for GGUF files in this subdirectory
                            for sub_item in os.listdir(item_path):
                                sub_item_path = os.path.join(item_path, sub_item)
                                if os.path.isfile(sub_item_path) and sub_item.endswith('.gguf'):
                                    model_name = os.path.splitext(sub_item)[0]
                                    found_models.append(model_name)
                                    print(f"  ✓ Found GGUF in subdir: {item}/{sub_item}")
                                    
                except Exception as e:
                    print(f"  ✗ Error scanning {model_dir}: {e}")
                    continue
        
        # Remove duplicates and sort
        found_models = sorted(list(set(found_models)))
        
        print(f"=== SCAN COMPLETE: Found {len(found_models)} models ===")
        return found_models
        
    except Exception as e:
        print(f"ERROR in scan_actual_models: {e}")
        import traceback
        traceback.print_exc()
        return []

# Route untuk refresh models manual
@app.route('/api/refresh-models', methods=['POST'])
def refresh_models():
    """Force refresh of available models"""
    try:
        available_models = scan_actual_models()
        
        return jsonify({
            'message': f'Successfully refreshed models. Found {len(available_models)} models.',
            'models': available_models,
            'total_models': len(available_models),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to refresh models: {str(e)}',
            'status': 'error'
        }), 500

# Route for classify (GET/POST)
@app.route('/classify', methods=['GET', 'POST'])
def classify_direktorat_only():
    """Hanya klasifikasi direktorat untuk data yang sudah berupa aduan"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
                
            file = request.files['file']
            model_name = request.form.get('model_select', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M')
            classification_mode = request.form.get('classification_mode', 'pure_llm')
            raw_prompt_config_id = request.form.get('prompt_config_id')
            prompt_config_id = int(raw_prompt_config_id) if raw_prompt_config_id and raw_prompt_config_id.isdigit() else None
            
            # Validate file
            if not file or file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            print("\n=== Starting Direktorat Classification Process ===")
            print(f"File: {file.filename}")
            print(f"Model: {model_name}")
            print(f"Prompt Config ID: {prompt_config_id}")

            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            file.save(file_path)

            # Step 1: Direktorat Classification ONLY
            socketio.emit('progress_update', {
                'status': 'Starting direktorat classification...',
                'percent': 0,
                'stage': 'Direktorat Classification'
            }, namespace='/')
            
            try:
                # Debug logging
                print("Starting direktorat classification...")
                print(f"Input file path: {file_path}")
                print(f"Classification mode: {classification_mode}")
                
                # ✅ PERBAIKAN: Pastikan klasifikasi berjalan dalam Flask app context
                with app.app_context():
                    direktorat_path = classify_direktorat(
                        file_path,
                        socketio,
                        config_id=prompt_config_id,
                        model_name=model_name,
                        classification_mode=classification_mode,
                    )
                print(f"Classification complete. Output path: {direktorat_path}")
                
                # Step 2: Automatic Comparison with Ground Truth (if available)
                comparison_results = None
                ground_truth_file = request.files.get('ground_truth_file')
                
                if ground_truth_file and ground_truth_file.filename != '':
                    socketio.emit('progress_update', {
                        'status': 'Starting automatic comparison with ground truth...',
                        'percent': 80,
                        'stage': 'Comparison'
                    }, namespace='/')
                    
                    try:
                        # Save ground truth file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_gt_file:
                            ground_truth_file.save(temp_gt_file.name)
                            ground_truth_path = temp_gt_file.name

                        print(f"Ground truth file: {ground_truth_file.filename}")
                        print(f"Predicted file: {direktorat_path}")
                        
                        # Compare datasets
                        comparison_results = compare_datasets(ground_truth_path, direktorat_path)
                        
                        socketio.emit('progress_update', {
                            'status': 'Comparison complete',
                            'percent': 95,
                            'stage': 'Comparison',
                            'comparison': comparison_results
                        }, namespace='/')
                        
                        # Clean up ground truth temp file
                        os.unlink(ground_truth_path)
                        
                    except Exception as e:
                        print(f"Comparison error: {str(e)}")
                        import traceback
                        print(f"Comparison traceback: {traceback.format_exc()}")
                        # Continue without comparison if it fails
                
                # Return results including comparison if available
                # Normalize path for JavaScript (convert backslashes to forward slashes)
                normalized_direktorat_path = direktorat_path.replace('\\', '/')
                
                result = {
                    'direktorat_path': normalized_direktorat_path,
                    'message': 'Direktorat classification completed successfully',
                    'prompt_config_id': prompt_config_id
                }
                
                if comparison_results:
                    result['comparison'] = comparison_results
                    result['message'] += ' with auto comparison'
                
                socketio.emit('progress_update', {
                    'status': 'Process completed successfully',
                    'percent': 100,
                    'stage': 'Complete',
                    'result': result
                }, namespace='/')
                
                return jsonify(result)
                
            except Exception as e:
                import traceback
                print(f"Direktorat classification error: {str(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                
                socketio.emit('progress_update', {
                    'status': f'Direktorat classification failed: {str(e)}',
                    'percent': 0,
                    'stage': 'Error',
                    'level': 'ERROR',
                }, namespace='/')

                return jsonify({
                    'error': f"Direktorat classification failed: {str(e)}"
                }), 500
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    # Handle GET request - just render the template
    return render_template('index.html')

@app.route('/classify-aduan', methods=['POST'])
def classify_aduan_route():
    """Endpoint untuk klasifikasi aduan text"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        # Get processing options
        model_type = request.form.get('aduan_model_type', 'svm_keyword')
        save_scores = request.form.get('save_detailed_scores') == 'on'
        filter_aduan_only = request.form.get('filter_aduan_only') == 'on'
        
        # ✅ PERBAIKAN: Gunakan fungsi yang sudah diimport
        result_path = classify_aduan_dataset(
            file_path, 
            socketio, 
            model_type=model_type,
            save_scores=save_scores,
            filter_aduan_only=filter_aduan_only
        )
        
        # Hitung statistik
        statistics = get_classification_statistics(result_path)
        
        # Normalize path for JavaScript (convert backslashes to forward slashes)
        normalized_result_path = result_path.replace('\\', '/')
        
        return jsonify({
            'success': True,
            'message': 'Klasifikasi aduan selesai',
            'model_type': model_type,
            'result_path': normalized_result_path,
            'statistics': statistics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/prompt-configs/<int:config_id>', methods=['PUT'])
def update_prompt_config(config_id):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        name = data.get('name')
        description = data.get('description')
        
        print(f"Updating config {config_id}: name={name}, description={description}")
        
        config = PromptConfig.query.get(config_id)
        if not config:
            return jsonify({'error': 'Configuration not found'}), 404
        
        if name:
            config.name = name
        if description is not None:  # Bisa string kosong
            config.description = description
            
        config.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Configuration updated successfully',
            'config': {
                'id': config.id,
                'name': config.name,
                'description': config.description,
                'is_active': config.is_active,
                'updated_at': config.updated_at.isoformat()
            }
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating config {config_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# API Routes untuk Prompt Management (tetap sama)
@app.route('/api/prompt-configs', methods=['GET'])
def get_prompt_configs():
    """Get all prompt configurations"""
    configs = PromptConfig.query.all()
    return jsonify([{
        'id': config.id,
        'name': config.name,
        'description': config.description,
        'is_active': config.is_active,
        'created_at': config.created_at.isoformat(),
        'component_count': len(config.components)
    } for config in configs])

@app.route('/api/prompt-configs', methods=['POST'])
def create_prompt_config():
    """Create new prompt configuration"""
    data = request.json
    new_config = prompt_manager.create_new_config(
        name=data['name'],
        description=data.get('description', ''),
        copy_from=data.get('copy_from')
    )
    return jsonify({'id': new_config.id, 'message': 'Configuration created'})

@app.route('/api/prompt-configs/<int:config_id>', methods=['GET'])
def get_prompt_config(config_id):
    """Get specific prompt configuration with components"""
    config = PromptConfig.query.get_or_404(config_id)
    components = prompt_manager.get_config_components(config_id)
    
    return jsonify({
        'config': {
            'id': config.id,
            'name': config.name,
            'description': config.description,
            'is_active': config.is_active
        },
        'components': [{
            'id': comp.id,
            'type': comp.component_type,
            'name': PROMPT_COMPONENT_LABELS.get(comp.component_type, comp.component_type.title()),
            'content': comp.content,
            'is_enabled': comp.is_enabled,
            'order_index': comp.order_index
        } for comp in components]
    })

@app.route('/api/prompt-configs/<int:config_id>/activate', methods=['POST'])
def activate_prompt_config(config_id):
    """Activate a prompt configuration"""
    prompt_manager.activate_config(config_id)
    return jsonify({'message': 'Configuration activated'})

@app.route('/api/prompt-components/<int:component_id>', methods=['PUT'])
def update_prompt_component(component_id):
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        content = data.get('content')
        is_enabled = data.get('is_enabled')
        
        print(f"Updating component {component_id}: content={content is not None}, is_enabled={is_enabled}")
        
        # Update the component
        prompt_manager.update_component(
            component_id=component_id,
            content=content,
            is_enabled=is_enabled
        )
        
        return jsonify({'message': 'Component updated successfully'})
        
    except Exception as e:
        print(f"Error updating component {component_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/build-prompt', methods=['POST'])
def build_prompt_preview():
    """Build and preview prompt for given text"""
    data = request.json
    text = data['text']
    raw_config_id = data.get('config_id')
    config_id = int(raw_config_id) if raw_config_id not in [None, '', 'null'] else None

    if config_id is None:
        active_config = prompt_manager.get_active_config()
        if not active_config:
            return jsonify({'error': 'No active prompt configuration found'}), 400
        config_id = active_config.id
    
    prompt = prompt_manager.build_prompt(text, config_id)
    
    return jsonify({
        'prompt': prompt,
        'length': len(prompt),
        'components_used': [comp.component_type for comp in prompt_manager.get_config_components(config_id) if comp.is_enabled]
    })


@app.route('/api/prompt-configs/<int:config_id>', methods=['DELETE'])
def delete_prompt_config(config_id):
    try:
        config = PromptConfig.query.get(config_id)
        if not config:
            return jsonify({'error': 'Configuration not found'}), 404
        
        # Prevent deletion of active config
        if config.is_active:
            return jsonify({'error': 'Cannot delete active configuration. Please activate another configuration first.'}), 400
        
        # Prevent deletion of default config
        if config.name == 'Default Config':
            return jsonify({'error': 'Cannot delete default configuration'}), 400
        
        config_name = config.name
        
        # Delete the configuration (cascade delete components)
        db.session.delete(config)
        db.session.commit()
        
        return jsonify({
            'message': f'Configuration "{config_name}" deleted successfully',
            'deleted_id': config_id
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting config {config_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-prompt', methods=['POST'])
def test_prompt():
    """Test prompt configuration with actual classification"""
    data = request.json
    text = data['text']
    config_id = data.get('config_id')
    
    # Build prompt
    prompt = prompt_manager.build_prompt(text, config_id)
    
    # Initialize LLM and test
    from classify import init_llm, parse_llm_output
    llm = init_llm()
    
    try:
        output = llm.invoke(prompt)
        direktorat, keyword = parse_llm_output(output)
        
        return jsonify({
            'success': True,
            'input_text': text,
            'generated_prompt': prompt,
            'llm_output': output,
            'parsed_result': {
                'direktorat': direktorat,
                'keyword': keyword
            },
            'prompt_length': len(prompt)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'generated_prompt': prompt
        }), 500


def _resolve_svm_model_path():
    """Pilih model SVM yang tersedia: active model dulu, fallback ke model dasar."""
    if os.path.exists(SVM_PRIMARY_MODEL_PATH):
        return SVM_PRIMARY_MODEL_PATH
    if os.path.exists(SVM_FALLBACK_MODEL_PATH):
        return SVM_FALLBACK_MODEL_PATH
    raise FileNotFoundError(
        f"SVM model tidak ditemukan. Cek: {SVM_PRIMARY_MODEL_PATH} atau {SVM_FALLBACK_MODEL_PATH}"
    )


def _load_svm_resources():
    """Load model + keyword sekali, lalu simpan di cache memory."""
    if not SVM_MODULE_AVAILABLE:
        raise RuntimeError("Integrasi modul SVM tidak tersedia.")

    model_path = _resolve_svm_model_path()
    if _svm_runtime_cache['model'] is None or _svm_runtime_cache['model_path'] != model_path:
        _svm_runtime_cache['model'] = joblib.load(model_path)
        _svm_runtime_cache['model_path'] = model_path

    if _svm_runtime_cache['aduan_keywords'] is None:
        _svm_runtime_cache['aduan_keywords'] = svm_load_keyword_set(SVM_KEYWORD_PATH)

    return (
        _svm_runtime_cache['model'],
        _svm_runtime_cache['aduan_keywords'],
        SVM_DEFAULT_NON_ADUAN_KEYWORDS,
    )


def _classify_single_with_model(text, model_type, enhanced_classifier=None):
    """Klasifikasi 1 teks dengan opsi model yang dipilih dari UI."""
    safe_text = str(text) if text is not None else ""

    if model_type in ['svm_keyword', 'svm_keyword_strict', 'svm_only']:
        svm_model, aduan_keywords, non_aduan_keywords = _load_svm_resources()
        cleaned = svm_preprocess_text(safe_text)
        proba = svm_model.predict_proba([cleaned])[0]
        prob_bukan = float(proba[0])
        prob_aduan = float(proba[1])

        if model_type == 'svm_only':
            label = 'aduan_text' if prob_aduan >= 0.5 else 'not_aduan_text'
            rule_score = 0.0
            confidence = prob_aduan if label == 'aduan_text' else prob_bukan
        else:
            rule_score = float(svm_keyword_rule_score(safe_text, aduan_keywords, non_aduan_keywords))
            strict_mode = model_type == 'svm_keyword_strict'
            raw_label, confidence = svm_decide_label_with_rules(
                prob_aduan=prob_aduan,
                rule_score=rule_score,
                cleaned_text=cleaned,
                threshold=0.6,
                strict_aduan_mode=strict_mode,
                high_conf_override=0.9,
            )
            label = 'aduan_text' if raw_label == 'aduan_text' else 'not_aduan_text'

            # Zona ambigu untuk mempermudah review manual.
            if label == 'not_aduan_text' and (0.45 <= prob_aduan <= 0.58) and (-0.5 <= rule_score <= 0.8):
                label = 'review_needed'

        return {
            'classification': label,
            'rule_score': float(rule_score),
            'ml_probability': prob_aduan,
            'prob_aduan_text': prob_aduan,
            'prob_not_aduan_text': prob_bukan,
            'confidence': float(confidence),
        }

    # Existing enhanced classifier flow.
    if enhanced_classifier is None:
        enhanced_classifier = EnhancedAduanClassifier()
        if model_type in ['enhanced_ml', 'ml_only']:
            enhanced_classifier.load_ml_model()

    if model_type == 'rule_only':
        score = float(enhanced_classifier.calculate_rule_score(safe_text))
        if score > 0.8:
            label = 'aduan_text'
        elif score > 0.1:
            label = 'review_needed'
        else:
            label = 'not_aduan_text'

        return {
            'classification': label,
            'rule_score': score,
            'ml_probability': None,
            'prob_aduan_text': None,
            'prob_not_aduan_text': None,
            'confidence': abs(score),
        }

    classification, score, ml_prob = enhanced_classifier.classify_text(safe_text)
    return {
        'classification': classification,
        'rule_score': float(score) if score is not None else 0.0,
        'ml_probability': float(ml_prob) if ml_prob is not None else None,
        'prob_aduan_text': float(ml_prob) if ml_prob is not None else None,
        'prob_not_aduan_text': (1.0 - float(ml_prob)) if ml_prob is not None else None,
        'confidence': float(ml_prob) if ml_prob is not None else abs(float(score) if score is not None else 0.0),
    }

# ✅ PERBAIKAN: Hanya satu fungsi classify_aduan_dataset
def classify_aduan_dataset(file_path, socketio=None, model_type='enhanced_ml', 
                          save_scores=True, filter_aduan_only=False):
    """Klasifikasi dataset aduan dengan beberapa mode model (enhanced/rule/ml/svm-keyword)."""
    
    def log_message(msg, level="INFO"):
        full_msg = f"[{level}] {msg}"
        print(full_msg)
        if socketio:
            try:
                socketio.emit('aduan_progress_update', {'status': msg, 'level': level}, namespace='/')
            except Exception as e:
                print(f"Error emitting aduan socket message: {e}")
    
    log_message(f"Starting aduan classification process for file: {file_path}")
    
    try:
        # Baca file lebih robust (encoding/delimiter).
        df = read_data_file(file_path)
        
        # Pastikan kolom text ada
        text_column = 'text'
        if text_column not in df.columns:
            potential_text_columns = [col for col in df.columns if 'text' in col.lower()]
            if potential_text_columns:
                text_column = potential_text_columns[0]
            else:
                raise ValueError(f"No text column found in the dataset. Available columns: {df.columns.tolist()}")
        
        classifier = None
        if model_type in ['enhanced_ml', 'ml_only', 'rule_only']:
            classifier = EnhancedAduanClassifier()
            if model_type in ['enhanced_ml', 'ml_only']:
                classifier.load_ml_model()
        
        total_rows = len(df)
        results = []
        
        log_message(f"Processing {total_rows} texts with model '{model_type}'...")

        aduan_count = 0
        non_aduan_count = 0
        review_count = 0
        
        # Proses klasifikasi
        for idx, text in enumerate(df[text_column]):
            prediction = _classify_single_with_model(
                text=str(text),
                model_type=model_type,
                enhanced_classifier=classifier,
            )

            classification = prediction['classification']
            score = prediction['rule_score']
            ml_prob = prediction['ml_probability']

            if classification == 'aduan_text':
                aduan_count += 1
            elif classification in ['not_aduan_text', 'bukan_aduan']:
                non_aduan_count += 1
            else:
                review_count += 1

            results.append({
                'text': text,
                'Kategori': classification,
                'Rule_Score': score,
                'ML_Probability': ml_prob if ml_prob is not None else 'N/A',
                'Prob_Aduan': prediction['prob_aduan_text'] if prediction['prob_aduan_text'] is not None else 'N/A',
                'Prob_Not_Aduan': prediction['prob_not_aduan_text'] if prediction['prob_not_aduan_text'] is not None else 'N/A',
                'Model_Type': model_type,
            })

            if socketio and idx % 10 == 0:
                socketio.emit('aduan_progress_update', {
                    'status': f'Processing text {idx + 1} of {total_rows}...',
                    'percent': int((idx + 1) / total_rows * 100),
                    'stage': 'Aduan Classification',
                    'stats': {
                        'processed': idx + 1,
                        'total': total_rows,
                        'aduan_count': aduan_count,
                        'non_aduan_count': non_aduan_count,
                        'review_count': review_count,
                        'current_text': str(text)[:100] + '...' if len(str(text)) > 100 else str(text),
                        'current_classification': classification,
                        'current_score': f"{score:.2f}",
                    }
                })
        
        # Buat DataFrame hasil
        results_df = pd.DataFrame(results)
        
        # Filter hanya aduan jika diminta
        if filter_aduan_only:
            original_count = len(results_df)
            results_df = results_df[results_df['Kategori'] == 'aduan_text']
            log_message(f"Filtered to {len(results_df)} aduan texts from {original_count} total texts")
        
        # Hapus kolom score jika tidak disimpan
        if not save_scores:
            cols_to_drop = ['Rule_Score', 'ML_Probability', 'Prob_Aduan', 'Prob_Not_Aduan']
            existing_cols = [c for c in cols_to_drop if c in results_df.columns]
            results_df = results_df.drop(existing_cols, axis=1)
        
        # Simpan hasil
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_aduan_classified.xlsx")
        results_df.to_excel(output_path, index=False)
        
        log_message(f"Aduan classification completed. Results saved to: {output_path}")
        
        if socketio:
            socketio.emit('aduan_progress_update', {
                'status': 'Aduan classification completed successfully',
                'percent': 100,
                'stage': 'Complete'
            })
        
        return output_path
        
    except Exception as e:
        error_msg = f"Error during aduan classification: {str(e)}"
        log_message(error_msg, "ERROR")
        if socketio:
            socketio.emit('aduan_progress_update', {
                'status': f'Error: {str(e)}',
                'percent': 0,
                'level': 'ERROR'
            })
        raise
    
@app.route('/quick-test-aduan', methods=['POST'])
def quick_test_aduan():
    """Endpoint untuk test cepat klasifikasi aduan"""
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model_type', 'svm_keyword')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        classifier = None
        if model_type in ['enhanced_ml', 'ml_only', 'rule_only']:
            classifier = EnhancedAduanClassifier()
            if model_type in ['enhanced_ml', 'ml_only']:
                classifier.load_ml_model()

        pred = _classify_single_with_model(text=text, model_type=model_type, enhanced_classifier=classifier)
        classification = pred['classification']
        score = pred['rule_score']
        ml_prob = pred['ml_probability']
        
        # Provide analysis
        analysis = f"Teks diklasifikasikan sebagai {classification} dengan skor {score:.2f}"
        if ml_prob:
            analysis += f" dan probabilitas ML {ml_prob:.2f}"
        
        return jsonify({
            'classification': classification,
            'score': score,
            'ml_probability': ml_prob,
            'model_type': model_type,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_classification_statistics(result_path):
    """Hitung statistik dari hasil klasifikasi"""
    try:
        df = pd.read_excel(result_path)
        total = len(df)
        aduan_count = len(df[df['Kategori'] == 'aduan_text'])
        non_aduan_count = len(df[df['Kategori'].isin(['not_aduan_text', 'bukan_aduan'])])
        review_count = len(df[df['Kategori'] == 'review_needed'])
        
        return {
            'total': total,
            'aduan_count': aduan_count,
            'non_aduan_count': non_aduan_count,
            'review_count': review_count,
            'aduan_percentage': round((aduan_count / total) * 100, 2) if total > 0 else 0
        }
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return {}
    
# Fungsi untuk membaca file data
def read_data_file(file_path):
    """Read CSV or Excel file with proper encoding detection"""
    try:
        print(f"🔍 Reading file: {file_path}")
        
        # Check file extension
        if file_path.endswith('.csv'):
            # Try multiple encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            for encoding in encodings:
                try:
                    print(f"  Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"  ✅ Success with encoding: {encoding}")
                    print(f"  DataFrame shape: {df.shape}")
                    print(f"  Columns: {df.columns.tolist()}")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"  ❌ Error with {encoding}: {e}")
                    continue
            
            # If all encodings fail, try without specifying encoding
            try:
                df = pd.read_csv(file_path)
                print(f"  ✅ Success with default encoding")
                return df
            except Exception as e:
                raise ValueError(f"Failed to read CSV with any encoding: {str(e)}")
                
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            # For Excel files, try different engines
            engines = ['openpyxl', 'xlrd']
            for engine in engines:
                try:
                    print(f"  Trying Excel engine: {engine}")
                    df = pd.read_excel(file_path, engine=engine)
                    print(f"  ✅ Success with engine: {engine}")
                    print(f"  DataFrame shape: {df.shape}")
                    print(f"  Columns: {df.columns.tolist()}")
                    return df
                except Exception as e:
                    print(f"  ❌ Error with {engine}: {e}")
                    continue
            
            # If all engines fail, try with no engine specified
            try:
                df = pd.read_excel(file_path)
                print(f"  ✅ Success with default Excel engine")
                return df
            except Exception as e:
                raise ValueError(f"Failed to read Excel with any engine: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {str(e)}")
        raise ValueError(f"Error reading file: {str(e)}")

# Fungsi untuk validasi dataframe
def normalize_label(label):
    """Normalize directorate labels to consistent format"""
    if not label or pd.isna(label):
        return "Uncategorized"
    
    label = str(label).strip()
    
    # Handle common variations and typos
    normalizations = {
        'pasca': 'Pasca Sarjana dan Advance Learning',
        'pasca sarjana': 'Pasca Sarjana dan Advance Learning',
        'pascasarjana': 'Pasca Sarjana dan Advance Learning',
        'sdm': 'Sumber Daya Manusia',
        'hr': 'Sumber Daya Manusia',
        'sumber daya manusia': 'Sumber Daya Manusia',
        'ti': 'Pusat Teknologi Informasi',
        'teknologi informasi': 'Pusat Teknologi Informasi',
        'ict': 'Pusat Teknologi Informasi',
        'keuangan': 'Keuangan',
        'finance': 'Keuangan',
        'akademik': 'Akademik',
        'academic': 'Akademik',
        'kemahasiswaan': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'kemahasiswaan, pengembangan karir, alumni': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'pengembangan karir': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'karir': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'alumni': 'Kemahasiswaan, Pengembangan Karir, Alumni',
        'aset': 'Aset & Sustainability',
        'aset & sustainability': 'Aset & Sustainability',
        'asset': 'Aset & Sustainability',
        'pusat teknologi informasi': 'Pusat Teknologi Informasi',
        'teknologi informasi': 'Pusat Teknologi Informasi',
        'pti': 'Pusat Teknologi Informasi',
        'sumber daya manusia': 'Sumber Daya Manusia',
        'sdm': 'Sumber Daya Manusia',
        'pasca sarjana': 'Pasca Sarjana dan Advance Learning',
        'pasca sarjana dan advance learning': 'Pasca Sarjana dan Advance Learning',
        'pascasarjana': 'Pasca Sarjana dan Advance Learning',
        'unknown': 'Uncategorized',
        'lainnya': 'Uncategorized',
        'other': 'Uncategorized'
    }
    
    label_lower = label.lower()
    for variation, standard in normalizations.items():
        if variation in label_lower:
            return standard
    
    # Capitalize first letter of each word if no match found
    return ' '.join(word.capitalize() for word in label.split())

def validate_dataframes(df_approved, df_predicted):
    """Validate dataframe structure with flexible column naming"""
    
    print("🔍 Validating dataframe structures...")
    print(f"Approved columns: {df_approved.columns.tolist()}")
    print(f"Predicted columns: {df_predicted.columns.tolist()}")
    
    # Find actual and predicted columns with flexible naming
    actual_direktorat_col = find_column(df_approved, 
        ['actual_direktorat', 'actual', 'direktorat', 'ground_truth', 'true_label'])
    
    predicted_direktorat_col = find_column(df_predicted,
        ['predicted_direktorat', 'predicted', 'direktorat', 'prediction', 'result'])
    
    text_col_approved = find_column(df_approved, 
        ['text', 'teks', 'content', 'isi', 'document'])
    
    text_col_predicted = find_column(df_predicted,
        ['text', 'teks', 'content', 'isi', 'document'])
    
    print(f"Found columns - Actual: '{actual_direktorat_col}', Predicted: '{predicted_direktorat_col}'")
    print(f"Text columns - Approved: '{text_col_approved}', Predicted: '{text_col_predicted}'")
    
    # Validate required columns
    if not actual_direktorat_col:
        raise ValueError(f"Approved file missing direktorat column. Available columns: {df_approved.columns.tolist()}")
    if not predicted_direktorat_col:
        raise ValueError(f"Predicted file missing direktorat column. Available columns: {df_predicted.columns.tolist()}")
    if not text_col_approved:
        raise ValueError(f"Approved file missing text column. Available columns: {df_approved.columns.tolist()}")
    if not text_col_predicted:
        raise ValueError(f"Predicted file missing text column. Available columns: {df_predicted.columns.tolist()}")
    
    # Clean and normalize data
    df_approved = df_approved.dropna(subset=[text_col_approved, actual_direktorat_col]).copy()
    df_predicted = df_predicted.dropna(subset=[text_col_predicted, predicted_direktorat_col]).copy()
    
    # Normalize labels
    df_approved['Direktorat'] = df_approved[actual_direktorat_col].apply(normalize_label)
    df_predicted['Direktorat'] = df_predicted[predicted_direktorat_col].apply(normalize_label)
    
    # Standardize text column names for matching
    df_approved['text'] = df_approved[text_col_approved].astype(str)
    df_predicted['text'] = df_predicted[text_col_predicted].astype(str)
    
    print(f"✅ Validation complete")
    print(f"  Approved records: {len(df_approved)}")
    print(f"  Predicted records: {len(df_predicted)}")
    
    return df_approved, df_predicted

def find_column(df, possible_names):
    """Find column with flexible name matching"""
    df_columns = [col.lower().strip() for col in df.columns]
    
    for name_pattern in possible_names:
        name_lower = name_pattern.lower()
        for i, col in enumerate(df_columns):
            if name_lower in col or col in name_lower:
                print(f"  ✅ Found '{df.columns[i]}' for pattern '{name_pattern}'")
                return df.columns[i]
    
    print(f"  ❌ No match found for patterns: {possible_names}")
    return None

# Fungsi untuk menghitung metrik
def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    # Get unique labels
    labels = sorted(set(actual) | set(predicted))
    
    # Calculate overall metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, average='weighted', zero_division=0)
    recall = recall_score(actual, predicted, average='weighted', zero_division=0)
    f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)
    cm_dict = {}
    for i, actual_label in enumerate(labels):
        cm_dict[actual_label] = {}
        for j, pred_label in enumerate(labels):
            cm_dict[actual_label][pred_label] = int(cm[i, j])
    
    # Classification report per class
    classification_report = {}
    for label in labels:
        tp = np.sum([1 for a, p in zip(actual, predicted) if a == label and p == label])
        fp = np.sum([1 for a, p in zip(actual, predicted) if a != label and p == label])
        fn = np.sum([1 for a, p in zip(actual, predicted) if a == label and p != label])
        
        precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_class = (2 * precision_class * recall_class / (precision_class + recall_class) 
                   if (precision_class + recall_class) > 0 else 0)
        support = np.sum([1 for a in actual if a == label])
        
        classification_report[label] = {
            'precision': float(precision_class),
            'recall': float(recall_class),
            'f1_score': float(f1_class),
            'support': int(support)
        }
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm_dict,
        'classification_report': classification_report,
        'total_data': len(actual),
        'correct_predictions': sum(1 for a, p in zip(actual, predicted) if a == p),
        'incorrect_predictions': sum(1 for a, p in zip(actual, predicted) if a != p)
    }

# Tambahkan route ini di app.py

@app.route('/evaluate-aduan-model', methods=['POST'])
def evaluate_aduan_model():
    """Endpoint untuk evaluasi mendalam model aduan"""
    if 'evaluation_dataset' not in request.files:
        return jsonify({'error': 'No evaluation dataset uploaded'}), 400
    
    file = request.files['evaluation_dataset']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    model_type = request.form.get('evaluation_model_type', 'enhanced_ml')
    
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, f"eval_{file.filename}")
        file.save(file_path)
        
        # Run detailed evaluation
        evaluation_results = run_detailed_aduan_evaluation(file_path, model_type)
        
        return jsonify({
            'success': True,
            'message': 'Evaluasi mendalam selesai',
            **evaluation_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_detailed_aduan_evaluation(file_path, model_type='enhanced_ml'):
    """Jalankan evaluasi mendalam seperti di improved_validation.py"""
    from enhanced_classifyaduan import EnhancedAduanClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    from datetime import datetime
    
    # Load data
    df = read_data_file(file_path)
    
    # Prepare true labels
    if 'corrected_type' in df.columns:
        y_true = df['corrected_type'].fillna(df['aduan_type'])
    else:
        y_true = df['aduan_type']
    
    # Convert to binary
    y_true_binary = (y_true == 'aduan_text').astype(int)
    
    classifier = None
    if model_type in ['enhanced_ml', 'ml_only', 'rule_only']:
        classifier = EnhancedAduanClassifier()
        if model_type in ['enhanced_ml', 'ml_only']:
            classifier.load_ml_model()
    
    y_pred = []
    y_pred_binary = []
    detailed_results = []
    
    for text in df['text']:
        pred = _classify_single_with_model(text=text, model_type=model_type, enhanced_classifier=classifier)
        classification = pred['classification']
        score = pred['rule_score']
        ml_prob = pred['ml_probability']

        y_pred.append(classification)
        y_pred_binary.append(1 if classification == 'aduan_text' else 0)
        
        detailed_results.append({
            'text': text,
            'prediction': classification,
            'rule_score': score,
            'ml_probability': ml_prob
        })
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    cm_dict = {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }
    
    # Classification report
    clf_report = classification_report(y_true_binary, y_pred_binary, 
                                     target_names=['Not Aduan', 'Aduan'],
                                     output_dict=True)
    
    # Calculate counts
    total_data = len(y_true_binary)
    correct_predictions = sum(1 for a, p in zip(y_true_binary, y_pred_binary) if a == p)
    incorrect_predictions = total_data - correct_predictions
    
    # False positives analysis
    false_positives_analysis = (
        analyze_false_positives_patterns(df, y_true, y_pred, classifier)
        if classifier is not None
        else {'total_false_positives': int(sum(1 for t, p in zip(y_true_binary, y_pred_binary) if t == 0 and p == 1))}
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_data': total_data,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions,
        'false_positives': cm_dict['false_positive'],
        'false_negatives': cm_dict['false_negative'],
        'confusion_matrix': cm_dict,
        'classification_report': clf_report,
        'false_positives_analysis': false_positives_analysis,
        'model_type': model_type,
        'evaluation_date': datetime.now().isoformat(),
        'processing_time': 'N/A'  # Bisa ditambahkan timing jika diperlukan
    }


def analyze_false_positives_patterns(df, y_true, y_pred, classifier):
    """Analisis pola false positives seperti di analyze_false_positives.py"""
    import re
    from collections import Counter
    
    false_positives = []
    
    for idx, (text, true_label, pred_label) in enumerate(zip(df['text'], y_true, y_pred)):
        if pred_label == 'aduan_text' and true_label == 'not_aduan_text':
            processed = classifier.preprocess_text(text)
            rule_score = classifier.calculate_rule_score(text)
            
            # Find triggering keywords
            triggering_keywords = []
            for keyword in classifier.WEAK_INDICATORS:
                if re.search(rf'\b{keyword}\b', processed):
                    triggering_keywords.append(keyword)
            
            false_positives.append({
                'text': text,
                'processed': processed,
                'rule_score': rule_score,
                'triggering_keywords': triggering_keywords,
                'word_count': len(processed.split())
            })
    
    # Analyze patterns
    analysis = {
        'total_false_positives': len(false_positives)
    }
    
    if false_positives:
        # Top triggering keywords
        all_keywords = []
        for fp in false_positives:
            all_keywords.extend(fp['triggering_keywords'])
        
        keyword_counts = Counter(all_keywords)
        analysis['top_triggering_keywords'] = dict(keyword_counts.most_common(10))
        
        # Averages
        analysis['average_rule_score'] = sum(fp['rule_score'] for fp in false_positives) / len(false_positives)
        analysis['average_word_count'] = sum(fp['word_count'] for fp in false_positives) / len(false_positives)
        
        # Sample false positives
        analysis['sample_false_positives'] = false_positives[:5]
    
    return analysis


# Route untuk independent comparison via HTTP POST
@app.route('/compare', methods=['POST'])
def compare_datasets_http():
    try:
        approved_file = request.files.get('approved_file')
        predicted_file = request.files.get('predicted_file')
        
        if not approved_file or not predicted_file:
            return jsonify({'error': 'Both files are required'}), 400
        
        print(f"📊 Starting comparison:")
        print(f"  Approved file: {approved_file.filename}")
        print(f"  Predicted file: {predicted_file.filename}")
        
        # Simpan file sementara
        import tempfile
        approved_path = None
        predicted_path = None
        
        try:
            # Simpan file approved
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(approved_file.filename)[1]) as temp_approved:
                approved_file.save(temp_approved.name)
                approved_path = temp_approved.name
                print(f"  Saved approved file to: {approved_path}")
            
            # Simpan file predicted  
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(predicted_file.filename)[1]) as temp_predicted:
                predicted_file.save(temp_predicted.name)
                predicted_path = temp_predicted.name
                print(f"  Saved predicted file to: {predicted_path}")
            
            # Baca file dengan fungsi yang diperbaiki
            print("📖 Reading approved file...")
            df_approved = read_data_file(approved_path)
            
            print("📖 Reading predicted file...")
            df_predicted = read_data_file(predicted_path)
            
            print("✅ Files read successfully")
            print(f"  Approved shape: {df_approved.shape}")
            print(f"  Predicted shape: {df_predicted.shape}")
            print(f"  Approved columns: {df_approved.columns.tolist()}")
            print(f"  Predicted columns: {df_predicted.columns.tolist()}")
            
            # Validasi
            df_approved, df_predicted = validate_dataframes(df_approved, df_predicted)
            
            # Match records berdasarkan text
            detailed_results = []
            actual_labels = []
            predicted_labels = []

            match_count = 0
            total_approved = len(df_approved)

            print("🔍 Matching records based on text...")

            for idx, approved_row in df_approved.iterrows():
                text = approved_row['text']
                actual_direktorat = approved_row['Direktorat']
                
                # Clean text for matching
                clean_text = str(text).lower().strip()
                
                # Cari matching text di predicted file (exact match first)
                matching_predicted = df_predicted[df_predicted['text'].str.lower().str.strip() == clean_text]
                
                # Jika tidak ketemu, cari partial match
                if matching_predicted.empty and len(clean_text) > 10:
                    # Cari text yang mengandung substring yang sama
                    matching_predicted = df_predicted[
                        df_predicted['text'].str.lower().str.contains(clean_text[:20], na=False, regex=False)
                    ]
                
                if not matching_predicted.empty:
                    predicted_direktorat = matching_predicted.iloc[0]['Direktorat']
                    match = actual_direktorat == predicted_direktorat
                    
                    detailed_results.append({
                        'text': text,
                        'actual_direktorat': actual_direktorat,
                        'predicted_direktorat': predicted_direktorat,
                        'match': match
                    })
                    
                    actual_labels.append(actual_direktorat)
                    predicted_labels.append(predicted_direktorat)
                    match_count += 1
                
                # Progress logging
                if (idx + 1) % 10 == 0 or (idx + 1) == total_approved:
                    print(f"  📊 Processed {idx + 1}/{total_approved} records, found {match_count} matches")

            if not actual_labels:
                # Provide more helpful error message
                sample_approved = df_approved['text'].head(3).tolist()
                sample_predicted = df_predicted['text'].head(3).tolist()
                
                error_msg = f"""
                No matching records found between the two files. 
                
                Possible reasons:
                1. Text content doesn't match between files
                2. Different preprocessing or cleaning applied
                3. Files from different sources
                
                Sample from approved file:
                {sample_approved}
                
                Sample from predicted file:  
                {sample_predicted}
                
                Please ensure both files contain the same text entries.
                """
                return jsonify({'error': error_msg}), 400
            
            print(f"✅ Found {len(actual_labels)} matching records")
            
            # Hitung metrik
            metrics = calculate_metrics(actual_labels, predicted_labels)
            
            # Siapkan hasil final
            results = {
                'total_data': metrics['total_data'],
                'correct_predictions': metrics['correct_predictions'],
                'incorrect_predictions': metrics['incorrect_predictions'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': metrics['confusion_matrix'],
                'classification_report': metrics['classification_report'],
                'comparison_date': datetime.now().isoformat(),
                'matched_records': len(actual_labels),
                'match_rate': f"{(len(actual_labels) / total_approved * 100):.1f}%"
            }
            
            print(f"🎉 Comparison completed successfully")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Matched records: {results['matched_records']}")
            
            return jsonify(results)
            
        except Exception as e:
            print(f"❌ Error during comparison: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Comparison failed: {str(e)}'}), 500
            
        finally:
            # Clean up temporary files
            if approved_path and os.path.exists(approved_path):
                try:
                    os.unlink(approved_path)
                    print(f"🧹 Cleaned up approved temp file")
                except:
                    pass
            if predicted_path and os.path.exists(predicted_path):
                try:
                    os.unlink(predicted_path)
                    print(f"🧹 Cleaned up predicted temp file")
                except:
                    pass
                
    except Exception as e:
        print(f"❌ Comparison endpoint error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500
    
@app.route('/api/check-columns', methods=['POST'])
def check_columns():
    """Check column structure of uploaded files"""
    try:
        approved_file = request.files.get('approved_file')
        predicted_file = request.files.get('predicted_file')
        
        result = {
            'approved_file': None,
            'predicted_file': None,
            'compatibility': None
        }
        
        if approved_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(approved_file.filename)[1]) as temp_file:
                approved_file.save(temp_file.name)
                try:
                    df_approved = read_data_file(temp_file.name)
                    result['approved_file'] = {
                        'filename': approved_file.filename,
                        'columns': df_approved.columns.tolist(),
                        'shape': df_approved.shape,
                        'sample_texts': df_approved.iloc[:, 0].head(3).tolist() if len(df_approved.columns) > 0 else []
                    }
                finally:
                    os.unlink(temp_file.name)
        
        if predicted_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(predicted_file.filename)[1]) as temp_file:
                predicted_file.save(temp_file.name)
                try:
                    df_predicted = read_data_file(temp_file.name)
                    result['predicted_file'] = {
                        'filename': predicted_file.filename,
                        'columns': df_predicted.columns.tolist(),
                        'shape': df_predicted.shape,
                        'sample_texts': df_predicted.iloc[:, 0].head(3).tolist() if len(df_predicted.columns) > 0 else []
                    }
                finally:
                    os.unlink(temp_file.name)
        
        # Check compatibility
        if result['approved_file'] and result['predicted_file']:
            approved_cols = [col.lower() for col in result['approved_file']['columns']]
            predicted_cols = [col.lower() for col in result['predicted_file']['columns']]
            
            # Find potential matches
            actual_candidates = [col for col in approved_cols if any(keyword in col for keyword in ['actual', 'true', 'direktorat'])]
            predicted_candidates = [col for col in predicted_cols if any(keyword in col for keyword in ['predicted', 'result', 'direktorat'])]
            text_candidates_approved = [col for col in approved_cols if any(keyword in col for keyword in ['text', 'teks', 'content'])]
            text_candidates_predicted = [col for col in predicted_cols if any(keyword in col for keyword in ['text', 'teks', 'content'])]
            
            result['compatibility'] = {
                'actual_direktorat_candidates': actual_candidates,
                'predicted_direktorat_candidates': predicted_candidates,
                'text_column_candidates_approved': text_candidates_approved,
                'text_column_candidates_predicted': text_candidates_predicted,
                'can_proceed': len(actual_candidates) > 0 and len(predicted_candidates) > 0 and len(text_candidates_approved) > 0 and len(text_candidates_predicted) > 0
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route untuk download file
@app.route('/download/<path:filename>')
def download_file(filename):
    # Cek apakah file ada di output folder
    if os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    # Cek apakah file ada di upload folder
    elif os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    else:
        return "File not found", 404

# Route for chat (GET/POST)
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        try:
            session_id = request.form.get('session_id', 'default')
            user_input = request.form.get('user_input')
            
            if not user_input:
                return jsonify({'error': 'No input provided'}), 400

            # Get model selection
            model_name = request.form.get('model_select')
            
            # Get or create chat session
            if session_id not in chat_sessions:
                chat_sessions[session_id] = ChatSession(model_name)
            elif model_name:  # If model changed, create new session
                chat_sessions[session_id] = ChatSession(model_name)

            # Get response
            response = chat_sessions[session_id].get_response(user_input)
            print("Response from chat session:", response)
            
            # Ensure the response is properly formatted
            if isinstance(response, str):
                response = response.strip()
            
            return jsonify({
                'session_id': session_id,
                'response': response
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    # Handle GET request
    return render_template('index.html')

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    socketio.run(app, debug=True, port=5000)