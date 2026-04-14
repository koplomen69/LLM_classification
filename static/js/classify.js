// Scan available models
async function scanAvailableModels() {
    try {
        console.log('🔍 Fetching available models...');
        const response = await fetch('/api/available-models');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📦 Models API response:', data);
        
        // ✅ PERBAIKAN: Check status dengan benar
        if (data.status !== 'success') {
            throw new Error(data.error || 'API returned non-success status');
        }
        
        const models = data.models || [];
        const select = document.getElementById('model_select');
        
        if (!select) {
            console.error('❌ Model select element not found');
            return;
        }
        
        // Clear existing options
        select.innerHTML = '<option value="">-- Pilih Model --</option>';
        
        // Add models to dropdown
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });
        
        // Update model count
        const modelCountElement = document.getElementById('model-count');
        if (modelCountElement) {
            if (models.length > 0) {
                modelCountElement.innerHTML = `
                    <span class="text-success">
                        ✅ Found ${models.length} available models
                    </span>
                `;
            } else {
                modelCountElement.innerHTML = `
                    <span class="text-warning">
                        ⚠️ No models found. 
                        <br>
                        <small>Please add GGUF model files to the 'model' directory</small>
                    </span>
                `;
            }
        }
        
        console.log('✅ Available models loaded:', models);
        
    } catch (error) {
        console.error('❌ Error scanning models:', error);
        
        const modelCountElement = document.getElementById('model-count');
        if (modelCountElement) {
            modelCountElement.innerHTML = `
                <span class="text-danger">
                    ❌ Error loading models: ${error.message}
                    <br>
                    <small>Check browser console for details</small>
                </span>
            `;
        }
        
        // Fallback: add some default options
        const select = document.getElementById('model_select');
        if (select) {
            const fallbackModels = [
                'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
                'Qwen2-7B-Instruct-Q4_K_M',
                'Mistral-7B-Instruct-v0.2-Q4_K_M'
            ];
            
            select.innerHTML = '<option value="">-- Pilih Model --</option>';
            fallbackModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model + ' (fallback)';
                select.appendChild(option);
            });
            
            if (modelCountElement) {
                modelCountElement.innerHTML = `
                    <span class="text-warning">
                        ⚠️ Using fallback models due to connection error
                    </span>
                `;
            }
        }
    }
}

async function loadPromptConfigurations() {
    try {
        console.log('🔍 Loading prompt configurations...');
        const response = await fetch('/api/prompt-configs');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const configs = await response.json();
        const select = document.getElementById('prompt_config_id');
        
        if (!select) {
            console.error('❌ Prompt config select element not found');
            return;
        }
        
        // Keep default option
        const defaultOption = select.querySelector('option[value=""]');
        select.innerHTML = '';
        if (defaultOption) select.appendChild(defaultOption);
        
        // Add configurations
        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.id;
            option.textContent = `${config.name} ${config.is_active ? '(Active)' : ''}`;
            select.appendChild(option);
        });
        
        console.log('✅ Prompt configurations loaded:', configs.length, 'configs');
        
    } catch (error) {
        console.error('❌ Error loading prompt configs:', error);
        
        const select = document.getElementById('prompt_config_id');
        if (select) {
            const errorOption = document.createElement('option');
            errorOption.value = '';
            errorOption.textContent = 'Error loading configurations';
            errorOption.disabled = true;
            select.appendChild(errorOption);
        }
    }
}

// Refresh models manually
async function refreshModels() {
    try {
        console.log('🔄 Refreshing models...');
        const response = await fetch('/api/refresh-models', {
            method: 'POST'
        });
        
        const data = await response.json();
        console.log('🔄 Refresh response:', data);
        
        if (data.status === 'success') {
            // Show success message
            const modelCountElement = document.getElementById('model-count');
            if (modelCountElement) {
                modelCountElement.innerHTML = `
                    <span class="text-success">
                        ✅ ${data.message}
                    </span>
                `;
            }
            
            // Reload the dropdown
            await scanAvailableModels();
        } else {
            throw new Error(data.error || 'Refresh failed');
        }
    } catch (error) {
        console.error('❌ Error refreshing models:', error);
        alert('Error refreshing models: ' + error.message);
    }
}

// Quick test prompt configuration
function quickTestPrompt() {
    const text = prompt('Enter text to test current prompt configuration:', 'telyu krs saya error tidak bisa daftar mata kuliah');
    if (!text) return;
    
    const configId = document.getElementById('prompt_config_id')?.value || '';
    
    fetch('/api/test-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, config_id: configId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Classification Result:\nDirektorat: ${data.parsed_result.direktorat}\nKeyword: ${data.parsed_result.keyword}`);
        } else {
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error testing prompt:', error);
        alert('Error testing prompt: ' + error.message);
    });
}

function triggerAutoDownload(filePath) {
    if (!filePath) return;

    const filename = filePath.split('/').pop();
    if (!filename) return;

    const downloadUrl = `/download/${encodeURIComponent(filename)}`;
    const downloadLink = document.createElement('a');
    downloadLink.href = downloadUrl;
    downloadLink.download = filename;
    downloadLink.style.display = 'none';

    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();
}
const classificationStateKey = 'llm_classification_last_state';
const classificationProgressKey = 'llm_classification_last_progress';

function saveClassificationState(state) {
    try {
        localStorage.setItem(classificationStateKey, JSON.stringify(state));
    } catch (error) {
        console.warn('Unable to persist classification state:', error);
    }
}

function loadClassificationState() {
    try {
        const rawState = localStorage.getItem(classificationStateKey);
        return rawState ? JSON.parse(rawState) : null;
    } catch (error) {
        console.warn('Unable to load classification state:', error);
        return null;
    }
}

function saveClassificationProgress(progress) {
    try {
        localStorage.setItem(classificationProgressKey, JSON.stringify(progress));
    } catch (error) {
        console.warn('Unable to persist classification progress:', error);
    }
}

function loadClassificationProgress() {
    try {
        const rawProgress = localStorage.getItem(classificationProgressKey);
        return rawProgress ? JSON.parse(rawProgress) : null;
    } catch (error) {
        console.warn('Unable to load classification progress:', error);
        return null;
    }
}

function clearClassificationPersistence() {
    try {
        localStorage.removeItem(classificationStateKey);
        localStorage.removeItem(classificationProgressKey);
    } catch (error) {
        console.warn('Unable to clear classification persistence:', error);
    }
}

function initializeClassificationPage() {
    console.log('LLM Classification App loaded');
    scanAvailableModels();
    loadPromptConfigurations();
    initializeModelRefreshButton();
    initializeTooltips();
    restoreClassificationProgress();
    restoreClassificationState();
    setupClassificationSocket();
    bindClassificationForm();
    bindIndependentComparisonForm();
    initializeRuntimeLogControls();
    bindChatSection();
    loadConversationHistory();
}

function initializeRuntimeLogControls() {
    const clearBtn = document.getElementById('clear-runtime-log-btn');
    if (!clearBtn) return;

    clearBtn.addEventListener('click', () => {
        clearRuntimeLog();
    });
}

function initializeModelRefreshButton() {
    const modelCountElement = document.getElementById('model-count');
    if (modelCountElement?.parentElement && !document.getElementById('refresh-model-btn')) {
        const refreshBtn = document.createElement('button');
        refreshBtn.id = 'refresh-model-btn';
        refreshBtn.type = 'button';
        refreshBtn.className = 'btn btn-outline-secondary btn-sm ms-2';
        refreshBtn.innerHTML = '<i class="fas fa-sync"></i>';
        refreshBtn.title = 'Refresh Models';
        refreshBtn.onclick = refreshModels;
        modelCountElement.parentElement.appendChild(refreshBtn);
    }
}

function initializeTooltips() {
    if (!globalThis.bootstrap?.Tooltip) return;

    const tooltipTriggerList = Array.from(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach((tooltipTriggerEl) => {
        globalThis.bootstrap.Tooltip.getOrCreateInstance(tooltipTriggerEl);
    });
}

function restoreClassificationProgress() {
    const restoredProgress = loadClassificationProgress();
    if (!restoredProgress) return;

    if (typeof restoredProgress.percent === 'number') {
        updateProgress(restoredProgress.percent);
    }

    if (restoredProgress.status) {
        updateStatus(restoredProgress.status, restoredProgress.level || 'info');
    }

    if (restoredProgress.stage) {
        const progressStage = document.getElementById('progress-stage');
        if (progressStage) {
            progressStage.textContent = 'Stage: ' + restoredProgress.stage;
        }
    }

    if (restoredProgress.stats) {
        updateProcessingStats(restoredProgress.stats);
        updateClassificationStats(restoredProgress.stats);
    }
}

function restoreClassificationState() {
    const restoredState = loadClassificationState();
    if (!restoredState?.response) return;

    showClassificationResults(restoredState.response, false);
    if (restoredState.response.comparison) {
        showComparisonResults(restoredState.response.comparison, 'comparison-results', 'comparison-metrics');
    }
}

function setupClassificationSocket() {
    const socket = io({
        transports: ['polling'],
        upgrade: false,
    });

    socket.on('connect', () => {
        console.log('Connected to server');
        updateStatus('Connected to server', 'success');
        appendRuntimeLog('Socket connected');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateStatus('Disconnected from server', 'danger');
        appendRuntimeLog('Socket disconnected');
    });

    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        updateStatus('Connection error: ' + error, 'danger');
        appendRuntimeLog('Socket error: ' + error, 'error');
    });

    socket.on('progress_update', (data) => {
        console.log('Received progress update:', data);

        saveClassificationProgress({
            percent: data.percent,
            status: data.status,
            stage: data.stage,
            level: data.level,
            stats: data.stats || null,
            updated_at: new Date().toISOString()
        });

        if (data.percent !== undefined) {
            updateProgress(data.percent);
        }

        const progressStage = document.getElementById('progress-stage');
        if (data.stage && progressStage) {
            progressStage.textContent = 'Stage: ' + data.stage;
        }

        if (data.status) {
            updateStatus(data.status, 'info');
            appendRuntimeLog(data.status, data.level === 'ERROR' ? 'error' : 'info');
        }

        if (data.stats) {
            updateProcessingStats(data.stats);
            updateClassificationStats(data.stats);
        }

        if (data.comparison) {
            showComparisonResults(data.comparison, 'comparison-results', 'comparison-metrics');
        }

        if (data.result) {
            saveClassificationState({
                response: data.result,
                saved_at: new Date().toISOString()
            });
        }
    });
}

function bindClassificationForm() {
    const uploadForm = document.getElementById('upload-form');
    if (!uploadForm) return;

    if (uploadForm.dataset.classifySubmitBound === '1') {
        return;
    }
    uploadForm.dataset.classifySubmitBound = '1';

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        if (globalThis.__classificationSubmitInFlight) {
            updateStatus('Classification is already running. Please wait...', 'warning');
            return;
        }
        globalThis.__classificationSubmitInFlight = true;

        const selectedModel = document.getElementById('model_select')?.value;
        if (!selectedModel) {
            updateStatus('Pilih model LLM terlebih dahulu.', 'warning');
            return;
        }

        const classificationMode = document.getElementById('classification_mode')?.value || 'pure_llm';

        clearClassificationPersistence();
        clearRuntimeLog();
        appendRuntimeLog('Classification request started');
        appendRuntimeLog('Mode: ' + classificationMode);
        updateProgress(0);
        updateStatus('Starting classification...', 'info');

        const processingStats = document.getElementById('processing-stats');
        const resultMessage = document.getElementById('result-message');
        const classificationResults = document.getElementById('classification-results');
        const comparisonResults = document.getElementById('comparison-results');

        if (processingStats) processingStats.innerHTML = '';
        if (resultMessage) resultMessage.innerHTML = '';
        if (classificationResults) classificationResults.style.display = 'none';
        if (comparisonResults) comparisonResults.style.display = 'none';

        const formData = new FormData(this);

        fetch('/classify', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            return data;
        })
        .then(data => {
            console.log('Classification success:', data);
            appendRuntimeLog('Classification completed successfully');
            saveClassificationState({
                response: data,
                saved_at: new Date().toISOString()
            });
            showClassificationResults(data);

            if (data.comparison) {
                showComparisonResults(data.comparison, 'comparison-results', 'comparison-metrics');
            }
        })
        .catch(error => {
            console.error('Classification error:', error);
            updateProgress(0);
            updateStatus('Classification failed!', 'danger');
            appendRuntimeLog('Classification failed: ' + error.message, 'error');

            if (resultMessage) {
                resultMessage.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            }
        })
        .finally(() => {
            globalThis.__classificationSubmitInFlight = false;
        });
    });
}

function bindIndependentComparisonForm() {
    const independentCompareForm = document.getElementById('independent-compare-form');
    if (!independentCompareForm) return;

    independentCompareForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(this);
        const resultsDiv = document.getElementById('independent-comparison-results');
        const metricsDiv = document.getElementById('independent-comparison-metrics');

        if (resultsDiv) resultsDiv.style.display = 'none';
        if (metricsDiv) metricsDiv.innerHTML = '<div class="alert alert-info">Memproses perbandingan...</div>';

        fetch('/compare', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            displayIndependentComparisonResults(data);
        })
        .catch(error => {
            console.error('Comparison error:', error);
            if (metricsDiv) {
                metricsDiv.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        Error: ${error.message}
                    </div>
                `;
            }
            if (resultsDiv) resultsDiv.style.display = 'block';
        });
    });
}

function bindChatSection() {
    const chatForm = document.getElementById('chat-form');
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const userInput = document.getElementById('user_input');
            const user_input = userInput.value.trim();
            if (!user_input) return;

            const formData = new FormData(this);

            saveMessage('user', user_input);
            loadConversationHistory();

            userInput.value = '';
            userInput.focus();

            fetch('/chat', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                saveMessage('assistant', data.response);
                loadConversationHistory();
            })
            .catch(error => {
                console.error('Chat error:', error);
                alert('Error saat mengirim pesan: ' + error.message);
            });
        });
    }

    const clearChatBtn = document.getElementById('clear-chat');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', clearChatHistory);
    }
}

// static/js/classify.js
document.addEventListener('DOMContentLoaded', initializeClassificationPage);

// ===== HELPER FUNCTIONS =====
function updateProgress(percent) {
    const progressBar = document.getElementById('progress-bar');
    if (!progressBar) return;
    
    const progress = Math.min(Math.max(percent, 0), 100);
    progressBar.value = progress;
    progressBar.textContent = progress + '%';
}

function updateStatus(message, type = 'info') {
    const statusElement = document.getElementById('progress-status');
    if (!statusElement) return;
    
    statusElement.textContent = message;
    statusElement.className = `mb-0 text-${type}`;
}

function clearRuntimeLog() {
    const logElement = document.getElementById('runtime-log');
    if (!logElement) return;
    logElement.innerHTML = '';
}

function appendRuntimeLog(message, level = 'info') {
    const logElement = document.getElementById('runtime-log');
    if (!logElement || !message) return;

    const now = new Date();
    const time = now.toLocaleTimeString('id-ID', { hour12: false });
    const line = document.createElement('div');
    const safeMsg = String(message);

    if (level === 'error') {
        line.style.color = '#fca5a5';
    } else if (level === 'warning') {
        line.style.color = '#fde68a';
    } else {
        line.style.color = '#e2e8f0';
    }

    line.textContent = `[${time}] ${safeMsg}`;
    logElement.appendChild(line);

    // Keep last 300 lines only to avoid memory bloat on long runs.
    while (logElement.childNodes.length > 300) {
        logElement.firstChild.remove();
    }

    logElement.scrollTop = logElement.scrollHeight;
}

function updateProcessingStats(stats) {
    if (!stats) return;
    
    const statsElement = document.getElementById('processing-stats');
    if (!statsElement) return;
    
    let statsHtml = `
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">Processing Details</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p>Processed: ${stats.processed || 0} / ${stats.total || 0}</p>
                        <p>Speed: ${stats.speed || '0 texts/sec'}</p>
                    </div>
                    <div class="col-md-6">
                        <p>Time Remaining: ${stats.estimated_remaining || 'calculating...'}</p>
                        <p>Errors: ${stats.errors || 0}</p>
                    </div>
                </div>
                <div class="mt-2">
                    <p><strong>Current Text:</strong><br>${stats.current_text || ''}</p>
                    <p><strong>Last Classification:</strong> ${stats.last_direktorat || '-'} (${stats.last_keyword || '-'})</p>
                </div>
            </div>
        </div>`;
    
    statsElement.innerHTML = statsHtml;
}

function updateClassificationStats(stats) {
    // Implement if needed
}

function showClassificationResults(response, autoDownload = true) {
    updateProgress(100);
    updateStatus('Classification completed!', 'success');
    
    let resultMessage = '<div class="alert alert-success">';
    resultMessage += '<h5>Classification Results:</h5>';
    resultMessage += '<ol>';
    
    if (response.aduan_path) {
        resultMessage += `<li>Aduan Classification: <a href="/download/${response.aduan_path.split('/').pop()}" download>Download Result</a></li>`;
    }
    
    if (response.direktorat_path) {
        resultMessage += `<li>Direktorat Classification: <a href="/download/${response.direktorat_path.split('/').pop()}" download>Download Result</a></li>`;
    }
    
    resultMessage += '</ol>';
    
    // Show comparison note if ground truth was provided
    if (response.comparison) {
        resultMessage += '<p class="mt-2"><strong>Auto comparison with ground truth completed!</strong></p>';
    }
    
    if (response.error) {
        resultMessage += `<p class="text-warning">Note: ${response.error}</p>`;
    }
    
    resultMessage += '</div>';

    if (response.direktorat_path) {
        resultMessage += '<div class="alert alert-info mt-2 mb-0">File hasil direktorat sedang diunduh otomatis. Jika unduhan terblokir browser, gunakan link hasil di bawah.</div>';
    }
    
    const resultMessageDiv = document.getElementById('result-message');
    if (resultMessageDiv) {
        resultMessageDiv.innerHTML = resultMessage;
    }
    
    const classificationResults = document.getElementById('classification-results');
    if (classificationResults) {
        classificationResults.style.display = 'block';
    }
    
    // Show file download links
    let filesHtml = '';
    if (response.aduan_path) {
        filesHtml += `<p><a href="/download/${response.aduan_path.split('/').pop()}" class="btn btn-outline-primary btn-sm" download>Download Aduan Results</a></p>`;
    }
    if (response.direktorat_path) {
        filesHtml += `<p><a href="/download/${response.direktorat_path.split('/').pop()}" class="btn btn-outline-primary btn-sm" download>Download Direktorat Results</a></p>`;
    }
    
    const resultFilesDiv = document.getElementById('result-files');
    if (resultFilesDiv) {
        resultFilesDiv.innerHTML = filesHtml;
    }

    if (response.direktorat_path && autoDownload) {
        globalThis.setTimeout(() => {
            triggerAutoDownload(response.direktorat_path);
        }, 300);
    }
}

function showComparisonResults(comparisonData, resultsContainerId, metricsContainerId) {
    const resultsContainer = document.getElementById(resultsContainerId);
    const metricsContainer = document.getElementById(metricsContainerId);
    
    if (!resultsContainer || !metricsContainer) return;
    
    let metricsHtml = `
        <div class="card mb-4">
            <div class="card-body">
                <h6 class="card-title">Overall Metrics</h6>
                <div class="row">
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-primary">${(comparisonData.accuracy * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Accuracy</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-success">${(comparisonData.precision * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Precision</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-info">${(comparisonData.recall * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Recall</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-warning">${(comparisonData.f1_score * 100).toFixed(2)}%</h4>
                                <p class="mb-0">F1-Score</p>
                            </div>
                        </div>
                    </div>
                </div>
                <p class="mt-3 mb-0"><strong>Total Data:</strong> ${comparisonData.total_data || 0}</p>
                <p class="mb-0"><strong>Correct Predictions:</strong> ${comparisonData.correct_predictions || 0}</p>
                <p class="mb-0"><strong>Incorrect Predictions:</strong> ${comparisonData.incorrect_predictions || 0}</p>
            </div>
        </div>
    `;
    
    metricsContainer.innerHTML = metricsHtml;
    resultsContainer.style.display = 'block';
}

function displayIndependentComparisonResults(data) {
    const resultsDiv = document.getElementById('independent-comparison-results');
    const metricsDiv = document.getElementById('independent-comparison-metrics');
    
    if (!resultsDiv || !metricsDiv) return;
    
    let metricsHtml = `
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Metrik Evaluasi</h6>
                        <table class="table table-sm">
                            <tr><td>Accuracy</td><td>${(data.accuracy * 100).toFixed(2)}%</td></tr>
                            <tr><td>Precision</td><td>${(data.precision * 100).toFixed(2)}%</td></tr>
                            <tr><td>Recall</td><td>${(data.recall * 100).toFixed(2)}%</td></tr>
                            <tr><td>F1-Score</td><td>${(data.f1_score * 100).toFixed(2)}%</td></tr>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Detail</h6>
                        <table class="table table-sm">
                            <tr><td>Total Data</td><td>${data.total_data || 0}</td></tr>
                            <tr><td>Data Sesuai</td><td>${data.correct_predictions || 0}</td></tr>
                            <tr><td>Data Tidak Sesuai</td><td>${data.incorrect_predictions || 0}</td></tr>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    if (data.confusion_matrix) {
        metricsHtml += createConfusionMatrixTable(data.confusion_matrix);
    }
    
    metricsDiv.innerHTML = metricsHtml;
    resultsDiv.style.display = 'block';
}

function createConfusionMatrixTable(confusionMatrix) {
    // Get all unique labels and sort them
    const allLabels = new Set();
    
    // Collect all labels from rows and columns
    Object.keys(confusionMatrix).forEach(rowLabel => {
        allLabels.add(rowLabel);
        Object.keys(confusionMatrix[rowLabel]).forEach(colLabel => {
            allLabels.add(colLabel);
        });
    });
    
    const sortedLabels = Array.from(allLabels).sort((left, right) => left.localeCompare(right));
    
    let html = `
        <div class="mt-4">
            <div class="card">
                <div class="card-header">
                    <h6 class="card-title mb-0">Confusion Matrix</h6>
                </div>
                <div class="card-body">
                    <div class="table-responsive" style="max-height: 500px; overflow: auto;">
                        <table class="table table-bordered table-sm table-hover">
                            <thead class="table-dark sticky-top">
                                <tr>
                                    <th style="min-width: 150px;">Actual \\ Predicted</th>
    `;
    
    // Header row with predicted labels
    sortedLabels.forEach(label => {
        html += `<th class="text-center" style="min-width: 120px;">${label}</th>`;
    });
    html += `<th class="text-center table-info">Total Actual</th>`;
    
    html += `</tr></thead><tbody>`;
    
    // Data rows
    sortedLabels.forEach(actualLabel => {
        html += `<tr>`;
        html += `<th class="table-dark">${actualLabel}</th>`;
        
        let rowTotal = 0;
        sortedLabels.forEach(predictedLabel => {
            const count = confusionMatrix[actualLabel]?.[predictedLabel] || 0;
            rowTotal += count;
            const isDiagonal = actualLabel === predictedLabel;
            let cellClass = '';
            if (isDiagonal) {
                cellClass = 'table-success';
            } else if (count > 0) {
                cellClass = 'table-warning';
            }
            
            html += `<td class="text-center ${cellClass}">${count}</td>`;
        });
        
        // Total for this actual label
        html += `<td class="text-center table-info fw-bold">${rowTotal}</td>`;
        html += `</tr>`;
    });
    
    // Footer row with predicted totals
    html += `<tr><th class="table-info">Total Predicted</th>`;
    sortedLabels.forEach(predictedLabel => {
        let colTotal = 0;
        sortedLabels.forEach(actualLabel => {
            colTotal += confusionMatrix[actualLabel]?.[predictedLabel] || 0;
        });
        html += `<td class="text-center table-info fw-bold">${colTotal}</td>`;
    });
    html += `<td class="text-center table-secondary fw-bold">${Object.values(confusionMatrix).reduce((sum, row) => sum + Object.values(row).reduce((rowSum, cell) => rowSum + cell, 0), 0)}</td>`;
    html += `</tr>`;
    
    html += `</tbody></table></div></div></div></div>`;
    
    return html;
}

function analyzeDataIssues(data) {
    if (!data.detailed_results) return '';
    
    const issues = {
        labelInconsistencies: new Set(),
        uncategorizedHigh: 0,
        totalSamples: data.detailed_results.length
    };
    
    data.detailed_results.forEach(item => {
        // Check for label inconsistencies
        if (item.actual_direktorat !== item.predicted_direktorat) {
            issues.labelInconsistencies.add(`${item.actual_direktorat} -> ${item.predicted_direktorat}`);
        }
        
        // Check for high uncategorized
        if (item.predicted_direktorat === 'Uncategorized') {
            issues.uncategorizedHigh++;
        }
    });
    
    let analysisHtml = `
        <div class="mt-4">
            <div class="card border-warning">
                <div class="card-header bg-warning text-dark">
                    <h6 class="card-title mb-0">Data Quality Analysis</h6>
                </div>
                <div class="card-body">
                    <p><strong>Total Samples:</strong> ${issues.totalSamples}</p>
                    <p><strong>High Uncategorized Predictions:</strong> ${issues.uncategorizedHigh} (${((issues.uncategorizedHigh/issues.totalSamples)*100).toFixed(1)}%)</p>
                    <p><strong>Label Mismatches Found:</strong> ${issues.labelInconsistencies.size}</p>
    `;
    
    if (issues.labelInconsistencies.size > 0) {
        analysisHtml += `<p><strong>Common Mismatches:</strong></p><ul>`;
        Array.from(issues.labelInconsistencies).slice(0, 10).forEach(mismatch => {
            analysisHtml += `<li>${mismatch}</li>`;
        });
        analysisHtml += `</ul>`;
    }
    
    analysisHtml += `</div></div></div>`;
    
    return analysisHtml;
}

// Chat functions
function loadConversationHistory() {
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    const container = document.querySelector('.conversation-container');
    if (!container) return;
    
    container.innerHTML = '';
    history.forEach(msg => {
        const messageClass = msg.role === 'user' ? 'text-primary' : 'text-success';
        const sender = msg.role === 'user' ? 'Pengguna' : 'Asisten';
        container.innerHTML += `<p class="${messageClass}"><strong>${sender}:</strong> ${msg.content}</p>`;
    });
    
    const chatHistory = document.getElementById('chat-history');
    if (chatHistory) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}

function saveMessage(role, content) {
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    history.push({ role, content });
    localStorage.setItem('chatHistory', JSON.stringify(history));
}

function clearChatHistory() {
    if (confirm('Apakah Anda yakin ingin menghapus semua riwayat chat?')) {
        localStorage.removeItem('chatHistory');
        const container = document.querySelector('.conversation-container');
        if (container) {
            container.innerHTML = '';
        }
    }
}