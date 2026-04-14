// static/js/evaluation.js
// Handles all evaluation and comparison functionality

class EvaluationManager {
    constructor() {
        this.socket = null;
        this.initializeSocket();
    }

    initializeSocket() {
        // Use existing socket if available, otherwise create new one
        if (typeof io !== 'undefined') {
            this.socket = io({
                transports: ['polling'],
                upgrade: false,
            });
            this.setupSocketHandlers();
        } else {
            console.warn('Socket.IO not available, progress updates will be limited');
        }
    }

    setupSocketHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            console.log('✅ Evaluation manager connected to server');
        });

        this.socket.on('progress_update', (data) => {
            this.handleProgressUpdate(data);
        });

        this.socket.on('comparison_results', (data) => {
            this.handleComparisonResults(data);
        });
    }

    // Handle progress updates from server
    handleProgressUpdate(data) {
        console.log('📊 Progress update:', data);

        // Update progress bar
        if (data.percent !== undefined) {
            this.updateProgressBar(data.percent);
        }

        // Update status text
        if (data.status) {
            this.updateStatus(data.status, data.level || 'info');
        }

        // Update stage
        if (data.stage) {
            this.updateStage(data.stage);
        }

        // Update statistics
        if (data.stats) {
            this.updateStatistics(data.stats);
        }

        // Handle comparison data
        if (data.comparison) {
            this.displayComparisonResults(data.comparison);
        }
    }

    // Update progress bar
    updateProgressBar(percent) {
        const progressBar = document.getElementById('progress-bar');
        if (progressBar) {
            const progress = Math.min(Math.max(percent, 0), 100);
            progressBar.value = progress;
            progressBar.textContent = progress + '%';
        }
    }

    // Update status message
    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('progress-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `mb-0 text-${type}`;
        }
    }

    // Update current stage
    updateStage(stage) {
        const stageElement = document.getElementById('progress-stage');
        if (stageElement) {
            stageElement.textContent = 'Stage: ' + stage;
        }
    }

    // Update processing statistics
    updateStatistics(stats) {
        const statsElement = document.getElementById('processing-stats');
        if (!statsElement || !stats) return;

        let statsHtml = `
            <div class="card mb-3">
                <div class="card-body">
                    <h6 class="card-title">Processing Details</h6>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Processed:</strong> ${stats.processed || 0} / ${stats.total || 0}</p>
                            <p><strong>Speed:</strong> ${stats.speed || '0 texts/sec'}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Time Remaining:</strong> ${stats.estimated_remaining || 'calculating...'}</p>
                            <p><strong>Errors:</strong> ${stats.errors || 0}</p>
                        </div>
                    </div>
        `;

        if (stats.current_text) {
            statsHtml += `
                <div class="mt-2">
                    <p><strong>Current Text:</strong><br><small class="text-muted">${stats.current_text}</small></p>
                </div>
            `;
        }

        if (stats.last_direktorat) {
            statsHtml += `
                <div class="mt-1">
                    <p><strong>Last Classification:</strong> ${stats.last_direktorat} (${stats.last_keyword || '-'})</p>
                </div>
            `;
        }

        statsHtml += `</div></div>`;
        statsElement.innerHTML = statsHtml;
    }

    // Handle comparison results
    handleComparisonResults(data) {
        console.log('📈 Comparison results:', data);
        this.displayComparisonResults(data);
    }

    // Display comparison results
    displayComparisonResults(comparisonData) {
        const resultsContainer = document.getElementById('comparison-results');
        const metricsContainer = document.getElementById('comparison-metrics');

        if (!resultsContainer || !metricsContainer) {
            console.warn('Comparison results containers not found');
            return;
        }

        let metricsHtml = this.createComparisonMetricsHTML(comparisonData);
        metricsContainer.innerHTML = metricsHtml;
        resultsContainer.style.display = 'block';

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Create comparison metrics HTML
    createComparisonMetricsHTML(data) {
        const overall = data.overall_metrics || {};
        const classMetrics = data.class_metrics || {};
        const confusionMatrix = data.confusion_matrix || [];
        const matchedData = data.matched_data || [];

        return `
            <div class="comparison-results">
                <!-- Overall Metrics -->
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">Overall Evaluation Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-primary">${(overall.accuracy * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Accuracy</strong></p>
                                        <small class="text-muted">Correct predictions</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-success">${(overall.precision * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Precision</strong></p>
                                        <small class="text-muted">Quality of predictions</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-info">${(overall.recall * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Recall</strong></p>
                                        <small class="text-muted">Coverage of actual positives</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-warning">${(overall.f1_score * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>F1-Score</strong></p>
                                        <small class="text-muted">Balance of precision & recall</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <p><strong>Total Samples:</strong> ${overall.total_samples || 0}</p>
                                <p><strong>Matched Records:</strong> ${matchedData.length}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Evaluation Date:</strong> ${new Date().toLocaleString()}</p>
                                <p><strong>Model:</strong> ${data.model_name || 'N/A'}</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Class-wise Metrics -->
                ${this.createClassMetricsHTML(classMetrics)}

                <!-- Confusion Matrix -->
                ${this.createConfusionMatrixHTML(confusionMatrix)}

                <!-- Matched Data Sample -->
                ${this.createMatchedDataHTML(matchedData)}
            </div>
        `;
    }

    // Create class-wise metrics HTML
    createClassMetricsHTML(classMetrics) {
        if (Object.keys(classMetrics).length === 0) {
            return '';
        }

        let rows = '';
        Object.entries(classMetrics).forEach(([className, metrics]) => {
            rows += `
                <tr>
                    <td><strong>${className}</strong></td>
                    <td>${(metrics.precision * 100).toFixed(1)}%</td>
                    <td>${(metrics.recall * 100).toFixed(1)}%</td>
                    <td>${(metrics.f1_score * 100).toFixed(1)}%</td>
                    <td>${metrics.support}</td>
                </tr>
            `;
        });

        return `
            <div class="card mb-4">
                <div class="card-header bg-info text-white">
                    <h5 class="card-title mb-0">Class-wise Performance</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Class</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    // Create confusion matrix HTML
    createConfusionMatrixHTML(confusionMatrix) {
        if (!confusionMatrix || confusionMatrix.length === 0) {
            return '';
        }

        // Extract unique labels
        const labels = [...new Set(confusionMatrix.map(item => item.actual).concat(confusionMatrix.map(item => item.predicted)))].sort();

        // Create matrix data structure
        const matrix = {};
        labels.forEach(actual => {
            matrix[actual] = {};
            labels.forEach(predicted => {
                const item = confusionMatrix.find(c => c.actual === actual && c.predicted === predicted);
                matrix[actual][predicted] = item ? item.count : 0;
            });
        });

        let headerRow = '<tr><th style="min-width: 150px;">Actual \\ Predicted</th>';
        labels.forEach(label => {
            headerRow += `<th class="text-center">${label}</th>`;
        });
        headerRow += '<th class="text-center table-primary">Total</th></tr>';

        let bodyRows = '';
        labels.forEach(actual => {
            let rowTotal = 0;
            bodyRows += `<tr><th class="table-secondary">${actual}</th>`;

            labels.forEach(predicted => {
                const count = matrix[actual][predicted];
                rowTotal += count;
                const isCorrect = actual === predicted;
                const cellClass = isCorrect ? 'table-success' : (count > 0 ? 'table-warning' : '');
                bodyRows += `<td class="text-center ${cellClass}">${count}</td>`;
            });

            bodyRows += `<td class="text-center table-primary fw-bold">${rowTotal}</td></tr>`;
        });

        return `
            <div class="card mb-4">
                <div class="card-header bg-warning text-dark">
                    <h5 class="card-title mb-0">Confusion Matrix</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive" style="max-height: 500px;">
                        <table class="table table-bordered table-sm table-hover">
                            <thead class="table-dark sticky-top">
                                ${headerRow}
                            </thead>
                            <tbody>
                                ${bodyRows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    // Create matched data sample HTML
    createMatchedDataHTML(matchedData) {
        if (!matchedData || matchedData.length === 0) {
            return '';
        }

        const sampleData = matchedData.slice(0, 10); // Show first 10 samples

        let rows = '';
        sampleData.forEach((item, index) => {
            const isCorrect = item.actual_direktorat === item.predicted_direktorat;
            const rowClass = isCorrect ? 'table-success' : 'table-warning';

            rows += `
                <tr class="${rowClass}">
                    <td>${index + 1}</td>
                    <td><small>${item.text.length > 100 ? item.text.substring(0, 100) + '...' : item.text}</small></td>
                    <td>${item.actual_direktorat}</td>
                    <td>${item.predicted_direktorat}</td>
                    <td class="text-center">
                        ${isCorrect ?
                    '<span class="badge bg-success">Correct</span>' :
                    '<span class="badge bg-danger">Incorrect</span>'
                }
                    </td>
                </tr>
            `;
        });

        return `
            <div class="card">
                <div class="card-header bg-secondary text-white">
                    <h5 class="card-title mb-0">Sample Matched Data (${matchedData.length} total)</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive" style="max-height: 400px;">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark sticky-top">
                                <tr>
                                    <th>#</th>
                                    <th>Text</th>
                                    <th>Actual</th>
                                    <th>Predicted</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${rows}
                            </tbody>
                        </table>
                    </div>
                    ${matchedData.length > 10 ?
                `<p class="text-muted mt-2">Showing 10 of ${matchedData.length} matched records</p>` :
                ''
            }
                </div>
            </div>
        `;
    }

    // Initialize evaluation forms
    initializeEvaluationForms() {
        this.initializeClassificationForm();
        this.initializeIndependentComparisonForm();
    }

    // Initialize classification form
    initializeClassificationForm() {
        const form = document.getElementById('upload-form');
        if (!form) return;

        if (form.dataset.evaluationSubmitBound === '1') {
            return;
        }
        form.dataset.evaluationSubmitBound = '1';

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleClassificationSubmit(form);
        });
    }

    // Handle classification form submission
    async handleClassificationSubmit(form) {
        if (globalThis.__classificationSubmitInFlight) {
            this.updateStatus('Classification is already running. Please wait...', 'warning');
            return;
        }
        globalThis.__classificationSubmitInFlight = true;

        // Reset UI
        this.resetEvaluationUI();

        const formData = new FormData(form);

        try {
            this.updateStatus('Starting classification process...', 'info');
            this.updateProgressBar(0);

            const response = await fetch('/classify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            console.log('Classification response:', data);

            this.handleClassificationSuccess(data);

        } catch (error) {
            console.error('Classification error:', error);
            this.handleEvaluationError(error.message);
        } finally {
            globalThis.__classificationSubmitInFlight = false;
        }
    }

    // Handle successful classification
    handleClassificationSuccess(data) {
        this.updateProgressBar(100);
        this.updateStatus('Classification completed successfully!', 'success');

        // Show classification results
        this.showClassificationResults(data);

        // Show comparison results if available
        if (data.comparison) {
            this.displayComparisonResults(data.comparison);
        }
    }

    // Show classification file results
    showClassificationResults(data) {
        const resultMessageDiv = document.getElementById('result-message');
        const classificationResults = document.getElementById('classification-results');
        const resultFilesDiv = document.getElementById('result-files');

        if (!resultMessageDiv || !classificationResults || !resultFilesDiv) return;

        let resultHtml = `
            <div class="alert alert-success">
                <h5>🎉 Classification Complete!</h5>
                <p><strong>Message:</strong> ${data.message || 'Process completed successfully'}</p>
        `;

        if (data.direktorat_path) {
            const filename = data.direktorat_path.split('/').pop();
            resultHtml += `
                <p>
                    <strong>Results File:</strong> 
                    <a href="/download/${filename}" class="btn btn-success btn-sm ms-2" download>
                        <i class="fas fa-download"></i> Download Results
                    </a>
                </p>
            `;
        }

        resultHtml += `</div>`;
        resultMessageDiv.innerHTML = resultHtml;

        // Show file download link
        if (data.direktorat_path) {
            const filename = data.direktorat_path.split('/').pop();
            resultFilesDiv.innerHTML = `
                <div class="d-grid gap-2">
                    <a href="/download/${filename}" class="btn btn-outline-primary" download>
                        <i class="fas fa-file-excel"></i> Download Classification Results
                    </a>
                </div>
            `;
        }

        classificationResults.style.display = 'block';
    }

    // Initialize independent comparison form
    initializeIndependentComparisonForm() {
        const form = document.getElementById('independent-compare-form');
        if (!form) return;

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleIndependentComparisonSubmit(form);
        });
    }

    // Handle independent comparison form submission
    async handleIndependentComparisonSubmit(form) {
        const formData = new FormData(form);
        const resultsDiv = document.getElementById('independent-comparison-results');
        const metricsDiv = document.getElementById('independent-comparison-metrics');

        if (resultsDiv) resultsDiv.style.display = 'none';
        if (metricsDiv) metricsDiv.innerHTML = '<div class="alert alert-info">Comparing datasets...</div>';

        try {
            const response = await fetch('/compare', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.displayIndependentComparisonResults(data);

        } catch (error) {
            console.error('Comparison error:', error);
            if (metricsDiv) {
                metricsDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <h5>❌ Comparison Failed</h5>
                        <p>${error.message}</p>
                    </div>
                `;
            }
            if (resultsDiv) resultsDiv.style.display = 'block';
        }
    }

    // Display independent comparison results
    displayIndependentComparisonResults(data) {
        const resultsDiv = document.getElementById('independent-comparison-results');
        const metricsDiv = document.getElementById('independent-comparison-metrics');

        if (!resultsDiv || !metricsDiv) return;

        metricsDiv.innerHTML = this.createIndependentComparisonHTML(data);
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    // Create HTML for independent comparison results
    createIndependentComparisonHTML(data) {
        return `
            <div class="independent-comparison-results">
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">Dataset Comparison Results</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-primary">${(data.accuracy * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Accuracy</strong></p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-success">${(data.precision * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Precision</strong></p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-info">${(data.recall * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>Recall</strong></p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h3 class="text-warning">${(data.f1_score * 100).toFixed(1)}%</h3>
                                        <p class="mb-0"><strong>F1-Score</strong></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <p><strong>Total Data:</strong> ${data.total_data || 0}</p>
                                <p><strong>Correct Predictions:</strong> ${data.correct_predictions || 0}</p>
                                <p><strong>Incorrect Predictions:</strong> ${data.incorrect_predictions || 0}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Comparison Date:</strong> ${data.comparison_date ? new Date(data.comparison_date).toLocaleString() : new Date().toLocaleString()}</p>
                            </div>
                        </div>
                    </div>
                </div>

                ${data.confusion_matrix ? this.createConfusionMatrixHTML(this.formatConfusionMatrix(data.confusion_matrix)) : ''}
                
                ${data.classification_report ? this.createClassificationReportHTML(data.classification_report) : ''}
            </div>
        `;
    }

    // Format confusion matrix for display
    formatConfusionMatrix(confusionMatrix) {
        const items = [];
        Object.keys(confusionMatrix).forEach(actual => {
            Object.keys(confusionMatrix[actual]).forEach(predicted => {
                items.push({
                    actual: actual,
                    predicted: predicted,
                    count: confusionMatrix[actual][predicted]
                });
            });
        });
        return items;
    }

    // Create classification report HTML
    createClassificationReportHTML(classificationReport) {
        let rows = '';
        Object.entries(classificationReport).forEach(([className, metrics]) => {
            if (typeof metrics === 'object' && metrics.precision !== undefined) {
                rows += `
                    <tr>
                        <td><strong>${className}</strong></td>
                        <td>${(metrics.precision * 100).toFixed(1)}%</td>
                        <td>${(metrics.recall * 100).toFixed(1)}%</td>
                        <td>${(metrics.f1_score * 100).toFixed(1)}%</td>
                        <td>${metrics.support}</td>
                    </tr>
                `;
            }
        });

        return `
            <div class="card mb-4">
                <div class="card-header bg-info text-white">
                    <h5 class="card-title mb-0">Detailed Classification Report</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead class="table-dark">
                                <tr>
                                    <th>Class</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${rows}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    // Reset evaluation UI
    resetEvaluationUI() {
        this.updateProgressBar(0);
        this.updateStatus('Preparing evaluation...', 'info');

        const elementsToReset = [
            'processing-stats',
            'result-message',
            'comparison-metrics',
            'independent-comparison-metrics'
        ];

        elementsToReset.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.innerHTML = '';
        });

        const containersToHide = [
            'classification-results',
            'comparison-results',
            'independent-comparison-results'
        ];

        containersToHide.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    }

    // Handle evaluation error
    handleEvaluationError(errorMessage) {
        this.updateProgressBar(0);
        this.updateStatus('Evaluation failed!', 'danger');

        const resultMessageDiv = document.getElementById('result-message');
        if (resultMessageDiv) {
            resultMessageDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h5>❌ Evaluation Error</h5>
                    <p>${errorMessage}</p>
                    <p class="mb-0"><small>Check the browser console for more details.</small></p>
                </div>
            `;
        }
    }
    // Tambahkan method di EvaluationManager class
    async debugFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/debug-file', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            console.log('🔍 File debug info:', result);
            return result;
        } catch (error) {
            console.error('Debug error:', error);
            return null;
        }
    }

    // Add pre-check before comparison
async preCheckFiles(approvedFile, predictedFile) {
    const formData = new FormData();
    formData.append('approved_file', approvedFile);
    formData.append('predicted_file', predictedFile);

    try {
        const response = await fetch('/api/check-columns', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log('🔍 Column check result:', result);
        return result;
    } catch (error) {
        console.error('Pre-check error:', error);
        return null;
    }
}

// Update handleIndependentComparisonSubmit
async handleIndependentComparisonSubmit(form) {
    const formData = new FormData(form);
    const resultsDiv = document.getElementById('independent-comparison-results');
    const metricsDiv = document.getElementById('independent-comparison-metrics');

    if (resultsDiv) resultsDiv.style.display = 'none';
    if (metricsDiv) metricsDiv.innerHTML = '<div class="alert alert-info">Checking file compatibility...</div>';

    try {
        const approvedFile = formData.get('approved_file');
        const predictedFile = formData.get('predicted_file');
        
        // Pre-check file compatibility
        const preCheck = await this.preCheckFiles(approvedFile, predictedFile);
        
        if (preCheck && !preCheck.compatibility?.can_proceed) {
            let errorHtml = `
                <div class="alert alert-warning">
                    <h5>⚠️ File Compatibility Issue</h5>
                    <p>Required columns not found in uploaded files:</p>
            `;
            
            if (preCheck.compatibility) {
                if (preCheck.compatibility.actual_direktorat_candidates.length === 0) {
                    errorHtml += `<p><strong>Ground Truth File:</strong> No 'actual_direktorat' column found. Available columns: ${preCheck.approved_file?.columns?.join(', ')}</p>`;
                }
                if (preCheck.compatibility.predicted_direktorat_candidates.length === 0) {
                    errorHtml += `<p><strong>Predicted File:</strong> No 'predicted_direktorat' column found. Available columns: ${preCheck.predicted_file?.columns?.join(', ')}</p>`;
                }
            }
            
            errorHtml += `</div>`;
            metricsDiv.innerHTML = errorHtml;
            resultsDiv.style.display = 'block';
            return;
        }

        // Continue with comparison
        metricsDiv.innerHTML = '<div class="alert alert-info">Comparing datasets...</div>';

        const response = await fetch('/compare', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        this.displayIndependentComparisonResults(data);

    } catch (error) {
        console.error('Comparison error:', error);
        if (metricsDiv) {
            metricsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h5>❌ Comparison Failed</h5>
                    <p><strong>Error:</strong> ${error.message}</p>
                </div>
            `;
        }
        if (resultsDiv) resultsDiv.style.display = 'block';
    }
}

    // Update handleIndependentComparisonSubmit
    async handleIndependentComparisonSubmit(form) {
        const formData = new FormData(form);
        const resultsDiv = document.getElementById('independent-comparison-results');
        const metricsDiv = document.getElementById('independent-comparison-metrics');

        if (resultsDiv) resultsDiv.style.display = 'none';
        if (metricsDiv) metricsDiv.innerHTML = '<div class="alert alert-info">Comparing datasets...</div>';

        try {
            // Debug files first
            const approvedFile = formData.get('approved_file');
            const predictedFile = formData.get('predicted_file');

            console.log('🔍 Debugging files...');
            const approvedDebug = await this.debugFile(approvedFile);
            const predictedDebug = await this.debugFile(predictedFile);

            console.log('Approved file debug:', approvedDebug);
            console.log('Predicted file debug:', predictedDebug);

            const response = await fetch('/compare', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.displayIndependentComparisonResults(data);

        } catch (error) {
            console.error('Comparison error:', error);
            if (metricsDiv) {
                metricsDiv.innerHTML = `
                <div class="alert alert-danger">
                    <h5>❌ Comparison Failed</h5>
                    <p><strong>Error:</strong> ${error.message}</p>
                    <p class="mb-0"><small>Check browser console for file debug information.</small></p>
                </div>
            `;
            }
            if (resultsDiv) resultsDiv.style.display = 'block';
        }
    }
}

// Initialize evaluation manager when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    window.evaluationManager = new EvaluationManager();
    window.evaluationManager.initializeEvaluationForms();
    console.log('✅ Evaluation manager initialized');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EvaluationManager;
}