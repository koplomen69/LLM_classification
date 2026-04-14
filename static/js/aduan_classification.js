// static/js/aduan_classification.js - WITH EVALUATION
document.addEventListener('DOMContentLoaded', function() {
    console.log('Aduan Classification module loaded');
    
    // Socket connection untuk progress update
    const socket = io({
        transports: ['polling'],
        upgrade: false,
    });

    // Progress update handler khusus aduan classification
    socket.on('aduan_progress_update', (data) => {
        console.log('Received aduan progress update:', data);
        
        if (data.percent !== undefined) {
            updateAduanProgress(data.percent);
        }
        
        if (data.stage) {
            document.getElementById('aduan-progress-stage').textContent = 'Stage: ' + data.stage;
        }
        
        if (data.status) {
            updateAduanStatus(data.status, data.level || 'info');
        }
        
        if (data.stats) {
            updateAduanProcessingStats(data.stats);
        }

        // Handle evaluation results
        if (data.evaluation) {
            showAduanEvaluationResults(data.evaluation);
        }
    });

    // ===== ADUAN CLASSIFICATION FORM =====
    const aduanForm = document.getElementById('aduan-classification-form');
    if (aduanForm) {
        aduanForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Reset progress and results
            resetAduanProgress();
            
            const submitBtn = document.getElementById('aduan-classify-btn');
            const spinner = submitBtn.querySelector('.spinner-border');
            
            // Show loading state
            spinner.classList.remove('d-none');
            submitBtn.disabled = true;
            
            const formData = new FormData(this);
            
            console.log('Submitting aduan classification form...');
            
            fetch('/classify-aduan', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Aduan classification success:', data);
                showAduanClassificationResults(data);
                
                // Show evaluation results if available
                if (data.evaluation) {
                    showAduanEvaluationResults(data.evaluation);
                }
            })
            .catch(error => {
                console.error('Aduan classification error:', error);
                updateAduanProgress(0);
                updateAduanStatus('Klasifikasi aduan gagal!', 'danger');
                document.getElementById('aduan-result-message').innerHTML = 
                    `<div class="alert alert-danger">Error: ${error.message}</div>`;
            })
            .finally(() => {
                spinner.classList.add('d-none');
                submitBtn.disabled = false;
            });
        });
    }

    // ===== QUICK TEST FORM =====
    const quickTestForm = document.getElementById('quick-test-form');
    if (quickTestForm) {
        quickTestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const testText = document.getElementById('test_text').value.trim();
            const selectedModel = document.getElementById('aduan_model_type')?.value || 'svm_keyword';
            if (!testText) {
                alert('Masukkan teks untuk test!');
                return;
            }
            
            const resultsDiv = document.getElementById('quick-test-results');
            const outputDiv = document.getElementById('quick-test-output');
            
            // Show loading
            outputDiv.innerHTML = '<div class="alert alert-info">Memproses...</div>';
            resultsDiv.style.display = 'block';
            
            fetch('/quick-test-aduan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: testText, model_type: selectedModel })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                displayQuickTestResults(data);
            })
            .catch(error => {
                console.error('Quick test error:', error);
                outputDiv.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        Error: ${error.message}
                    </div>
                `;
            });
        });
    }

    // ===== MANUAL EVALUATION FORM =====
    const manualEvalForm = document.getElementById('manual-evaluation-form');
    if (manualEvalForm) {
        manualEvalForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const resultsDiv = document.getElementById('manual-evaluation-results');
            const metricsDiv = document.getElementById('detailed-evaluation-metrics');
            
            // Show loading
            metricsDiv.innerHTML = `
                <div class="alert alert-info">
                    <div class="spinner-border spinner-border-sm me-2" role="status"></div>
                    Menjalankan evaluasi mendalam...
                </div>
            `;
            resultsDiv.style.display = 'block';
            
            fetch('/evaluate-aduan-model', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                displayDetailedEvaluationResults(data);
            })
            .catch(error => {
                console.error('Manual evaluation error:', error);
                metricsDiv.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        Error: ${error.message}
                    </div>
                `;
            });
        });
    }
});

// ===== ADUAN-SPECIFIC HELPER FUNCTIONS =====
function resetAduanProgress() {
    updateAduanProgress(0);
    updateAduanStatus('Memulai klasifikasi...', 'info');
    document.getElementById('aduan-processing-stats').innerHTML = '';
    document.getElementById('aduan-result-message').innerHTML = '';
    document.getElementById('aduan-classification-results').style.display = 'none';
    document.getElementById('aduan-evaluation-results').style.display = 'none';
    document.getElementById('aduan-statistics').innerHTML = '';
}

function updateAduanProgress(percent) {
    const progressBar = document.getElementById('aduan-progress-bar');
    if (!progressBar) return;
    
    const progress = Math.min(Math.max(percent, 0), 100);
    progressBar.style.width = progress + '%';
    progressBar.setAttribute('aria-valuenow', progress);
    progressBar.textContent = progress + '%';
}

function updateAduanStatus(message, type = 'info') {
    const statusElement = document.getElementById('aduan-progress-status');
    if (!statusElement) return;
    
    statusElement.textContent = message;
    statusElement.className = `mb-0 text-${type}`;
}

function updateAduanProcessingStats(stats) {
    if (!stats) return;
    
    const statsElement = document.getElementById('aduan-processing-stats');
    if (!statsElement) return;
    
    let statsHtml = `
        <div class="card mb-3">
            <div class="card-body">
                <h6 class="card-title">Detail Pemrosesan</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Diproses:</strong> ${stats.processed || 0} / ${stats.total || 0}</p>
                        <p><strong>Kecepatan:</strong> ${stats.speed || '0 texts/sec'}</p>
                        <p><strong>Waktu Tersisa:</strong> ${stats.estimated_remaining || 'menghitung...'}</p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Aduan:</strong> ${stats.aduan_count || 0}</p>
                        <p><strong>Bukan Aduan:</strong> ${stats.non_aduan_count || 0}</p>
                        <p><strong>Perlu Review:</strong> ${stats.review_count || 0}</p>
                    </div>
                </div>
                ${stats.current_text ? `
                <div class="mt-2">
                    <p><strong>Teks Saat Ini:</strong><br><small>${stats.current_text}</small></p>
                    ${stats.current_classification ? `<p><strong>Klasifikasi:</strong> ${stats.current_classification}</p>` : ''}
                    ${stats.current_score ? `<p><strong>Skor:</strong> ${stats.current_score}</p>` : ''}
                </div>
                ` : ''}
            </div>
        </div>`;
    
    statsElement.innerHTML = statsHtml;
}

function showAduanClassificationResults(response) {
    updateAduanProgress(100);
    updateAduanStatus('Klasifikasi aduan selesai!', 'success');
    
    let resultMessage = '<div class="alert alert-success">';
    resultMessage += '<h5>Hasil Klasifikasi Aduan:</h5>';
    
    if (response.result_path) {
        resultMessage += `<p>File hasil: <a href="/download/${response.result_path.split('/').pop()}" download class="btn btn-outline-primary btn-sm">Download Hasil</a></p>`;
    }
    
    if (response.statistics) {
        resultMessage += `<p><strong>Statistik:</strong> ${response.statistics.total} teks diproses</p>`;
    }
    
    resultMessage += '</div>';
    
    const resultMessageDiv = document.getElementById('aduan-result-message');
    if (resultMessageDiv) {
        resultMessageDiv.innerHTML = resultMessage;
    }
    
    // Show file download links
    let filesHtml = '';
    if (response.result_path) {
        filesHtml += `<a href="/download/${response.result_path.split('/').pop()}" class="btn btn-primary btn-sm" download>Download Hasil Lengkap</a>`;
    }
    
    const resultFilesDiv = document.getElementById('aduan-result-files');
    if (resultFilesDiv) {
        resultFilesDiv.innerHTML = filesHtml;
    }
    
    // Show statistics
    if (response.statistics) {
        const stats = response.statistics;
        let statsHtml = `
            <div class="row text-center">
                <div class="col-4">
                    <div class="card bg-success text-white">
                        <div class="card-body p-2">
                            <h4>${stats.aduan_count || 0}</h4>
                            <small>Aduan</small>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-danger text-white">
                        <div class="card-body p-2">
                            <h4>${stats.non_aduan_count || 0}</h4>
                            <small>Bukan Aduan</small>
                        </div>
                    </div>
                </div>
                <div class="col-4">
                    <div class="card bg-warning text-dark">
                        <div class="card-body p-2">
                            <h4>${stats.review_count || 0}</h4>
                            <small>Perlu Review</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="mt-2">
                <p class="mb-1"><strong>Total:</strong> ${stats.total || 0} teks</p>
                <p class="mb-0"><strong>Persentase Aduan:</strong> ${stats.aduan_percentage || 0}%</p>
            </div>
        `;
        
        const statsDiv = document.getElementById('aduan-statistics');
        if (statsDiv) {
            statsDiv.innerHTML = statsHtml;
        }
    }
    
    const classificationResults = document.getElementById('aduan-classification-results');
    if (classificationResults) {
        classificationResults.style.display = 'block';
    }
}

function showAduanEvaluationResults(evaluationData) {
    const resultsContainer = document.getElementById('aduan-evaluation-results');
    const metricsContainer = document.getElementById('aduan-evaluation-metrics');
    
    if (!resultsContainer || !metricsContainer) return;
    
    let metricsHtml = createEvaluationMetrics(evaluationData);
    metricsContainer.innerHTML = metricsHtml;
    resultsContainer.style.display = 'block';
}

function displayDetailedEvaluationResults(data) {
    const resultsDiv = document.getElementById('manual-evaluation-results');
    const metricsDiv = document.getElementById('detailed-evaluation-metrics');
    
    if (!resultsDiv || !metricsDiv) return;
    
    let metricsHtml = createDetailedEvaluationMetrics(data);
    metricsDiv.innerHTML = metricsHtml;
    resultsDiv.style.display = 'block';
}

function createEvaluationMetrics(evaluationData) {
    return `
        <div class="card mb-4">
            <div class="card-body">
                <h6 class="card-title">Metrik Evaluasi</h6>
                <div class="row">
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-primary">${(evaluationData.accuracy * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Accuracy</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-success">${(evaluationData.precision * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Precision</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-info">${(evaluationData.recall * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Recall</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-warning">${(evaluationData.f1_score * 100).toFixed(2)}%</h4>
                                <p class="mb-0">F1-Score</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mt-3">
                    <p class="mb-1"><strong>Total Data:</strong> ${evaluationData.total_data || 0}</p>
                    <p class="mb-1"><strong>Prediksi Benar:</strong> ${evaluationData.correct_predictions || 0}</p>
                    <p class="mb-1"><strong>Prediksi Salah:</strong> ${evaluationData.incorrect_predictions || 0}</p>
                </div>
            </div>
        </div>
        
        ${evaluationData.confusion_matrix ? createAduanConfusionMatrix(evaluationData.confusion_matrix) : ''}
    `;
}

function createDetailedEvaluationMetrics(data) {
    let html = `
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h6 class="card-title mb-0">Hasil Evaluasi Mendalam</h6>
            </div>
            <div class="card-body">
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-primary">${(data.accuracy * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Accuracy</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-success">${(data.precision * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Precision</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-info">${(data.recall * 100).toFixed(2)}%</h4>
                                <p class="mb-0">Recall</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h4 class="text-warning">${(data.f1_score * 100).toFixed(2)}%</h4>
                                <p class="mb-0">F1-Score</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">Statistik Dataset</h6>
                                <table class="table table-sm">
                                    <tr><td>Total Samples</td><td>${data.total_data || 0}</td></tr>
                                    <tr><td>Correct Predictions</td><td>${data.correct_predictions || 0}</td></tr>
                                    <tr><td>Incorrect Predictions</td><td>${data.incorrect_predictions || 0}</td></tr>
                                    <tr><td>False Positives</td><td>${data.false_positives || 0}</td></tr>
                                    <tr><td>False Negatives</td><td>${data.false_negatives || 0}</td></tr>
                                </table>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">Model Performance</h6>
                                <table class="table table-sm">
                                    <tr><td>Model Type</td><td>${data.model_type || 'N/A'}</td></tr>
                                    <tr><td>Processing Time</td><td>${data.processing_time || 'N/A'}</td></tr>
                                    <tr><td>Evaluation Date</td><td>${data.evaluation_date || 'N/A'}</td></tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add confusion matrix
    if (data.confusion_matrix) {
        html += createAduanConfusionMatrix(data.confusion_matrix);
    }
    
    // Add classification report
    if (data.classification_report) {
        html += createClassificationReport(data.classification_report);
    }
    
    // Add false positives analysis
    if (data.false_positives_analysis) {
        html += createFalsePositivesAnalysis(data.false_positives_analysis);
    }
    
    return html;
}

function createAduanConfusionMatrix(confusionMatrix) {
    return `
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="card-title mb-0">Confusion Matrix</h6>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-bordered table-sm">
                        <thead class="table-dark">
                            <tr>
                                <th>Aktual \\ Prediksi</th>
                                <th class="text-center">Bukan Aduan</th>
                                <th class="text-center">Aduan</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th class="table-dark">Bukan Aduan</th>
                                <td class="text-center table-success">${confusionMatrix.true_negative || 0}</td>
                                <td class="text-center table-danger">${confusionMatrix.false_positive || 0}</td>
                            </tr>
                            <tr>
                                <th class="table-dark">Aduan</th>
                                <td class="text-center table-danger">${confusionMatrix.false_negative || 0}</td>
                                <td class="text-center table-success">${confusionMatrix.true_positive || 0}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

function createClassificationReport(classificationReport) {
    let html = `
        <div class="card mb-3">
            <div class="card-header">
                <h6 class="card-title mb-0">Classification Report</h6>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-bordered table-sm">
                        <thead class="table-dark">
                            <tr>
                                <th>Class</th>
                                <th class="text-center">Precision</th>
                                <th class="text-center">Recall</th>
                                <th class="text-center">F1-Score</th>
                                <th class="text-center">Support</th>
                            </tr>
                        </thead>
                        <tbody>
    `;
    
    for (const [className, metrics] of Object.entries(classificationReport)) {
        html += `
            <tr>
                <th>${className}</th>
                <td class="text-center">${(metrics.precision * 100).toFixed(2)}%</td>
                <td class="text-center">${(metrics.recall * 100).toFixed(2)}%</td>
                <td class="text-center">${(metrics.f1_score * 100).toFixed(2)}%</td>
                <td class="text-center">${metrics.support}</td>
            </tr>
        `;
    }
    
    html += `</tbody></table></div></div></div>`;
    return html;
}

function createFalsePositivesAnalysis(analysis) {
    let html = `
        <div class="card border-warning">
            <div class="card-header bg-warning text-dark">
                <h6 class="card-title mb-0">False Positives Analysis</h6>
            </div>
            <div class="card-body">
                <p><strong>Total False Positives:</strong> ${analysis.total_false_positives || 0}</p>
    `;
    
    if (analysis.top_triggering_keywords) {
        html += `<p><strong>Top Triggering Keywords:</strong></p><ul>`;
        for (const [keyword, count] of Object.entries(analysis.top_triggering_keywords)) {
            html += `<li>${keyword}: ${count} occurrences</li>`;
        }
        html += `</ul>`;
    }
    
    if (analysis.average_rule_score !== undefined) {
        html += `<p><strong>Average Rule Score:</strong> ${analysis.average_rule_score.toFixed(2)}</p>`;
    }
    
    if (analysis.sample_false_positives) {
        html += `<p><strong>Sample False Positives:</strong></p>`;
        analysis.sample_false_positives.forEach((fp, index) => {
            html += `<div class="border p-2 mb-2">
                <strong>${index + 1}.</strong> ${fp.text}<br>
                <small class="text-muted">Score: ${fp.rule_score.toFixed(2)}, Triggers: ${fp.triggering_keywords.join(', ')}</small>
            </div>`;
        });
    }
    
    html += `</div></div>`;
    return html;
}

function displayQuickTestResults(data) {
    const outputDiv = document.getElementById('quick-test-output');
    const classification = data.classification;
    const score = data.score || 0;
    const mlProb = data.ml_probability || 0;
    
    let badgeClass = 'bg-secondary';
    let badgeText = 'Unknown';
    
    switch(classification) {
        case 'aduan_text':
            badgeClass = 'bg-success';
            badgeText = 'Aduan Text';
            break;
        case 'not_aduan_text':
        case 'bukan_aduan':
            badgeClass = 'bg-danger';
            badgeText = 'Bukan Aduan';
            break;
        case 'review_needed':
            badgeClass = 'bg-warning text-dark';
            badgeText = 'Perlu Review';
            break;
    }
    
    let outputHtml = `
        <div class="d-flex justify-content-between align-items-center mb-2">
            <h6 class="mb-0">Hasil Klasifikasi:</h6>
            <span class="badge ${badgeClass}">${badgeText}</span>
        </div>
        <div class="row">
            <div class="col-md-6">
                <p><strong>Skor Rule-Based:</strong> ${score.toFixed(2)}</p>
            </div>
            <div class="col-md-6">
                <p><strong>Probabilitas ML:</strong> ${mlProb ? (mlProb * 100).toFixed(2) + '%' : 'N/A'}</p>
            </div>
        </div>
        <div class="mt-2">
            <p><strong>Analisis:</strong></p>
            <p class="small">${data.analysis || 'Tidak ada analisis tambahan'}</p>
        </div>
    `;
    
    outputDiv.innerHTML = outputHtml;
}