let currentConfigId = null;

// Load configurations list
function loadConfigurations() {
    fetch('/api/prompt-configs')
        .then(response => response.json())
        .then(configs => {
            const container = document.getElementById('config-list');
            container.innerHTML = '';

            configs.forEach(config => {
                const badge = config.is_active ?
                    '<span class="badge bg-success ms-2">Active</span>' : '';

                const item = document.createElement('a');
                item.href = '#';
                item.className = `list-group-item list-group-item-action ${config.is_active ? 'active' : ''}`;
                item.dataset.configId = config.id; // Tambahkan ini
                item.innerHTML = `
                    <div class="d-flex w-100 justify-content-between">
                        <h6 class="mb-1">${config.name} ${badge}</h6>
                        <small>${new Date(config.created_at).toLocaleDateString()}</small>
                    </div>
                    <p class="mb-1 small text-muted">${config.description || 'No description'}</p>
                    <small>${config.component_count} components</small>
                    <button class="btn btn-outline-danger btn-sm mt-2 delete-list-btn" 
                            data-config-id="${config.id}" 
                            data-config-name="${config.name}">
                        <i class="fas fa-trash"></i> Delete
                    </button>
                `;
                container.appendChild(item);
            });

            // Setup event delegation untuk list
            setupListEventDelegation();
        })
        .catch(error => {
            console.error('Error loading configurations:', error);
            alert('Error loading configurations: ' + error.message);
        });
}

// Load configuration editor
function loadConfigEditor(configId) {
    currentConfigId = configId;

    fetch(`/api/prompt-configs/${configId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('welcome-message').style.display = 'none';

            const editor = document.getElementById('config-editor');
            editor.innerHTML = `
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <div>
                            <h4 class="mb-0">${data.config.name}</h4>
                            <p class="mb-0 text-muted">${data.config.description || 'No description'}</p>
                        </div>
                        <div>
                            ${!data.config.is_active ?
                    `<button class="btn btn-success btn-sm activate-config-btn" data-config-id="${configId}">
                                <i class="fas fa-play"></i> Activate
                            </button>` :
                    '<span class="badge bg-success">Active</span>'
                }
                            <button class="btn btn-outline-primary btn-sm ms-2 test-config-btn" data-config-id="${configId}">
                                <i class="fas fa-vial"></i> Test
                            </button>
                            <button class="btn btn-outline-info btn-sm ms-2 preview-config-btn" data-config-id="${configId}">
                                <i class="fas fa-eye"></i> Preview
                            </button>
                            <button class="btn btn-outline-secondary btn-sm ms-2 settings-config-btn" data-config-id="${configId}">
                                <i class="fas fa-cog"></i> Settings
                            </button>
                            <button class="btn btn-outline-danger btn-sm ms-2 delete-config-btn" data-config-id="${configId}">
                                <i class="fas fa-trash"></i> Delete
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> 
                            Edit the components below to customize your prompt. Changes are saved automatically.
                        </div>
                        
                        <div class="components-container">
                            ${data.components.map(comp => `
                                <div class="card mb-3 component-card" data-component-id="${comp.id}">
                                    <div class="card-header d-flex justify-content-between align-items-center">
                                        <div class="form-check form-switch">
                                            <input class="form-check-input component-toggle" 
                                                   type="checkbox" 
                                                   data-component-id="${comp.id}"
                                                   ${comp.is_enabled ? 'checked' : ''}>
                                            <label class="form-check-label fw-bold">${comp.name}</label>
                                        </div>
                                        <button class="btn btn-outline-secondary btn-sm edit-component-btn" data-component-id="${comp.id}">
                                            <i class="fas fa-edit"></i> Edit
                                        </button>
                                    </div>
                                    <div class="card-body">
                                        <div class="component-content-preview">
                                            <pre class="text-muted" style="white-space: pre-wrap; max-height: 150px; overflow-y: auto; font-size: 0.8rem;">${comp.content}</pre>
                                        </div>
                                        <div class="mt-2 text-end">
                                            <small class="text-muted">
                                                ${comp.content.length} characters
                                            </small>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
            editor.style.display = 'block';

            // Setup event delegation untuk editor
            setupEditorEventDelegation();
        })
        .catch(error => {
            console.error('Error loading configuration:', error);
            alert('Error loading configuration: ' + error.message);
        });
}

// Edit component in modal
function editComponentInModal(componentId) {
    fetch(`/api/prompt-configs/${currentConfigId}`)
        .then(response => response.json())
        .then(data => {
            const component = data.components.find(comp => comp.id === componentId);
            if (component) {
                document.getElementById('edit-component-id').value = component.id;
                document.getElementById('edit-component-name').value = component.name;
                document.getElementById('edit-component-content').value = component.content;
                document.getElementById('edit-component-enabled').checked = component.is_enabled;

                const modal = new bootstrap.Modal(document.getElementById('editComponentModal'));
                modal.show();
            }
        });
}

// Save component changes from modal
function saveComponentChanges() {
    const componentId = document.getElementById('edit-component-id').value;
    const content = document.getElementById('edit-component-content').value;
    const isEnabled = document.getElementById('edit-component-enabled').checked;

    fetch(`/api/prompt-components/${componentId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            content: content,
            is_enabled: isEnabled
        })
    })
        .then(response => response.json())
        .then(data => {
            bootstrap.Modal.getInstance(document.getElementById('editComponentModal')).hide();
            loadConfigEditor(currentConfigId);
            showNotification('Component updated successfully!', 'success');
        })
        .catch(error => {
            console.error('Error updating component:', error);
            showNotification('Error updating component: ' + error.message, 'error');
        });
}

// Toggle component
function toggleComponent(componentId, isEnabled) {
    fetch(`/api/prompt-components/${componentId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_enabled: isEnabled })
    })
        .then(response => response.json())
        .then(data => {
            console.log('Component toggled:', data);
            showNotification('Component updated!', 'success');
        })
        .catch(error => {
            console.error('Error toggling component:', error);
            showNotification('Error updating component: ' + error.message, 'error');
        });
}

function activateConfig(configId) {
    // Show a non-intrusive notification instead of confirm
    showNotification('Activating configuration...', 'info');
    
    fetch(`/api/prompt-configs/${configId}/activate`, {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            showNotification('Configuration activated!', 'success');
            loadConfigurations();
            if (currentConfigId === configId) {
                loadConfigEditor(configId);
            }
        })
        .catch(error => {
            console.error('Error activating configuration:', error);
            showNotification('Error activating configuration: ' + error.message, 'error');
        });
}

// Test configuration
function testConfiguration(configId) {
    const text = prompt('Enter text to test classification:', 'telyu krs saya error tidak bisa daftar mata kuliah');
    if (!text) return;

    fetch('/api/test-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, config_id: configId })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showTestResults(data);
            } else {
                showNotification('Error: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error testing prompt:', error);
            showNotification('Error testing prompt: ' + error.message, 'error');
        });
}

// Show test results
function showTestResults(data) {
    const modal = new bootstrap.Modal(document.getElementById('testResultsModal'));
    document.getElementById('test-input-text').textContent = data.input_text;
    document.getElementById('test-parsed-result').textContent =
        `Direktorat: ${data.parsed_result.direktorat}, Keyword: ${data.parsed_result.keyword}`;
    document.getElementById('test-prompt-preview').textContent = data.generated_prompt;
    document.getElementById('test-llm-output').textContent = data.llm_output;

    modal.show();
}

// Show quick preview
function showQuickPreview(configId) {
    const modal = new bootstrap.Modal(document.getElementById('quickPreviewModal'));
    document.getElementById('quickPreviewModal').setAttribute('data-config-id', configId);
    updatePromptPreview();
    modal.show();
}

// Update prompt preview
function updatePromptPreview() {
    const modal = document.getElementById('quickPreviewModal');
    const configId = modal.getAttribute('data-config-id');
    const text = document.getElementById('preview-text-input').value;

    if (!text) {
        showNotification('Please enter some text for preview', 'error');
        return;
    }

    fetch('/api/build-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, config_id: configId })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('prompt-preview-content').textContent = data.prompt;
            document.getElementById('prompt-length').textContent = data.length;
        })
        .catch(error => {
            console.error('Error building prompt preview:', error);
            showNotification('Error building preview: ' + error.message, 'error');
        });
}

// Show config settings
function showConfigSettings(configId) {
    fetch(`/api/prompt-configs/${configId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('config-settings-id').value = data.config.id;
            document.getElementById('config-settings-name').value = data.config.name;
            document.getElementById('config-settings-description').value = data.config.description || '';
            document.getElementById('config-settings-active').checked = data.config.is_active;

            const modal = new bootstrap.Modal(document.getElementById('configSettingsModal'));
            modal.show();
        });
}

// Save config settings
function saveConfigSettings() {
    const configId = document.getElementById('config-settings-id').value;
    const name = document.getElementById('config-settings-name').value;
    const description = document.getElementById('config-settings-description').value;

    if (!name.trim()) {
        showNotification('Configuration name cannot be empty!', 'error');
        return;
    }

    fetch(`/api/prompt-configs/${configId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
            name: name,
            description: description
        })
    })
        .then(async response => {
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            showNotification('Configuration settings updated successfully!', 'success');
            const modal = bootstrap.Modal.getInstance(document.getElementById('configSettingsModal'));
            if (modal) {
                modal.hide();
            }

            loadConfigurations();
            if (currentConfigId === parseInt(configId)) {
                setTimeout(() => loadConfigEditor(currentConfigId), 500);
            }
        })
        .catch(error => {
            console.error('Error saving config settings:', error);
            showNotification(`Error saving configuration: ${error.message}`, 'error');
        });
}

function showNotification(message, type = 'info') {
    const existingNotifications = document.querySelectorAll('.custom-notification');
    existingNotifications.forEach(notif => notif.remove());

    const alertClass = type === 'error' ? 'alert-danger' :
        type === 'success' ? 'alert-success' : 'alert-info';

    const icon = type === 'error' ? 'fa-exclamation-circle' :
        type === 'success' ? 'fa-check-circle' : 'fa-info-circle';

    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show custom-notification position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px; max-width: 500px;';
    notification.innerHTML = `
        <i class="fas ${icon} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Create new configuration
function showCreateConfigModal() {
    fetch('/api/prompt-configs')
        .then(response => response.json())
        .then(configs => {
            const select = document.getElementById('copy-from-select');
            select.innerHTML = '<option value="">Create from scratch</option>' +
                configs.map(config =>
                    `<option value="${config.id}">${config.name}</option>`
                ).join('');

            const modal = new bootstrap.Modal(document.getElementById('createConfigModal'));
            modal.show();
        })
        .catch(error => {
            console.error('Error loading configurations:', error);
            showNotification('Error loading configurations: ' + error.message, 'error');
        });
}

function createNewConfiguration() {
    const name = document.getElementById('new-config-name').value;
    const description = document.getElementById('new-config-description').value;
    const copyFrom = document.getElementById('copy-from-select').value;

    if (!name) {
        showNotification('Please enter a name for the configuration', 'error');
        return;
    }

    fetch('/api/prompt-configs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: name,
            description: description,
            copy_from: copyFrom || null
        })
    })
        .then(response => response.json())
        .then(data => {
            bootstrap.Modal.getInstance(document.getElementById('createConfigModal')).hide();
            loadConfigurations();
            loadConfigEditor(data.id);
            showNotification('Configuration created successfully!', 'success');
        })
        .catch(error => {
            console.error('Error creating configuration:', error);
            showNotification('Error creating configuration: ' + error.message, 'error');
        });
}

function showDeleteConfigModal(configId, configName) {
    document.getElementById('delete-config-id').value = configId;
    document.getElementById('delete-config-name').textContent = configName;
    
    const modal = new bootstrap.Modal(document.getElementById('deleteConfigModal'));
    modal.show();
}

function confirmDeleteConfig() {
    const configId = document.getElementById('delete-config-id').value;
    
    fetch(`/api/prompt-configs/${configId}`, {
        method: 'DELETE',
        headers: { 
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(async response => {
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        showNotification(data.message, 'success');
        const modal = bootstrap.Modal.getInstance(document.getElementById('deleteConfigModal'));
        if (modal) {
            modal.hide();
        }
        
        loadConfigurations();
        document.getElementById('config-editor').style.display = 'none';
        document.getElementById('welcome-message').style.display = 'block';
        currentConfigId = null;
    })
    .catch(error => {
        console.error('Error deleting config:', error);
        showNotification(`Error deleting configuration: ${error.message}`, 'error');
    });
}

function setupEditorEventDelegation() {
    const editor = document.getElementById('config-editor');
    
    editor.addEventListener('change', function(e) {
        if (e.target.classList.contains('component-toggle')) {
            const componentId = parseInt(e.target.dataset.componentId);
            const isEnabled = e.target.checked;
            toggleComponent(componentId, isEnabled);
        }
    });
    
    editor.addEventListener('click', function(e) {
        if (e.target.classList.contains('edit-component-btn') || e.target.closest('.edit-component-btn')) {
            const btn = e.target.classList.contains('edit-component-btn') ? e.target : e.target.closest('.edit-component-btn');
            const componentId = parseInt(btn.dataset.componentId);
            editComponentInModal(componentId);
        }
        else if (e.target.classList.contains('activate-config-btn') || e.target.closest('.activate-config-btn')) {
            const btn = e.target.classList.contains('activate-config-btn') ? e.target : e.target.closest('.activate-config-btn');
            const configId = parseInt(btn.dataset.configId);
            activateConfig(configId);
        }
        else if (e.target.classList.contains('test-config-btn') || e.target.closest('.test-config-btn')) {
            const btn = e.target.classList.contains('test-config-btn') ? e.target : e.target.closest('.test-config-btn');
            const configId = parseInt(btn.dataset.configId);
            testConfiguration(configId);
        }
        else if (e.target.classList.contains('preview-config-btn') || e.target.closest('.preview-config-btn')) {
            const btn = e.target.classList.contains('preview-config-btn') ? e.target : e.target.closest('.preview-config-btn');
            const configId = parseInt(btn.dataset.configId);
            showQuickPreview(configId);
        }
        else if (e.target.classList.contains('settings-config-btn') || e.target.closest('.settings-config-btn')) {
            const btn = e.target.classList.contains('settings-config-btn') ? e.target : e.target.closest('.settings-config-btn');
            const configId = parseInt(btn.dataset.configId);
            showConfigSettings(configId);
        }
        else if (e.target.classList.contains('delete-config-btn') || e.target.closest('.delete-config-btn')) {
            const btn = e.target.classList.contains('delete-config-btn') ? e.target : e.target.closest('.delete-config-btn');
            const configId = parseInt(btn.dataset.configId);
            const configName = btn.closest('.card').querySelector('h4').textContent;
            showDeleteConfigModal(configId, configName);
        }
    });
}

function setupListEventDelegation() {
    const container = document.getElementById('config-list');
    
    container.addEventListener('click', function(e) {
        const listItem = e.target.closest('.list-group-item');
        if (listItem && listItem.dataset.configId) {
            e.preventDefault();
            
            if (e.target.classList.contains('delete-list-btn') || e.target.closest('.delete-list-btn')) {
                const btn = e.target.classList.contains('delete-list-btn') ? e.target : e.target.closest('.delete-list-btn');
                const configId = parseInt(btn.dataset.configId);
                const configName = btn.dataset.configName;
                showDeleteConfigModal(configId, configName);
            } else {
                loadConfigEditor(parseInt(listItem.dataset.configId));
            }
        }
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', function () {
    loadConfigurations();
    setupEditorEventDelegation();
});