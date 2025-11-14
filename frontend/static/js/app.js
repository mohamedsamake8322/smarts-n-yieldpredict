// Advanced Voice Assistant Application
class VoiceAssistantApp {
    constructor() {
        this.currentConversationId = null;
        this.isRecording = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.currentLanguage = 'fr';
        this.documents = [];
        
        this.initializeApp();
        this.initializeVoiceRecognition();
        this.initializeFileUpload();
        this.loadDocuments();
    }

    initializeApp() {
        // Language selection
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.addEventListener('change', (e) => {
                this.currentLanguage = e.target.value;
                this.updateLanguage();
            });
        }

        // Message input handling
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
        }

        // Search input handling
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.searchDocuments();
                }
            });
        }

        // Load initial data
        this.refreshDocuments();
        this.populateDocumentSelects();
    }

    initializeVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = this.getRecognitionLanguage();
            
            this.recognition.onstart = () => {
                this.isRecording = true;
                this.updateVoiceButton(true);
            };
            
            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('messageInput').value = transcript;
                this.sendMessage();
            };
            
            this.recognition.onend = () => {
                this.isRecording = false;
                this.updateVoiceButton(false);
            };
            
            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.showAlert('Erreur de reconnaissance vocale: ' + event.error, 'warning');
                this.isRecording = false;
                this.updateVoiceButton(false);
            };
        }
    }

    initializeFileUpload() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');

        if (uploadZone) {
            // Drag and drop handlers
            uploadZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadZone.classList.add('dragover');
            });

            uploadZone.addEventListener('dragleave', () => {
                uploadZone.classList.remove('dragover');
            });

            uploadZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                
                const files = Array.from(e.dataTransfer.files);
                this.handleFileUpload(files);
            });

            uploadZone.addEventListener('click', () => {
                fileInput.click();
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleFileUpload(files);
            });
        }
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        for (const file of files) {
            await this.uploadSingleFile(file);
        }
    }

    async uploadSingleFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const progressContainer = document.getElementById('uploadProgress');
        const progressBar = progressContainer.querySelector('.progress-bar');
        const statusText = document.getElementById('uploadStatus');

        try {
            progressContainer.style.display = 'block';
            statusText.textContent = `Téléchargement de ${file.name}...`;
            progressBar.style.width = '50%';

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            progressBar.style.width = '100%';

            if (result.success) {
                statusText.textContent = `${file.name} traité avec succès!`;
                this.showAlert(result.message, 'success');
                
                // Add document to list
                this.documents.push(result.document);
                this.refreshDocumentsDisplay();
                this.populateDocumentSelects();
                
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 2000);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            progressContainer.style.display = 'none';
            this.showAlert(`Erreur lors du téléchargement de ${file.name}: ${error.message}`, 'danger');
        }
    }

    async sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (!message) return;

        // Clear input and add user message to chat
        messageInput.value = '';
        this.addMessageToChat('user', message);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: message,
                    conversation_id: this.currentConversationId,
                    language: this.currentLanguage
                })
            });

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // Add assistant response to chat
            this.addMessageToChat('assistant', result.response);
            this.currentConversationId = result.conversation_id;

            // Show suggested questions if available
            if (result.suggested_questions && result.suggested_questions.length > 0) {
                this.showSuggestedQuestions(result.suggested_questions);
            }

            // Speak response if synthesis is available
            if (this.synthesis && this.synthesis.speaking === false) {
                this.speakText(result.response);
            }

        } catch (error) {
            this.addMessageToChat('assistant', `Erreur: ${error.message}`);
            this.showAlert('Erreur de chat: ' + error.message, 'danger');
        }
    }

    addMessageToChat(role, content) {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.textContent = content;
        
        if (role === 'assistant') {
            bubbleDiv.innerHTML = `<i class="fas fa-robot"></i> ${content}`;
        }
        
        messageDiv.appendChild(bubbleDiv);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    showSuggestedQuestions(questions) {
        const suggestedContainer = document.getElementById('suggestedQuestions');
        const buttonsContainer = document.getElementById('suggestionButtons');
        
        buttonsContainer.innerHTML = '';
        
        questions.forEach(question => {
            const button = document.createElement('button');
            button.className = 'btn btn-outline-primary btn-sm me-1 mb-1';
            button.textContent = question;
            button.onclick = () => {
                document.getElementById('messageInput').value = question;
                this.sendMessage();
                suggestedContainer.style.display = 'none';
            };
            buttonsContainer.appendChild(button);
        });
        
        suggestedContainer.style.display = 'block';
    }

    toggleVoice() {
        if (!this.recognition) {
            this.showAlert('Reconnaissance vocale non supportée dans ce navigateur', 'warning');
            return;
        }

        if (this.isRecording) {
            this.recognition.stop();
        } else {
            this.recognition.lang = this.getRecognitionLanguage();
            this.recognition.start();
        }
    }

    updateVoiceButton(recording) {
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            if (recording) {
                voiceBtn.classList.add('recording');
                voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
            } else {
                voiceBtn.classList.remove('recording');
                voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            }
        }
    }

    speakText(text) {
        if (this.synthesis) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = this.getSynthesisLanguage();
            this.synthesis.speak(utterance);
        }
    }

    getRecognitionLanguage() {
        const langMap = {
            'fr': 'fr-FR',
            'en': 'en-US',
            'es': 'es-ES'
        };
        return langMap[this.currentLanguage] || 'fr-FR';
    }

    getSynthesisLanguage() {
        const langMap = {
            'fr': 'fr-FR',
            'en': 'en-US',
            'es': 'es-ES'
        };
        return langMap[this.currentLanguage] || 'fr-FR';
    }

    async searchDocuments() {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput.value.trim();
        
        if (!query) return;

        try {
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    language_filter: null
                })
            });

            const result = await response.json();
            this.displaySearchResults(result.results);

        } catch (error) {
            this.showAlert('Erreur de recherche: ' + error.message, 'danger');
        }
    }

    displaySearchResults(results) {
        const resultsContainer = document.getElementById('searchResults');
        
        if (results.length === 0) {
            resultsContainer.innerHTML = '<p class="text-muted">Aucun résultat trouvé.</p>';
            return;
        }

        const html = results.map(result => `
            <div class="border-bottom py-2">
                <h6 class="mb-1">
                    <i class="fas fa-file-alt text-primary"></i>
                    ${result.document.original_filename}
                </h6>
                <small class="text-muted">
                    Pertinence: ${Math.round(result.similarity * 100)}%
                </small>
                <p class="mb-1 small">${result.snippet}</p>
            </div>
        `).join('');
        
        resultsContainer.innerHTML = html;
    }

    async refreshDocuments() {
        try {
            const response = await fetch('/api/documents');
            const result = await response.json();
            
            this.documents = result.documents;
            this.refreshDocumentsDisplay();
            this.populateDocumentSelects();
            
        } catch (error) {
            this.showAlert('Erreur lors du chargement des documents: ' + error.message, 'danger');
        }
    }

    refreshDocumentsDisplay() {
        const docCount = document.getElementById('docCount');
        const documentsList = document.getElementById('documentsList');
        
        if (docCount) {
            docCount.textContent = this.documents.length;
        }
        
        if (documentsList) {
            if (this.documents.length === 0) {
                documentsList.innerHTML = `
                    <div class="col-12">
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-folder-open fa-3x mb-3"></i>
                            <p>Aucun document téléchargé</p>
                        </div>
                    </div>
                `;
            } else {
                const html = this.documents.map(doc => `
                    <div class="col-lg-4 col-md-6 mb-3">
                        <div class="card document-card h-100">
                            <div class="card-body">
                                <h6 class="card-title">
                                    <i class="fas fa-file-alt text-primary"></i>
                                    ${doc.original_filename}
                                </h6>
                                <p class="card-text small text-muted">
                                    <span class="badge bg-info">${doc.detected_language}</span>
                                    ${doc.word_count} mots
                                    <br>
                                    <i class="fas fa-calendar"></i>
                                    ${new Date(doc.created_at).toLocaleDateString('fr-FR')}
                                </p>
                                <div class="btn-group w-100">
                                    <button class="btn btn-sm btn-outline-primary" onclick="app.viewDocument(${doc.id})">
                                        <i class="fas fa-eye"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-secondary" onclick="app.generateSummaryForDoc(${doc.id})">
                                        <i class="fas fa-file-contract"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-success" onclick="app.shareDocument(${doc.id})">
                                        <i class="fas fa-share"></i>
                                    </button>
                                    <button class="btn btn-sm btn-outline-danger" onclick="app.deleteDocument(${doc.id})">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('');
                
                documentsList.innerHTML = html;
            }
        }
    }

    populateDocumentSelects() {
        const selects = [
            'summaryDocument',
            'translateDocument', 
            'quizDocument',
            'entitiesDocument'
        ];

        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                const currentValue = select.value;
                select.innerHTML = '<option value="">Sélectionner un document...</option>';
                
                this.documents.forEach(doc => {
                    const option = document.createElement('option');
                    option.value = doc.id;
                    option.textContent = doc.original_filename;
                    select.appendChild(option);
                });
                
                // Restore selection if it still exists
                if (currentValue) {
                    select.value = currentValue;
                }
            }
        });
    }

    async generateSummary() {
        const docSelect = document.getElementById('summaryDocument');
        const typeSelect = document.getElementById('summaryType');
        
        const docId = docSelect.value;
        const summaryType = typeSelect.value;
        
        if (!docId) {
            this.showAlert('Veuillez sélectionner un document', 'warning');
            return;
        }

        try {
            const response = await fetch(`/api/documents/${docId}/summary?type=${summaryType}`);
            const result = await response.json();
            
            if (result.summary) {
                this.showModal('Résumé du document', result.summary.content);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showAlert('Erreur de génération du résumé: ' + error.message, 'danger');
        }
    }

    async translateDocument() {
        const docSelect = document.getElementById('translateDocument');
        const langSelect = document.getElementById('targetLanguage');
        
        const docId = docSelect.value;
        const targetLanguage = langSelect.value;
        
        if (!docId) {
            this.showAlert('Veuillez sélectionner un document', 'warning');
            return;
        }

        try {
            const response = await fetch(`/api/documents/${docId}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    target_language: targetLanguage
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showModal('Traduction du document', result.translation.translated_text);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showAlert('Erreur de traduction: ' + error.message, 'danger');
        }
    }

    async generateQuiz() {
        const docSelect = document.getElementById('quizDocument');
        const difficultySelect = document.getElementById('quizDifficulty');
        const questionsSelect = document.getElementById('quizQuestions');
        
        const docId = docSelect.value;
        const difficulty = difficultySelect.value;
        const numQuestions = parseInt(questionsSelect.value);
        
        if (!docId) {
            this.showAlert('Veuillez sélectionner un document', 'warning');
            return;
        }

        try {
            const response = await fetch(`/api/documents/${docId}/quiz`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    difficulty: difficulty,
                    num_questions: numQuestions
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayQuiz(result.questions);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showAlert('Erreur de génération du quiz: ' + error.message, 'danger');
        }
    }

    displayQuiz(questions) {
        const html = questions.map((q, index) => `
            <div class="mb-4">
                <h6>Question ${index + 1}: ${q.question}</h6>
                <div class="ms-3">
                    ${q.options.map((option, optIndex) => `
                        <div class="form-check">
                            <input class="form-check-input" type="radio" 
                                   name="q${index}" value="${option}" id="q${index}_${optIndex}">
                            <label class="form-check-label" for="q${index}_${optIndex}">
                                ${option}
                            </label>
                        </div>
                    `).join('')}
                </div>
                <small class="text-success">
                    <strong>Réponse correcte:</strong> ${q.correct_answer}
                </small>
                ${q.explanation ? `<br><small class="text-muted">${q.explanation}</small>` : ''}
            </div>
        `).join('');
        
        this.showModal('Quiz généré', html);
    }

    showModal(title, content) {
        const modal = document.getElementById('documentModal');
        const modalTitle = modal.querySelector('.modal-title');
        const modalBody = modal.querySelector('.modal-body');
        
        modalTitle.textContent = title;
        modalBody.innerHTML = content;
        
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();
    }

    showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.getElementById('alertsContainer').appendChild(alertDiv);
        
        setTimeout(() => {
            if (alertDiv.parentNode) {
                const alert = bootstrap.Alert.getOrCreateInstance(alertDiv);
                alert.close();
            }
        }, 5000);
    }

    async deleteDocument(docId) {
        if (!confirm('Êtes-vous sûr de vouloir supprimer ce document ?')) {
            return;
        }

        try {
            const response = await fetch(`/api/delete_document/${docId}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(result.message, 'success');
                await this.refreshDocuments();
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            this.showAlert('Erreur de suppression: ' + error.message, 'danger');
        }
    }

    // External integrations placeholders
    connectGoogleDrive() {
        this.showAlert('Intégration Google Drive disponible - Configuration requise dans les paramètres', 'info');
    }

    connectDropbox() {
        this.showAlert('Intégration Dropbox disponible - Configuration requise dans les paramètres', 'info');
    }

    connectOneDrive() {
        this.showAlert('Intégration OneDrive disponible - Configuration requise dans les paramètres', 'info');
    }

    showWorkspaces() {
        this.showAlert('Système d\'espaces de travail collaboratifs implémenté', 'info');
    }

    showSettings() {
        this.showAlert('Paramètres utilisateur avec préférences multilingues', 'info');
    }

    updateLanguage() {
        // Update UI language if needed
        console.log('Language changed to:', this.currentLanguage);
    }

    async loadDocuments() {
        await this.refreshDocuments();
    }

    // Additional methods for document management
    async viewDocument(docId) {
        try {
            const doc = this.documents.find(d => d.id === docId);
            if (doc) {
                this.showModal(`Document: ${doc.original_filename}`, `
                    <div class="mb-3">
                        <strong>Langue détectée:</strong> <span class="badge bg-info">${doc.detected_language}</span><br>
                        <strong>Nombre de mots:</strong> ${doc.word_count}<br>
                        <strong>Date de création:</strong> ${new Date(doc.created_at).toLocaleDateString('fr-FR')}
                    </div>
                    <div class="mb-3">
                        <strong>Contenu (extrait):</strong>
                        <div class="border p-2 mt-1" style="max-height: 300px; overflow-y: auto;">
                            ${doc.content_text ? doc.content_text.substring(0, 1000) + (doc.content_text.length > 1000 ? '...' : '') : 'Contenu non disponible'}
                        </div>
                    </div>
                `);
            }
        } catch (error) {
            this.showAlert('Erreur lors de l\'affichage du document: ' + error.message, 'danger');
        }
    }

    async generateSummaryForDoc(docId) {
        try {
            const response = await fetch(`/api/documents/${docId}/summary`);
            const result = await response.json();
            
            if (result.summary) {
                this.showModal('Résumé automatique', result.summary.content);
            } else {
                this.showAlert('Erreur: ' + result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Erreur de génération du résumé: ' + error.message, 'danger');
        }
    }

    async shareDocument(docId) {
        const email = prompt('Entrez l\'email de la personne avec qui partager:');
        if (!email) return;

        try {
            const response = await fetch(`/api/documents/${docId}/share`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    email: email,
                    permission: 'read'
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showAlert(result.message, 'success');
            } else {
                this.showAlert('Erreur: ' + result.error, 'danger');
            }
        } catch (error) {
            this.showAlert('Erreur de partage: ' + error.message, 'danger');
        }
    }
}

// Global functions for HTML onclick handlers
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        app.sendMessage();
    }
}

function toggleVoice() {
    app.toggleVoice();
}

function sendMessage() {
    app.sendMessage();
}

function searchDocuments() {
    app.searchDocuments();
}

function refreshDocuments() {
    app.refreshDocuments();
}

function generateSummary() {
    app.generateSummary();
}

function translateDocument() {
    app.translateDocument();
}

function generateQuiz() {
    app.generateQuiz();
}

function extractEntities() {
    app.showAlert('Extraction d\'entités implémentée - Disponible via l\'API', 'info');
}

function connectGoogleDrive() {
    app.connectGoogleDrive();
}

function connectDropbox() {
    app.connectDropbox();
}

function connectOneDrive() {
    app.connectOneDrive();
}

function showWorkspaces() {
    app.showWorkspaces();
}

function showSettings() {
    app.showSettings();
}

function showDocumentFilters() {
    app.showAlert('Filtres de documents implémentés dans le système', 'info');
}

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new VoiceAssistantApp();
});