// CCTV Video Analysis Application - Frontend JavaScript

class CCTVAnalysisApp {
    constructor() {
        this.currentTaskId = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadTaskHistory();
    }

    setupEventListeners() {
        // File selection and drag & drop
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('videoFile');
        const selectFileBtn = document.getElementById('selectFileBtn');

        selectFileBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files));

        // Drag & drop events
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        uploadArea.addEventListener('click', () => fileInput.click());

        // Analysis
        document.getElementById('startAnalysisBtn').addEventListener('click', () => this.startAnalysis());

        // Downloads
        document.getElementById('downloadVideoBtn').addEventListener('click', () => this.downloadFile('video'));
        document.getElementById('downloadAnnotationsBtn').addEventListener('click', () => this.downloadFile('annotations'));

        // Retry
        document.getElementById('retryBtn').addEventListener('click', () => this.reset());
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        const files = e.dataTransfer.files;
        this.handleFileSelect(files);
    }

    async handleFileSelect(files) {
        if (files.length === 0) return;

        const file = files[0];
        
        // Validate file type
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/x-msvideo', 'video/quicktime'];
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|wmv)$/i)) {
            this.showError('対応していないファイル形式です。MP4, AVI, MOV, MKV, WMVのいずれかを選択してください。');
            return;
        }

        // Validate file size (1GB)
        if (file.size > 1024 * 1024 * 1024) {
            this.showError('ファイルサイズが大きすぎます。最大1GBまで対応しています。');
            return;
        }

        try {
            await this.uploadFile(file);
        } catch (error) {
            this.showError(`アップロードエラー: ${error.message}`);
        }
    }

    async uploadFile(file) {
        // Show progress
        this.showProgress();
        
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();

        // Progress event
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percentComplete = (e.loaded / e.total) * 100;
                this.updateProgress(percentComplete, 'アップロード中...');
            }
        });

        // Complete event
        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                this.handleUploadSuccess(response, file);
            } else {
                const error = JSON.parse(xhr.responseText);
                this.showError(error.error || 'アップロードに失敗しました');
            }
        });

        // Error event
        xhr.addEventListener('error', () => {
            this.showError('アップロードエラーが発生しました');
        });

        xhr.open('POST', '/upload');
        xhr.send(formData);
    }

    showProgress() {
        this.hideAllSections();
        document.getElementById('uploadProgress').style.display = 'block';
        this.updateProgress(0, 'アップロード準備中...');
    }

    updateProgress(percent, status) {
        document.getElementById('progressFill').style.width = `${percent}%`;
        document.getElementById('uploadStatus').textContent = status;
    }

    handleUploadSuccess(response, file) {
        this.currentTaskId = response.task_id;
        
        // Hide progress and show file info
        document.getElementById('uploadProgress').style.display = 'none';
        
        // Populate file info
        document.getElementById('fileName').textContent = response.filename;
        document.getElementById('fileSize').textContent = this.formatFileSize(response.file_size);
        document.getElementById('taskId').textContent = response.task_id;
        
        // Show sections
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('analysisSection').style.display = 'block';
        document.getElementById('analysisSection').classList.add('section-fade-in');

        this.showSuccess('ファイルのアップロードが完了しました！');
    }

    async startAnalysis() {
        const textPrompt = document.getElementById('textPrompt').value.trim();
        
        if (!textPrompt) {
            this.showError('検出したいオブジェクトを入力してください');
            return;
        }

        if (!this.currentTaskId) {
            this.showError('ファイルをアップロードしてください');
            return;
        }

        try {
            // Show processing section
            this.showProcessing();

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    task_id: this.currentTaskId,
                    text_prompt: textPrompt
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.handleAnalysisSuccess(result);
            } else {
                this.showError(result.error || '解析に失敗しました');
            }
        } catch (error) {
            this.showError(`解析エラー: ${error.message}`);
        }
    }

    showProcessing() {
        this.hideAllSections();
        document.getElementById('processingSection').style.display = 'block';
        document.getElementById('processingSection').classList.add('section-fade-in');
        
        // Start status polling
        this.pollStatus();
    }

    async pollStatus() {
        try {
            const response = await fetch(`/status/${this.currentTaskId}`);
            const status = await response.json();

            if (status.status === 'completed') {
                this.handleAnalysisSuccess(status);
            } else if (status.status === 'error') {
                this.showError(status.error || '解析中にエラーが発生しました');
            } else {
                // Continue polling
                setTimeout(() => this.pollStatus(), 2000);
            }
        } catch (error) {
            this.showError(`ステータス確認エラー: ${error.message}`);
        }
    }

    handleAnalysisSuccess(result) {
        this.hideAllSections();
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').classList.add('section-fade-in');

        // Populate summary
        const summary = result.result?.summary || {};
        document.getElementById('processedFrames').textContent = summary.processed_frames || '-';
        document.getElementById('uniqueObjects').textContent = summary.unique_objects || '-';
        document.getElementById('processingDuration').textContent = summary.tracking_duration || '-';

        // Populate detection results table
        this.populateResultsTable(result.result?.detected_objects || []);

        this.showSuccess('解析が完了しました！');
        this.loadTaskHistory(); // Refresh history
    }

    populateResultsTable(detections) {
        const tbody = document.getElementById('resultsTableBody');
        tbody.innerHTML = '';

        if (detections.length === 0) {
            const row = tbody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 5;
            cell.textContent = 'オブジェクトが検出されませんでした';
            cell.style.textAlign = 'center';
            cell.style.fontStyle = 'italic';
            cell.style.color = '#7f8c8d';
            return;
        }

        detections.forEach(detection => {
            const row = tbody.insertRow();
            row.insertCell().textContent = detection.frame;
            row.insertCell().textContent = detection.class_name;
            row.insertCell().textContent = (detection.confidence * 100).toFixed(1) + '%';
            row.insertCell().textContent = detection.track_id;
            row.insertCell().textContent = `[${detection.bbox.join(', ')}]`;
        });
    }

    async downloadFile(fileType) {
        if (!this.currentTaskId) {
            this.showError('タスクIDが見つかりません。再度動画をアップロードしてください。');
            return;
        }

        try {
            // First check if task exists and is completed
            const statusResponse = await fetch(`/status/${this.currentTaskId}`);
            if (!statusResponse.ok) {
                this.showError('タスクが見つかりません。動画を再度アップロードして解析してください。');
                return;
            }

            const taskStatus = await statusResponse.json();
            if (taskStatus.status !== 'completed') {
                this.showError('解析が完了していません。解析完了後に再度お試しください。');
                return;
            }

            // Try direct download approach first
            try {
                const downloadUrl = `/download/${this.currentTaskId}/${fileType}`;
                const a = document.createElement('a');
                a.href = downloadUrl;
                a.download = `result_${fileType}_${this.currentTaskId}.${fileType === 'video' ? 'mp4' : 'json'}`;
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                this.showSuccess(`${fileType === 'video' ? '動画' : '注釈データ'}のダウンロードを開始しました`);
                return; // Exit if direct download works
            } catch (directError) {
                console.log('Direct download failed, trying fetch approach:', directError);
            }
            
            // Fallback to fetch approach
            const response = await fetch(`/download/${this.currentTaskId}/${fileType}`, {
                method: 'GET',
                headers: {
                    'Accept': fileType === 'video' ? 'video/mp4' : 'application/json'
                }
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `result_${fileType}_${this.currentTaskId}.${fileType === 'video' ? 'mp4' : 'json'}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showSuccess(`${fileType === 'video' ? '動画' : '注釈データ'}のダウンロードが完了しました`);
            } else {
                let errorMessage;
                try {
                    const error = await response.json();
                    errorMessage = error.error || 'ダウンロードに失敗しました';
                } catch {
                    errorMessage = `ダウンロードに失敗しました (HTTP ${response.status})`;
                }
                this.showError(errorMessage);
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showError(`ダウンロードエラー: ${error.message}`);
        }
    }

    async loadTaskHistory() {
        try {
            const response = await fetch('/tasks');
            const tasks = await response.json();
            this.displayTaskHistory(tasks);
        } catch (error) {
            console.error('履歴の読み込みに失敗:', error);
        }
    }

    displayTaskHistory(tasks) {
        const historyList = document.getElementById('historyList');
        
        if (tasks.length === 0) {
            historyList.innerHTML = '<p class="no-history">まだ処理履歴がありません</p>';
            return;
        }

        historyList.innerHTML = '';
        
        tasks.forEach(task => {
            const item = document.createElement('div');
            item.className = 'history-item';
            
            const statusClass = `status-${task.status}`;
            
            item.innerHTML = `
                <div class="history-item-header">
                    <span class="history-filename">${task.filename || 'Unknown File'}</span>
                    <span class="history-status ${statusClass}">${this.getStatusText(task.status)}</span>
                </div>
                <div class="history-prompt">${task.text_prompt || 'プロンプト未設定'}</div>
                <div style="font-size: 0.8rem; color: #aaa; margin-top: 5px;">
                    ${task.upload_time ? new Date(task.upload_time).toLocaleString('ja-JP') : ''}
                </div>
            `;
            
            historyList.appendChild(item);
        });
    }

    getStatusText(status) {
        const statusMap = {
            'uploaded': 'アップロード済み',
            'processing': '処理中',
            'completed': '完了',
            'error': 'エラー'
        };
        return statusMap[status] || status;
    }

    hideAllSections() {
        const sections = [
            'analysisSection', 
            'processingSection', 
            'resultsSection', 
            'errorSection'
        ];
        
        sections.forEach(id => {
            const element = document.getElementById(id);
            element.style.display = 'none';
            element.classList.remove('section-fade-in');
        });
    }

    showError(message) {
        this.hideAllSections();
        document.getElementById('errorSection').style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
        document.getElementById('errorSection').classList.add('section-fade-in');
    }

    showSuccess(message) {
        // You could implement a success notification here
        console.log('Success:', message);
    }

    reset() {
        this.currentTaskId = null;
        document.getElementById('videoFile').value = '';
        document.getElementById('textPrompt').value = '';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('uploadProgress').style.display = 'none';
        this.hideAllSections();
        this.loadTaskHistory();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CCTVAnalysisApp();
});