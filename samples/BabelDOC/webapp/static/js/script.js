document.addEventListener('DOMContentLoaded', function() {
    // 元素引用
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const uploadInfo = document.getElementById('upload-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const translateBtn = document.getElementById('translate-button');
    const apiKeyInput = document.getElementById('api-key');
    const toggleApiKeyBtn = document.getElementById('toggle-api-key');
    const filesSection = document.getElementById('files-section');
    const filesList = document.getElementById('files-list');
    const translationProgress = document.getElementById('translation-progress');
    const modal = document.getElementById('modal');
    const modalTitle = document.getElementById('modal-title');
    const modalMessage = document.getElementById('modal-message');
    const modalIcon = document.getElementById('modal-icon');
    const modalOkBtn = document.getElementById('modal-ok');
    const closeModal = document.querySelector('.close');

    // 变量
    let currentUploadedFile = null;
    let uploadedFileName = null;

    // 拖放文件处理
    dropArea.addEventListener('dragenter', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('dragover');
        
        // 如果已经有选择的文件，则先重置
        if (currentUploadedFile) {
            resetFileUpload();
        }
        
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // 点击上传区域触发文件输入
    // 只为上传按钮添加点击事件
    const uploadButton = dropArea.querySelector('.upload-button');
    uploadButton.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!currentUploadedFile) {
            fileInput.click();
        }
    });

    // 文件输入变化处理
    fileInput.addEventListener('change', function(e) {
        // 如果已经有选择的文件，则先重置
        if (currentUploadedFile) {
            resetFileUpload();
        }
        
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
        }
    });

    // 移除文件按钮
    removeFileBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        resetFileUpload();
    });

    // 切换API密钥可见性
    toggleApiKeyBtn.addEventListener('click', function() {
        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            toggleApiKeyBtn.innerHTML = '<i class="fas fa-eye-slash"></i>';
        } else {
            apiKeyInput.type = 'password';
            toggleApiKeyBtn.innerHTML = '<i class="fas fa-eye"></i>';
        }
    });

    // API密钥输入检查
    apiKeyInput.addEventListener('input', updateTranslateButton);

    // 翻译按钮点击
    translateBtn.addEventListener('click', startTranslation);

    // 模态窗口关闭按钮
    closeModal.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    modalOkBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // 点击模态窗口外部关闭
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // 处理文件函数
    function handleFiles(files) {
        if (files[0]) {  // 移除了对currentUploadedFile的检查，因为在change事件中已经处理过了
            const file = files[0];
            
            // 检查文件类型
            if (!file.type.match('application/pdf')) {
                showModal('错误', '请上传PDF格式的文件。', 'error');
                resetFileUpload();
                return;
            }
            
            // 检查文件大小（限制为16MB）
            if (file.size > 16 * 1024 * 1024) {
                showModal('错误', '文件大小超过限制（16MB）。', 'error');
                resetFileUpload();
                return;
            }
            
            // 先设置界面状态，再上传文件
            currentUploadedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            uploadInfo.style.display = 'flex';
            dropArea.querySelector('.upload-prompt').style.display = 'none';
            
            // 上传文件到服务器
            uploadFile(file);
        }
    }

    // 重置文件上传
    function resetFileUpload() {
        currentUploadedFile = null;
        uploadedFileName = null;
        fileInput.value = '';
        uploadInfo.style.display = 'none';
        dropArea.querySelector('.upload-prompt').style.display = 'flex';
        updateTranslateButton();
    }

    // 更新翻译按钮状态
    function updateTranslateButton() {
        translateBtn.disabled = !uploadedFileName || !apiKeyInput.value.trim();
    }

    // 格式化文件大小
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 上传文件到服务器
    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('网络响应错误');
            }
            return response.json();
        })
        .then(data => {
            if (data.filename) {
                uploadedFileName = data.filename;
                showModal('成功', '文件上传成功，可以开始翻译。', 'success');
                updateTranslateButton();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showModal('错误', '文件上传失败: ' + error.message, 'error');
            resetFileUpload();  // 确保在上传失败时重置状态
        });
    }

    // 开始翻译
    function startTranslation() {
        if (!uploadedFileName || !apiKeyInput.value.trim()) {
            showModal('警告', '请上传文件并输入API密钥。', 'warning');
            return;
        }
        
        const apiKey = apiKeyInput.value.trim();
        const model = document.getElementById('model').value.trim() || 'Qwen2.5-VL-72B-Instruct'; // 使用手动输入的模型名称，如果为空则使用默认值
        const apiBaseUrl = document.getElementById('api-base-url').value.trim();
        const outputType = document.getElementById('output-type').value;
        
        // 显示进度条
        filesSection.style.display = 'block';
        translationProgress.style.display = 'block';
        translateBtn.disabled = true;
        
        fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filename: uploadedFileName,
                api_key: apiKey,
                model: model,
                api_base_url: apiBaseUrl,
                output_type: outputType
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { 
                    // 显示终端输出（如果有）
                    if (err.command_output) {
                        displayTerminalOutput(err.command_output, err.details || '');
                    }
                    throw new Error(err.error || '翻译过程中出错'); 
                });
            }
            return response.json();
        })
        .then(data => {
            translationProgress.style.display = 'none';
            translateBtn.disabled = false;
            
            // 显示终端输出
            if (data.command_output) {
                displayTerminalOutput(data.command_output, data.command_error || '');
            }
            
            if (data.output_file) {
                showModal('成功', '文件翻译完成！', 'success');
                loadTranslatedFiles();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            translationProgress.style.display = 'none';
            translateBtn.disabled = false;
            showModal('错误', '翻译失败: ' + error.message, 'error');
        });
    }

    // 加载已翻译的文件
    function loadTranslatedFiles() {
        fetch('/files')
            .then(response => {
                if (!response.ok) {
                    throw new Error('网络响应错误');
                }
                return response.json();
            })
            .then(files => {
                filesSection.style.display = 'block';
                document.getElementById('download-section').style.display = 'block';
                renderFilesList(files);
            })
            .catch(error => {
                console.error('Error:', error);
                showModal('错误', '无法加载已翻译文件: ' + error.message, 'error');
            });
    }

    // 渲染文件列表
    function renderFilesList(files) {
        const filesList = document.getElementById('files-list');
        const filesListDownload = document.getElementById('files-list-download');
        
        // 清空列表
        filesList.innerHTML = '';
        filesListDownload.innerHTML = '';
        
        if (files.length === 0) {
            filesList.innerHTML = '<p class="no-files">暂无已翻译的文件</p>';
            filesListDownload.innerHTML = '<p class="no-files">暂无已翻译的文件</p>';
            return;
        }
        
        files.forEach(file => {
            // 在files-list中显示简单的文件项
            const fileElement = document.createElement('div');
            fileElement.className = 'file-item';
            fileElement.innerHTML = `<p>已翻译文件: ${formatFileName(file.name)}</p>`;
            filesList.appendChild(fileElement);
            
            // 在download-section中显示带下载按钮的卡片
            const fileDownloadElement = document.createElement('div');
            fileDownloadElement.className = 'file-card';
            
            // 根据文件类型显示不同的图标和提示
            const isDirectory = file.type === 'directory';
            const fileIcon = isDirectory ? 'fa-folder' : 'fa-file-pdf';
            const downloadText = isDirectory ? '下载压缩包' : '下载文件';
            
            fileDownloadElement.innerHTML = `
                <div class="file-card-header">
                    <i class="fas ${fileIcon}"></i>
                    <div class="file-card-name">${formatFileName(file.name)}</div>
                </div>
                <div class="file-card-size">${formatFileSize(file.size)}</div>
                <div class="file-card-actions">
                    <a href="${file.url}" class="download-button">
                        <i class="fas fa-download"></i> ${downloadText}
                    </a>
                </div>
            `;
            
            filesListDownload.appendChild(fileDownloadElement);
        });
    }

    // 格式化文件名
    function formatFileName(name) {
        // 如果文件名超过20个字符，截断并添加省略号
        if (name.length > 20) {
            return name.substring(0, 17) + '...';
        }
        return name;
    }

    // 显示模态窗口
    function showModal(title, message, type = 'info') {
        modalTitle.textContent = title;
        modalMessage.textContent = message;
        
        // 设置图标类型
        modalIcon.className = 'modal-icon ' + type;
        
        // 根据类型设置图标
        let iconClass = 'fa-info-circle';
        switch (type) {
            case 'success':
                iconClass = 'fa-check-circle';
                break;
            case 'error':
                iconClass = 'fa-times-circle';
                break;
            case 'warning':
                iconClass = 'fa-exclamation-triangle';
                break;
        }
        
        modalIcon.innerHTML = `<i class="fas ${iconClass}"></i>`;
        
        // 显示模态窗口
        modal.style.display = 'flex';
    }

    // 显示终端输出
    function displayTerminalOutput(stdout, stderr) {
        const terminalOutput = document.getElementById('terminal-output');
        const terminalContent = document.getElementById('terminal-content');
        const combinedOutput = [];
        
        if (stdout && stdout.trim()) {
            combinedOutput.push("标准输出 (STDOUT):\n" + stdout);
        }
        
        if (stderr && stderr.trim()) {
            combinedOutput.push("\n错误输出 (STDERR):\n" + stderr);
        }
        
        if (combinedOutput.length > 0) {
            terminalContent.textContent = combinedOutput.join('\n\n');
            terminalOutput.style.display = 'block';
        } else {
            terminalOutput.style.display = 'none';
        }
    }

    // 初始加载已翻译的文件
    loadTranslatedFiles();
});
