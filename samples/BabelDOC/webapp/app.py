import os
import subprocess
import tempfile
import uuid
from flask import Flask, request, render_template, send_file, jsonify, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['DOWNLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

babeldoc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../BabelDOC')

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件在请求中
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    
    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不允许的文件类型，只接受PDF文件'}), 400
    
    # 安全地保存文件
    unique_id = str(uuid.uuid4())
    original_filename = secure_filename(file.filename)
    base_name, extension = os.path.splitext(original_filename)
    
    filename = f"{base_name}_{unique_id}{extension}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({
        'message': '文件上传成功',
        'filename': filename,
        'original_filename': original_filename
    })

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    filename = data.get('filename')
    api_key = data.get('api_key')
    model = data.get('model', 'Qwen2.5-VL-72B-Instruct')
    api_base_url = data.get('api_base_url')
    output_type = data.get('output_type', 'default')
    
    if not filename or not api_key:
        return jsonify({'error': '缺少文件名或API密钥'}), 400
    
    input_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(input_file):
        return jsonify({'error': '找不到文件'}), 404
    
    # 创建输出文件名（保留原始文件名格式）
    original_filename = filename.split('_', 1)[0] if '_' in filename else filename
    base_name, _ = os.path.splitext(original_filename)
    output_filename = f"{base_name}_translated_{str(uuid.uuid4())[:8]}.pdf"
    output_file = os.path.join(app.config['DOWNLOAD_FOLDER'], output_filename)
    
    try:
        # 构建并执行翻译命令 - 使用cwd参数设置工作目录而不是使用cd命令
        command = [".venv/bin/babeldoc", 
            "--files", input_file, 
            "--openai", 
            "--openai-model", model,
            "--openai-base-url", api_base_url,
            "--openai-api-key", api_key,
            "--output", output_file
        ]
        
        # 根据输出类型添加相应的参数
        if output_type == "mono-only":
            command.append("--no-dual")
        elif output_type == "dual-only":
            command.append("--no-mono")
        
        # 使用cwd参数设置工作目录，不需要使用cd命令
        # 异步执行命令，实际应用中可能需要使用任务队列
        # 这里简化处理，同步执行
        result = subprocess.run(command, capture_output=True, text=True, cwd=babeldoc_path)
        
        # 捕获命令输出
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
            print(f"Translation error: {stderr}")
            return jsonify({
                'error': '翻译过程中出错', 
                'details': stderr, 
                'command_output': stdout
            }), 500
        
        return jsonify({
            'message': '翻译完成',
            'output_file': output_filename,
            'command_output': stdout,
            'command_error': stderr
        })
        
    except Exception as e:
        print(f"Exception during translation: {str(e)}")
        return jsonify({'error': f'翻译过程中出错: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': '找不到文件'}), 404
    
    # 获取原始文件名（去除UUID部分）
    original_name = filename
    if '_translated_' in filename:
        parts = filename.split('_translated_')
        if len(parts) > 1:
            original_name = f"{parts[0]}_translated"
    
    # 检查是否是目录
    if os.path.isdir(file_path):
        # 创建一个临时目录来保存zip文件
        temp_dir = tempfile.gettempdir()
        zip_filename = f"{original_name}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        # 压缩目录内容
        import shutil
        shutil.make_archive(os.path.join(temp_dir, original_name), 'zip', file_path)
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
    else:
        # 对于文件，正常处理
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"{original_name}.pdf",
            mimetype='application/pdf'
        )

@app.route('/files')
def list_files():
    files = []
    for filename in os.listdir(app.config['DOWNLOAD_FOLDER']):
        file_path = os.path.join(app.config['DOWNLOAD_FOLDER'], filename)
        
        # 检查是否为PDF文件或目录
        is_valid = filename.endswith('.pdf') or (os.path.isdir(file_path) and '_translated_' in filename)
        
        if is_valid:
            # 确定文件类型和大小
            file_type = 'directory' if os.path.isdir(file_path) else 'pdf'
            
            # 对于目录，计算其中所有文件的总大小
            if file_type == 'directory':
                size = 0
                for root, dirs, dir_files in os.walk(file_path):
                    for f in dir_files:
                        fp = os.path.join(root, f)
                        if os.path.exists(fp):
                            size += os.path.getsize(fp)
            else:
                size = os.path.getsize(file_path)
            
            files.append({
                'name': filename,
                'size': size,
                'type': file_type,
                'url': url_for('download_file', filename=filename)
            })
    
    return jsonify(files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
