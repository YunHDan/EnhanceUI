import os
from flask import Flask, request, jsonify, send_from_directory, abort
import subprocess
from flask_cors import CORS

# 配置路径
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
PROJECT_PATH = '../'

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 清理函数
def clear_with_subprocess(folder_path):
    """通过 subprocess 执行系统命令清理文件夹"""
    try:
        # 先删除整个文件夹
        subprocess.run(
            ['rm', '-rf', folder_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # 重新创建空文件夹
        subprocess.run(
            ['mkdir', '-p', folder_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"清理失败: {e.stderr.decode().strip()}" if e.stderr else "未知错误"
        raise RuntimeError(error_msg)


# 接口
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # 上传前清空上传文件夹
        clear_with_subprocess(UPLOAD_FOLDER)  # <--- 新增清理

        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/run', methods=['POST'])
def run_model():
    try:
        # 运行前清空结果文件夹
        clear_with_subprocess(RESULT_FOLDER)  # <--- 新增清理

        subprocess.run(
            ['python', 'basicsr/test.py', '-opt', 'options/test.yml'],
            cwd=PROJECT_PATH,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return jsonify({'message': 'Model executed successfully'})
    except subprocess.CalledProcessError as e:
        print("模型运行错误输出：", e.stderr.decode())
        error_msg = e.stderr.decode().strip() if e.stderr else "未知错误"
        return jsonify({'error': f'模型执行失败: {error_msg}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 获取处理后图片接口
@app.route('/result/handled/<original_name>', methods=['GET'])
def get_handled_image(original_name):
    base, ext = os.path.splitext(original_name)
    handled_name = f"{base}_handled{ext}"
    handled_path = os.path.join(RESULT_FOLDER, handled_name)

    if not os.path.exists(handled_path):
        return abort(404, description="Handled file not found")

    return send_from_directory(RESULT_FOLDER, handled_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
