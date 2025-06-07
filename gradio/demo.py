import os
import shutil
import subprocess
import base64
from typing import Union

import gradio as gr


def image_to_base64(path: str) -> str:
    """将本地图片转换为Base64编码"""
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"


# 初始化路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
ji_she_dir = os.path.join(current_dir, "../JiShe_image")
result_dir = os.path.join(current_dir, "../results")

# 确保目录存在
os.makedirs(ji_she_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 初始图片配置
DEFAULT_IMAGES = {
    "main_view": "car_1.png",
    "street_bg": "ori.jpg",
    "screen": "new.png",
}


def generate_html(street_bg_path: str, screen_path: str) -> str:
    """生成动态HTML内容"""
    return f"""
<div class="dashboard-container">
    <div class="viewer-box">
        <div id="container">
            <img id="street-bg" src="{image_to_base64(street_bg_path)}">
            <img id="main-view" src="{image_to_base64(DEFAULT_IMAGES['main_view'])}">
            <img id="central-screen" src="{image_to_base64(screen_path)}">
        </div>
    </div>
    <div class="description-box">
        <div class="placeholder">图片描述区域</div>
    </div>
</div>
"""


CSS = """
.dashboard-container {
    display: flex;
    gap: 20px;
    max-width: 1400px;
    margin: 20px auto;
    padding: 20px;
    border-radius: 15px;
}

.viewer-box {
    flex: 7;
    position: relative;
    min-height: 600px;
    background: black;
    border-radius: 10px;
    overflow: hidden;
}

.description-box {
    flex: 3;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    min-height: 600px;
}

.placeholder {
    color: white;
    text-align: center;
    margin-top: 50%;
    opacity: 0.3;
}

#container {
    position: relative;
    width: 100%;
    height: 100%;
}

#street-bg {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.5);
    z-index: 1;
}

#main-view {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: contain;
    z-index: 2;
    pointer-events: none;
}

#central-screen {
    position: absolute;
    width: 18%;
    left: 41%;
    top: 54%;
    z-index: 3;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0,255,255,0.4);
    transform: perspective(800px) rotateX(5deg);
    transition: all 0.3s;
}

#central-screen:hover {
    transform: perspective(800px) rotateX(0deg);
    box-shadow: 0 0 30px rgba(0,255,255,0.6);
}
"""


def process_image(file_path: Union[str, None]) -> str:
    """处理图片处理流程"""
    if not file_path:
        return generate_html(DEFAULT_IMAGES["street_bg"], DEFAULT_IMAGES["screen"])

    try:
        # 保存用户上传的图片
        target_path = os.path.join(ji_she_dir, "ori.jpg")
        shutil.copy(file_path, target_path)

        # 执行模型处理命令
        subprocess.run(
            ["python", "basicsr/test.py", "-opt", "options/test.yml"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # 获取处理结果
        result_path = os.path.join(result_dir, "new.jpg")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"处理结果不存在: {result_path}")

        return generate_html(target_path, result_path)
    except subprocess.CalledProcessError as e:
        print(f"模型执行错误: {e.stderr}")
    except Exception as e:
        print(f"处理异常: {str(e)}")

    return generate_html(DEFAULT_IMAGES["street_bg"], DEFAULT_IMAGES["screen"])


with gr.Blocks(css=CSS, title="Demo") as demo:
    # 初始化HTML组件
    html_component = gr.HTML(
        generate_html(DEFAULT_IMAGES["street_bg"], DEFAULT_IMAGES["screen"])
    )

    # 创建控制面板
    with gr.Row():
        upload = gr.File(
            label="上传图片",
            file_types=["image"],
            type="filepath",
        )
        process_btn = gr.Button("开始处理", variant="primary")

    # 绑定事件处理
    process_btn.click(
        fn=process_image,
        inputs=upload,
        outputs=html_component,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        share=True
    )