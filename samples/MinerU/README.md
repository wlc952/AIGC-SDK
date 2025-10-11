# MinerU: A high-quality tool for convert PDF to Markdown and JSON.一站式开源高质量数据提取工具，将PDF转换成Markdown和JSON格式

## TPU 版本使用方法

推荐在虚拟环境中安装：

```bash
git clone https://github.com/wlc952/MinerU.git
cd MinerU

uv venv .venv --python 3.10
source  .venv/bin/activbate
uv pip install -e .[full]
```

下载模型：

```bash
python scripts/download_models_tpu.py
```

使用方法（详见：[user_guide](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)）：

```bash
magic-pdf -p {some_pdf} -o {some_output_dir} -m ocr
```

网页demo：

```bash
cd projects/gradio_app
uv pip install gradio gradio-pdf
python app.py
```

在浏览器中访问 <http://127.0.0.1:7860>
