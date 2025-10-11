# Videolingo: Netflix级字幕切割、翻译、对齐、甚至加上配音，一键全自动视频搬运AI字幕组

项目地址：<https://github.com/wlc952/VideoLingo.git>

项目使用本地llm asr tts api，请先在`AIGC-SDK`路径下运行api：

```bash
bash scripts/run.sh llm_general whisper emotivoice
```

推荐在其他设备的虚拟环境中进行：

安装初始化:

```bash
git clone https://github.com/wlc952/VideoLingo.git
cd VideoLingo

## 使用conda
# conda create -n videolingo python=3.10.0 -y
# conda activate videolingo

# 使用uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install pip

# 环境安装
python install.py
```

运行方法：

```bash
streamlit run st.py
```

设置api后即可使用。
