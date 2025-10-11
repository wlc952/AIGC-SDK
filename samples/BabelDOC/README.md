项目地址：<https://github.com/wlc952/BabelDOC.git>

推荐在虚拟环境中进行。

安装初始化:

```bash
git clone https://github.com/wlc952/BabelDOC.git
cd BabelDOC

uv run babeldoc --help # 初始化

# 使用方法
uv run babeldoc --files example.pdf --openai --openai-model "gpt-4o-mini" --openai-base-url "https://api.openai.com/v1" --openai-api-key "your-api-key-here"
```

webapp (使用前先完成前面的“初始化”步骤):

```bash
cd webapp && python app.py
```
