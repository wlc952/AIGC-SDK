# aTrain: 一个跨平台、图形界面的自动语音识别工具

项目地址：<https://github.com/wlc952/aTrain.git>

项目使用本地asr api，请先在`AIGC-SDK`路径下运行api：

```bash
bash scripts/run.sh sherpa
# 或 `bash scripts/run.sh bmwhisper`
# 或 `bash scripts/run.sh sherpa bmwhisper`
```

推荐在其他设备的虚拟环境中进行：

安装初始化:

```bash
git clone https://github.com/wlc952/aTrain.git
cd aTrain
uv venv .venv --python 3.10
uv pip install -e .

pip install -e .

# 使用方法
aTrain start
```

除了命令行打开方式，还可以编译成独立的可执行程序：请参考：[aTrain.wiki](https://github.com/JuergenFleiss/aTrain/wiki/Manual-Installation-and-Builds#how-to-build-a-standalone-executable-)
