# AIGC-SDK

本示例为算能AIPC及Airbox等BM1684X芯片产品提供支持。视频教程：[AIGC-SDK环境配置](https://b23.tv/eq2gwTz)

## 支持api列表

| 模块名称                                                 | 功能描述                                                                | api router                                      | 视频教程                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| llm_general                                              | 通用LLM（Qwen2.5、Qwen3、DeepSeek-R1-0528-Qwen3、Llama3.2、MiniCPM3等） | `/v1/chat/completions`                        | [llm_general](https://b23.tv/LcL4yQR)                |
| llm_general_v2                                           | 同上，python封装，可修改。                                              | `/v1/chat/completions`                        | [同llm_general](https://b23.tv/LcL4yQR)              |
| qwen2_5vl                                                | Qwen2.5-VL                                                              | `/v1/chat/completions`                        | [qwen2_5vl &amp; emotivoice](https://b23.tv/nwrwxe4) |
| minicpmv                                                 | MiniCPM-V2.6 等                                                         | `/v1/chat/completions`                        |                                                   |
| [bmwhisper](https://github.com/wlc952/whisper-TPU.git)      | whisper 语言转文本                                                      | `/v0/audio/transcriptions`                    |                                                   |
| [emotivoice](https://github.com/ZillaRU/EmotiVoice-TPU.git) | 文本转语音（支持情感控制）`<br>`音色克隆                              | `/v1/audio/speech<br>``/v1/audio/translation` | [qwen2_5vl &amp; emotivoice](https://b23.tv/nwrwxe4) |
| [rmbg](https://github.com/wlc952/rmbg_tpu.git)              | 图像去背景                                                              | `/v1/images/edit`                             |                                                   |
| [upscaler](https://github.com/ZillaRU/upscaler_tpu.git)     | 图像超分                                                                | `/v1/images/edit`                             |                                                   |
| [indextts](https://github.com/wlc952/index-tts.git)         | 音色克隆                                                                | `/v0/audio/speech`                            |                                                   |

## 启动 API

### 1. 初始化环境 (初次使用 AIGC-SDK)

a. 执行命令：

```sh
bash scripts/init_env.sh
```

### 2. 应用初始化（初次安装某个应用）

```sh
bash scripts/init_app.sh 服务名称
```

### 3. 启动指定服务

```sh
bash scripts/run.sh 服务名称
```

`服务名称`可以是多个，用空格分隔。例如，同时启动LLM和ASR `bash scripts/run.sh llm_general bmwhisper`。

* 运行命令后，浏览器访问 `ip:8000/docs`，此时可以看到后台开始启动各模块的应用。
* 查看并测试接口：选择对应接口并点击 `Try it out`即可在当前选项卡编辑请求并发送，response 将会显示在下方。各 API 的 request定义可以在页面最下方看到。
