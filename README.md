# AIGC-SDK

本项目为算能AIPC及Airbox等BM1684X芯片产品提供支持。视频教程：[AIGC-SDK环境配置](https://b23.tv/eq2gwTz)

## 支持api列表

| 模块名称                                                 | 功能描述                                                                | api router                                      | 视频教程                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| llm_general                                              | 通用LLM（Qwen2.5、Qwen3、DeepSeek-R1-0528-Qwen3、Llama3.2、MiniCPM3等） | `/v1/chat/completions`                        | [llm_general](https://b23.tv/LcL4yQR)                |
| llm_general_v2                                           | 同上                                                                    | `/v1/chat/completions`                        | [同llm_general](https://b23.tv/LcL4yQR)              |
| qwen2_5vl                                                | Qwen2.5-VL                                                              | `/v1/chat/completions`                        | [qwen2_5vl &amp; emotivoice](https://b23.tv/nwrwxe4) |
| qwen3vl                                                  | Qwen3-VL                                                                | `/v1/chat/completions`                        | 参考qwen2_5vl                                     |
| minicpmv                                                 | MiniCPM-V2.6 等                                                         | `/v1/chat/completions`                        |                                                   |
| [sherpa](https://github.com/wlc952/Kaldi-TPU.git)           | sherpa-onnx 语言转文本                                                  | `/v1/audio/transcriptions`                    |                                                   |
| [bmwhisper](https://github.com/wlc952/whisper-TPU.git)      | whisper 语言转文本                                                      | `/v0/audio/transcriptions`                    |                                                   |
| [emotivoice](https://github.com/ZillaRU/EmotiVoice-TPU.git) | 文本转语音（支持情感控制）`<br>`音色克隆                              | `/v1/audio/speech<br>``/v1/audio/translation` | [qwen2_5vl &amp; emotivoice](https://b23.tv/nwrwxe4) |
| [roop_face](https://github.com/ZillaRU/roop_face.git)       | 人像换脸 `<br>`人脸增强                                               | `/v1/images/variations<br>``/v1/images/edit`  |                                                   |
| [sd_lcm](https://github.com/ZillaRU/SD-lcm-tpu)             | SD文生图、图生图                                                        | `/v1/images/generations<br>``/v1/images/edit` |                                                   |
| [rmbg](https://github.com/wlc952/rmbg_tpu.git)              | 图像去背景                                                              | `/v1/images/edit`                             |                                                   |
| [img2txt](https://github.com/ZillaRU/ImageSpeaking.git)     | 看图说话、生成图像描述                                                  | `/v1/images/variations`                       |                                                   |
| [upscaler](https://github.com/ZillaRU/upscaler_tpu.git)     | 图像超分                                                                | `/v1/images/edit`                             |                                                   |
| [indextts](https://github.com/wlc952/index-tts.git)         | 音色克隆                                                                | `/v0/audio/speech`                            |                                                   |

* 相关推荐：[双卡推理Qwen3-32B](https://www.modelscope.cn/models/wlc952/TwinStream-LLM)；[AI证件照生成](https://github.com/wlc952/HivisionIDPhotos-TPU)

## samples列表

| 样例名称                                        | 功能描述                                                                                          | 视频展示                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------- |
| [aTrain](https://github.com/wlc952/aTrain)         | 一个图形界面的自动语音识别工具                                                                    | [aTrain](https://b23.tv/YrpctQY)     |
| [BabelDOC](https://github.com/wlc952/BabelDOC)     | 基于LLM api的文档翻译程序                                                                         |                                   |
| [lobe-chat](https://github.com/lobehub/lobe-chat)  | 现代化设计的开源 ChatGPT/LLMs 聊天应用                                                            |                                   |
| MeetingHelper                                   | 会议助手webdemo：文本总结，语音转文字，文件整理……                                               |                                   |
| [MinerU OCR](https://github.com/wlc952/MinerU)     | 一站式开源高质量数据提取工具，将PDF转换成Markdown和JSON格式                                       | [MinerU](https://b23.tv/be4kgYj)     |
| OCR                                             | 基于VLM api的图片文字识别程序， 如：[QwenVL-Batch-OCR](https://github.com/WarmneoN/QwenVL-Batch-OCR) |                                   |
| [Videolingo](https://github.com/wlc952/VideoLingo) | Netflix级字幕切割、翻译、对齐、甚至加上配音，一键全自动视频搬运AI字幕组                           | [Videolingo](https://b23.tv/vpJUw2a) |
| VoiceChat                                       | 基于ASR/LLM/TTS api的语音对话webdemo                                                              |                                   |
| WebHub                                          | 基于多种api的网页端AIGC Hub                                                                       |                                   |

## 如何使用 AIGC-SDK 中的API

### 1. 下载本项目并初始化环境 (初次使用 AIGC-SDK)

a. 执行命令：

```sh
cd AIGC-SDK
bash scripts/init_env.sh
```

### 2. 应用初始化（初次安装某个应用）

\* 如果本项目有更新的版本，建议先执行 `git pull`更新本项目代码。

执行命令：

```sh
bash scripts/init_app.sh 应用名称
```

其中 `应用名称`可以是多个，用空格分开。比如 `bash scripts/init_app.sh emotivoice sherpa`。

这一步会配置环境、下载默认的模型文件。

### 3. 启动指定的后端服务

```sh
bash scripts/run.sh 应用名称
```

模块名称可以是多个，用空格分隔。例如，同时启动LLM和ASR `bash scripts/run.sh llm_general sherpa`。

请注意，由于Airbox的 TPU 内存限制，部分应用不能同时启动，内存修改的方法请参考[TPU内存大小修改方法](docs/TPU内存大小修改方法.md)。

* 运行命令后，浏览器访问 `ip:8000/docs`，此时可以看到后台开始启动各模块的应用。
* 查看并测试接口：选择对应接口并点击 `Try it out`即可在当前选项卡编辑请求并发送，response 将会显示在下方。各 API 的 request定义可以在页面最下方看到。

### 4. LLM多机多卡提升并发性能（注意修改ip，示例中两台设备ip分别为 `192.168.150.1`和 `192.168.150.10`）

#### 4.1 分别启动api

两台设备上分别按照前面的说明启动相同的api。

#### 4.2 nginx配置(airbox已装nginx，省略安装步骤)

在一台设备上执行

```bash
sudo rm /etc/nginx/sites-enabled/default
sudo rm /etc/nginx/conf.d/*
sudo vim /etc/nginx/conf.d/chat.conf
```

写入下面内容：

```conf
upstream chat_backend {
    server 192.168.150.1:8000 max_fails=3 fail_timeout=5s;  # 本机
    server 192.168.150.10:8000 max_fails=3 fail_timeout=5s;  # 对端
    keepalive 32;
}

server {
    listen 80;
    location / {
        proxy_pass http://chat_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_buffering off;
    }
}
```

然后重启nginx:

```bash
sudo nginx -s reload
```

#### 4.3 使用说明

按照上面配置好之后，api的url的端口为80，而非单机的8000。如：`http://192.168.150.1/v1/chat/completions`

## 使用 Docker 镜像运行 AIGC-SDK 的 API

### 下载镜像

```bash
# soc版本镜像
wget https://www.modelscope.cn/models/wlc952/aigchub_models/resolve/master/docker-img/aigc-sdk-arm-latest.tar
# pcie版本镜像
# wget https://www.modelscope.cn/models/wlc952/aigchub_models/resolve/master/docker-img/aigc-sdk-arm-pcie-latest.tar
```

### 加载镜像

```bash
docker load < aigc-sdk-arm-latest.tar
```

### 准备挂载文件

需要创建或准备以下文件或文件夹：

```bash
bmodels # 存放模型和token_config等文件
config.yml # 记录模型名的bmodel文件和token_config路径等信息
tmpdir # 临时文件夹
```

### 运行容器（建议使用 docker-compose）

(1) 使用 docker-compose

`docker-compose`安装方法：

```bash
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
docker-compose --version
```

修改[`docker-compose.yml`](docker-compose.yml)中的环境变量，

```yml
environment:
    - MODEL_NAME=qwen2.5-3b # 要推理的LLM模型名
    - API=llm_general # 要使用的 API
```

然后运行：

```bash
docker-compose up
```

(2) 使用 docker run 命令：

```bash
docker run -it \
    --name aigc-sdk \
    --privileged \
    -p 8000:8000 \
    -v /opt/sophon:/opt/sophon \
    -v /etc/profile.d:/etc/profile.d \
    -v /etc/ld.so.conf.d:/etc/ld.so.conf.d \
    -v "$(pwd)/bmodels:/AIGC-SDK/bmodels" \
    -v "$(pwd)/tmpdir:/AIGC-SDK/tmpdir" \
    -v "$(pwd)/config.yml:/AIGC-SDK/config.yml" \
    -e MODEL_NAME=minicpm3o \
    -e API=sherpa \
    aigc-sdk-arm:latest
```

### 使用 API

API服务在 `localhost:8000`端口提供，访问 `http://localhost:8000/docs`查看
