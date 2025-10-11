介绍已有的API使用说明：

---

## llm_general

**简介**：  
通用大语言模型API，支持多种主流LLM模型。

**初始化环境及下载模型**：

```bash
bash scripts/init_app.sh llm_general
```

**设置环境变量**（指定要使用的模型，可用模型见[config.yml](../config.yml)）：

```bash
export MODEL_NAME="qwen2.5-3b"
```

**启动API服务**：

```bash
bash scripts/run.sh llm_general
```

**接口访问**：  
服务启动后，浏览器访问 `http://<设备IP>:8000/docs` 可查看Swagger文档，支持在线测试。

**请求参数说明**：

- `model` (string, 可选)：模型名称，默认环境变量`MODEL_NAME`指定的模型
- `messages` (list, 必填)：对话历史，格式如`[{"role":"user","content":"hello"}]`
- `stream` (bool, 可选)：是否流式返回，默认`false`

**curl示例**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-3b","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

**Python示例**：
```python
import requests
url = "http://localhost:8000/v1/chat/completions"
data = {
    "model": "qwen2.5-3b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False
}
resp = requests.post(url, json=data)
print(resp.json())

## 或者使用openai
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="0")
response = client.chat.completions.create(
    model="qwen3-1.7b",
    messages=[
        {
            "role": "user",
            "content": "hello",
        },
    ],
)
print(response.choices[0].message.content.strip())
```

---

## *llm_general_v2*

llm_general_v2采用untool的python封装接口，速度略慢但扩展性更好。初始化与llm_general相同，仅需在启动时替换名称：
```bash
bash scripts/run.sh llm_general_v2
```

---

## minicpmv

**简介**：  
多模态LLM（如MiniCPM-V）API，支持文本、图片等多模态输入。

**初始化环境及下载模型**：
```bash
bash scripts/init_app.sh minicpmv
```

**设置环境变量**（可用模型见[config.yml](../config.yml)）：
```bash
export MODEL_NAME="minicpmv26"
```

**启动API服务**：
```bash
bash scripts/run.sh minicpmv
```

**请求参数说明**：

- `messages` (list, 必填)：对话历史，最后一条支持文本或图片输入。  
  - 纯文本：`{"role":"user","content":"hello"}`
  - 多模态：`{"role":"user","content":[{"type":"text","text":"描述"},{"type":"image_url","image_url":{"url":"图片URL/base64"}}]}`
- `stream` (bool, 可选)：是否流式返回，默认`false`

**curl示例**（文本）：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":false}'
```
**curl示例**（图片）：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":[
        {"type":"text","text":"这是什么？"},
        {"type":"image_url","image_url":{"url":"https://example.com/image.jpg"}}
      ]}
    ],
    "stream": false
  }'
```

**Python示例**（文本）：
```python
import requests
url = "http://localhost:8000/v1/chat/completions"
data = {
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False
}
resp = requests.post(url, json=data)
print(resp.json())
```
**Python示例**（图片）：
```python
import requests
url = "http://localhost:8000/v1/chat/completions"
data = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "这是什么？"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    "stream": False
}
resp = requests.post(url, json=data)
print(resp.json())
```

---

## deepseek

**简介**：  
DeepSeek相关模型API，支持特定任务（如代码生成、文本生成等）。

**初始化环境及下载模型**：
```bash
bash scripts/init_app.sh deepseek
```

**模型路径**：

在[config.yml](../config.yml)中修改。
```yaml
deepseek:
  model_path: bmodels/deepseek/deepseek-r1-distill-qwen-1.5b-2048
```

**启动API服务**：
```bash
bash scripts/run.sh deepseek
```

**接口访问**：  
浏览器访问 `http://<设备IP>:8000/docs`。

**请求参数说明**：

- `model` (string, 可选)：模型名称，默认`deepseek-r1-distill-qwen`
- `messages` (list, 必填)：对话历史，格式如`[{"role":"user","content":"hello"}]`
- `stream` (bool, 可选)：是否流式返回，默认`false`

**curl示例**：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-r1-distill-qwen","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

**Python示例**：
```python
import requests
url = "http://localhost:8000/v1/chat/completions"
data = {
    "model": "deepseek-r1-distill-qwen",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False
}
resp = requests.post(url, json=data)
print(resp.json())
```

---

## emotivoice (TTS)

**简介**：  
语音合成（Text-to-Speech）API，基于emotivoice模型。

**初始化环境及下载模型**：
```bash
bash scripts/init_app.sh emotivoice
```

**设置音色（可选）**：

支持的音色见[speaker2](../repo/emotivoice/data/youdao/text/speaker2)。
- 系统变量方式：
    ```bash
    export VOICE="8051"
    ```
- 或修改[config.yml](../config.yml)
    ```yaml
    emotivoice:
      voice: '8051' 
    ```

**启动API服务**：
```bash
bash scripts/run.sh emotivoice
```

**接口访问**：  
浏览器访问 `http://<设备IP>:8000/docs`。

**接口1：文本转语音**

- 路径：`/v1/audio/speech`
- 方法：POST
- Content-Type: application/json

**请求参数说明**：

- `input` (string, 必填)：要转换为语音的文本
- `response_format` (string, 可选)：音频格式，默认`wav`
- `emotion` (string, 可选)：情感提示，见[emotion](../repo/emotivoice/data/youdao/text/emotion)
- `audio_path` (string, 可选)：参考音色路径
- 其他参数可忽略

**curl示例**：
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，世界","response_format":"wav"}' --output speech.wav
```

**Python示例**：
```python
import requests
url = "http://localhost:8000/v1/audio/speech"
data = {"input": "你好，世界", "response_format": "wav"}
resp = requests.post(url, json=data)
with open("speech.wav", "wb") as f:
    f.write(resp.content)
```

**接口2：音色转换**

- 路径：`/v1/audio/translation`
- 方法：POST
- Content-Type: multipart/form-data

**请求参数说明**：

- `file` (file, 必填)：待转换音频文件
- `ref_file` (file, 必填)：参考音色音频文件

**curl示例**：
```bash
curl -X POST "http://localhost:8000/v1/audio/translation" \
  -F "file=@source.wav" \
  -F "ref_file=@target.wav"
```

**Python示例**：
```python
import requests
url = "http://localhost:8000/v1/audio/translation"
files = {
    "file": open("source.wav", "rb"),
    "ref_file": open("target.wav", "rb")
}
resp = requests.post(url, files=files)
print(resp.json())
```

---

## sherpa (STT)

**简介**：  
语音识别（Speech-to-Text）API，基于sherpa-onnx模型。

**初始化环境及下载模型**：
```bash
bash scripts/init_app.sh sherpa
```

**启动API服务**：
```bash
bash scripts/run.sh sherpa
```

**接口访问**：  
浏览器访问 `http://<设备IP>:8000/docs`。

**接口：语音识别**

- 路径：`/v1/audio/transcriptions`
- 方法：POST
- Content-Type: multipart/form-data

**请求参数说明**：

- `file` (file, 必填)：音频文件（如wav）
- `response_format` (string, 可选)：返回格式，`text`或`json`，默认`text`

**curl示例**：
```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F "response_format=text"
```

**Python示例**：
```python
import requests
url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("test.wav", "rb")}
data = {"response_format": "text"}
resp = requests.post(url, files=files, data=data)
print(resp.text)
```

---

## 通用说明

- **接口测试**：所有API服务启动后均可通过Swagger页面（`http://<设备IP>:8000/docs`）进行接口测试。
- **多模块同时启动**：部分应用可同时启动，命令如：
  ```bash
  bash scripts/run.sh sherpa emotivoice
  ```
  但需注意TPU内存限制，详见[TPU内存大小修改方法.md](./TPU内存大小修改方法.md)。
- **常见问题**：
  - 启动失败：检查模型是否下载完整、环境变量是否正确设置
  - 内存不足：尝试关闭其他服务或调整内存分配

---

## 参考

- [主项目README](../README.md)
- [配置文件说明](../config.yml)
- [各模块源码及API定义](../api/)


