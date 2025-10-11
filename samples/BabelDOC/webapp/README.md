# BabelDOC Web应用

这是一个基于BabelDOC的PDF翻译Web应用，提供了友好的用户界面，方便用户上传PDF文件、进行翻译和下载翻译后的文件。

## 功能特点

- 拖放上传PDF文件
- 支持OpenAI API密钥配置
- 可选择不同的翻译模型
- 翻译状态实时显示
- 已翻译文件管理和下载

### 安装依赖

```bash
pip install flask werkzeug uuid
```

### 运行应用

```bash
# 在webapp目录下运行
cd /data/AIGC-SDK/samples/BabelDOC/webapp
python app.py
```

应用将在 http://0.0.0.0:5000 启动，可以通过浏览器访问。

## 使用方法

1. 打开浏览器访问应用
2. 上传PDF文件（拖放或点击选择）
3. 输入您的OpenAI API密钥
4. 选择翻译模型（默认为gpt-4o-mini）
5. 点击"开始翻译"按钮
6. 翻译完成后，可在"已翻译文件"部分下载翻译后的PDF

## 目录结构

```
webapp/
├── app.py                  # 主应用文件
├── uploads/                # 上传文件存储目录
├── downloads/              # 翻译后文件存储目录
├── templates/              # HTML模板
│   └── index.html
└── static/                 # 静态资源
    ├── css/
    │   └── styles.css
    └── js/
        └── script.js
```
