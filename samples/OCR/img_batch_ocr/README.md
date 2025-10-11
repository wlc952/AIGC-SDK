# Images-Batch-OCR

基于`VLLM API`实现的批量图片OCR处理工具，支持从文件夹读取图片并输出识别结果到文本文件。

## ✨ 功能特性

✅ 批量处理文件夹中的多张图片  
✅ 支持常见图片格式（jpg/png/bmp等）  
✅ 输出结构化文本文件  
✅ 支持自定义API端点及模型参数

## 运行示例
1. 在`images`目录中放入待识别图片
2. 执行处理：
```bash
python ocr.py
```
3. 识别结果将保存至`ocr_results.txt`


## 📂 文件结构
```
├── requirements.txt        # 依赖列表
├── qwen-vl-ocr.py          # 主程序
├── README.md               # 说明文档
└── .env.example            # 环境变量模板
```

## 📌 注意事项
❗ 首次运行前请创建`images`目录  
❗ API调用会产生相应费用，请关注用量额度  
❗ 建议图片尺寸不超过`2048x2048`像素  
❗ 当前支持常见图片格式：`jpg/jpeg/png/bmp`

