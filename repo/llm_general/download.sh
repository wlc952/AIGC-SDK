#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)

echo "请选择要下载的模型:"
echo "0. 不下载模型（默认）- 请手动拷贝模型到 ${sdk_dir}/bmodels 并修改 ${sdk_dir}/config.yml"
echo "1. 下载 qwen2.5-3b 模型"
echo "2. 下载 qwen3-1.7b 模型"
echo "3. 下载 qwen3-4b 模型"
echo "4. 下载 deepseek-r1-0528-qwen3-8b 模型"
echo "5. 下载 llama3.2-3b 模型" 
echo "6. 下载 megrez-3b 模型" 
echo -n "请输入选择 (0-6，默认0): "

read -r choice
choice=${choice:-0}

case $choice in
    0)
        echo "未下载任何模型。"
        echo "请手动拷贝模型文件到 ${sdk_dir}/bmodels 目录"
        echo "并修改 ${sdk_dir}/config.yml 配置文件"
        ;;
    1)
        echo "下载 qwen2.5-3b 模型..."
        mkdir -p ${sdk_dir}/bmodels/qwen2_5
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-3b-instruct-gptq-int4_w4bf16_seq2048_bm1684x_1dev_20250620_134431.bmodel
        mv qwen2.5-3b*.bmodel ${sdk_dir}/bmodels/qwen2_5
        echo "qwen2.5-3b 模型下载完成"
        ;;
    2)
        echo "下载 qwen3-1.7b 模型..."
        mkdir -p ${sdk_dir}/bmodels/qwen3
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-1.7b_w4bf16_seq2048_bm1684x_1dev_20250528_173737.bmodel
        mv qwen3-1.7b*.bmodel ${sdk_dir}/bmodels/qwen3
        echo "qwen3-1.7b 模型下载完成"
        ;;
    3)
        echo "下载 qwen3-4b 模型..."
        mkdir -p ${sdk_dir}/bmodels/qwen3
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen3-4b-awq_w4bf16_seq512_bm1684x_1dev_20250514_161445.bmodel
        mv qwen3-4b*.bmodel ${sdk_dir}/bmodels/qwen3
        echo "qwen3-4b 模型下载完成"
        ;;
    4)
        echo "下载 deepseek-r1-0528-qwen3-8b 模型..."
        mkdir -p ${sdk_dir}/bmodels/deepseek
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/deepseek-r1-0528-qwen3-8b_w4bf16_seq512_bm1684x.tar
        tar -xvf deepseek-r1-0528-qwen3-8b*.tar
        rm deepseek-r1-0528-qwen3-8b*.tar
        mv deepseek-r1-0528-qwen3-8b ${sdk_dir}/bmodels/deepseek
        echo "deepseek-r1-0528-qwen3-8b 模型下载完成"
        ;;
    5)
        echo "下载 llama3.2-3b 模型..."
        mkdir -p ${sdk_dir}/bmodels/llama3.2
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/llama-3.2-3b-instruct_w4f16_seq512_bm1684x_1dev_20250526_160605.bmodel
        mv llama3.2-3b*.bmodel ${sdk_dir}/bmodels/llama3.2
        echo "llama3.2-3b 模型下载完成"
        ;;
    6)
        echo "下载 megrez-3b 模型..."
        mkdir -p ${sdk_dir}/bmodels/megrez
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/megrez_bm1684x_int4_seq512.bmodel
        mv megrez*.bmodel ${sdk_dir}/bmodels/megrez
        echo "megrez-3b 模型下载完成"
        ;;
    *)
        echo "无效选择，退出。"
        exit 1
        ;;
esac