#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)

echo "请选择要下载的模型:"
echo "0. 不下载模型（默认）- 请手动拷贝模型到 ${sdk_dir}/bmodels 并修改 ${sdk_dir}/config.yml"
echo "1. 下载 minicpmv26 模型"
echo "2. 下载 minicpm3o 模型"
echo -n "请输入选择 (0-2，默认0): "

read -r choice
choice=${choice:-0}

case $choice in
    0)
        echo "未下载任何模型。"
        echo "请手动拷贝模型文件到 ${sdk_dir}/bmodels 目录"
        echo "并修改 ${sdk_dir}/config.yml 配置文件"
        ;;
    1)
        echo "下载 minicpmv26 模型..."
        mkdir -p ${sdk_dir}/bmodels/minicpmv
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpmv26_bm1684x_int4_seq1024_imsize448.bmodel
        mv minicpmv*.bmodel ${sdk_dir}/bmodels/minicpmv
        echo "minicpmv26 模型下载完成"
        ;;
    2)
        echo "下载 minicpm3o 模型..."
        mkdir -p ${sdk_dir}/bmodels/minicpmo
        uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/minicpm3o_bm1684x_int4_seq1024.bmodel
        mv minicpm3o*.bmodel ${sdk_dir}/bmodels/minicpmo
        echo "minicpm3o 模型下载完成"
        ;;
    *)
        echo "无效选择，退出。"
        exit 1
        ;;
esac