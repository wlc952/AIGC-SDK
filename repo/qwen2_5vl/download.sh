#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PROJECT_ROOT="$DIR"

cd "$PROJECT_ROOT/python_demo"
mkdir -p build
rm -rf build/*
cd build
cmake .. && make && cp ./*cpython* ..
cd "$PROJECT_ROOT"

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/qwen2_5vl
uv run python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/qwen2.5-vl-3b-instruct-awq_w4bf16_seq256_bm1684x_1dev_imsz280.bmodel
mv *.bmodel ${sdk_dir}/bmodels/qwen2_5vl