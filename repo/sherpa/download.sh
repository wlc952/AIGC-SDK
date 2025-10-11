#!/bin/bash
set -e

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make clean
make -j$(nproc)

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/sherpa
wget https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/sherpa-onnx/sherpa.tar.gz -O ${sdk_dir}/bmodels/sherpa.tar.gz
tar -zxvf ${sdk_dir}/bmodels/sherpa.tar.gz -C ${sdk_dir}/bmodels/sherpa
rm -rf ${sdk_dir}/bmodels/sherpa.tar.gz