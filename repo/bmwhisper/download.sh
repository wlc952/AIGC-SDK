#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/bmwhisper
wget https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/whisper/bmodel_base.tar.gz -O bmodel.tar.gz
tar -zxvf bmodel.tar.gz -C ${sdk_dir}/bmodels/bmwhisper
rm bmodel.tar.gz
echo "bmodel download!"

if [ ! -d "assets" ]; 
then
    python -m dfss --url=open@sophgo.com:sophon-demo/Whisper/model_240408/assets.zip
    unzip assets.zip
    rm assets.zip
    echo "assets download!"
else
    echo "Assets folder exist! Remove it if you need to update."
fi