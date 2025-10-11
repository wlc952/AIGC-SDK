#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
sd_dir=${sdk_dir}/bmodels/sd_lcm

if [ ! -d "./models" ]; then
    mkdir models
fi
if [ ! -d "./models/basic" ]; then
    mkdir models/basic
fi
if [ ! -d "./models/controlnet" ]; then
    mkdir models/controlnet && uv run python -m dfss --url=open@sophgo.com:/aigc/sd/canny_multize.bmodel && mv canny_multize.bmodel models/controlnet 
fi
if [ ! -d "./models/basic/hellonijicute" ]; then
    uv run python -m dfss --url=open@sophgo.com:/aigc/sd/hellonijicute.tgz && tar -xzvf hellonijicute.tgz && rm hellonijicute.tgz && mv hellonijicute models/basic/
fi

mkdir -p $sd_dir
mv models/* $sd_dir
rm -rf models
ln -s ../../bmodels/sd_lcm models