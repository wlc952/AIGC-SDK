#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/rmbg
wget https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/rmbg/rmbg.bmodel
mv *.bmodel ${sdk_dir}/bmodels/rmbg