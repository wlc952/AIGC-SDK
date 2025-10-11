#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/upscaler
wget https://github.com/ZillaRU/upscaler_tpu/releases/download/v0.1/resrgan4x.bmodel -O ${sdk_dir}/bmodels/upscaler/resrgan4x.bmodel