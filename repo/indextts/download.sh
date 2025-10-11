#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
cd ${sdk_dir}/bmodels/indextts
python -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/indextts_bm1684x_f16_seq256.bmodel
