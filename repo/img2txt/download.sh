#!/bin/bash
set -e

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/img2txt
uv run python -m dfss --url=open@sophgo.com:/aigc/hik_llm.tar.gz
tar xzvf hik_llm.tar.gz
mv data/ImageSpeaking/bmodel/* ${sdk_dir}/bmodels/img2txt
rm -rf hik_llm.tar.gz ./data