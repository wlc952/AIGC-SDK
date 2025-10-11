#!/bin/bash
set -e

uv run python -m nltk.downloader averaged_perceptron_tagger_eng

sudo mkdir -p ~/.cache/torch/hub/
sudo cp assets/master.zip ~/.cache/torch/hub/
cd ~/.cache/torch/hub/
sudo unzip master.zip
sudo rm master.zip
sudo mv snakers4-silero-vad-6c8d844 snakers4_silero-vad_master
cd -

sdk_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)
mkdir -p ${sdk_dir}/bmodels/emotivoice
if [ ! -f model_file.tar.gz ]; then
    wget https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/emotivoice/model_file.tar.gz
else
    echo "model_file.tar.gz already exists, skipping download."
fi

tar -xzvf model_file.tar.gz
rm model_file.tar.gz
mv model_file/* ${sdk_dir}/bmodels/emotivoice/