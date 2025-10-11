#!/bin/bash
set -euo pipefail

SDK_TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
cd "$SDK_TOP"
mkdir -p "$SDK_TOP/tmpdir"

sudo apt-get update -y
sudo apt-get install -y unzip ffmpeg alsa-utils libasound2-dev libsndfile1

LIBSOPHON_INCLUDE=${LIBSOPHON_INCLUDE:-/opt/sophon/libsophon-current/include}
sudo cp "$SDK_TOP/support/libsophon/include/utils.h" "$LIBSOPHON_INCLUDE/"
sudo cp "$SDK_TOP/support/libsophon/include/cnpy.h"  "$LIBSOPHON_INCLUDE/"


ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return
    fi
    sudo apt-get install -y pipx python3-venv 
    pipx install uv --force
    pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "已安装 uv，如终端仍未识别请执行 'source ~/.bashrc'" >&2
    fi
}

ensure_uv
uv sync


DOWNLOAD_MODE=${DOWNLOAD_LOBECHAT:-ask}

should_download_lobechat() {
    case "$DOWNLOAD_MODE" in
        yes|y|Y|true|1)
            return 0
            ;;
        no|n|N|false|0)
            return 1
            ;;
        ask)
            if [ -t 0 ]; then
                read -r -p "是否下载 lobechat docker 镜像？(Y/n，默认Y): " answer
                answer=${answer:-y}
                [[ "$answer" =~ ^([yY]|yes)$ ]]
                return
            else
                return 1
            fi
            ;;
        *)
            echo "无法识别的 DOWNLOAD_LOBECHAT=$DOWNLOAD_MODE，默认跳过" >&2
            return 1
            ;;
    esac
}

if should_download_lobechat; then
    echo "下载 lobechat docker 镜像..."
    mkdir -p "$SDK_TOP/samples/lobechat"
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        TARGET_FILE="$SDK_TOP/samples/lobechat/lobechat_x86.tar"
        URL="https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/lobechat/lobechat_x86.tar"
    else
        TARGET_FILE="$SDK_TOP/samples/lobechat/lobechat_aarch64.tar"
        URL="https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/lobechat/lobechat_aarch64.tar"
    fi
    wget -O "$TARGET_FILE" "$URL"
    echo "lobechat docker 镜像下载完成: $TARGET_FILE"
else
    echo "跳过下载 lobechat docker 镜像"
fi

TPU_MODE_CHOICE=${UNTOOL_MODE:-auto}
if [ "$TPU_MODE_CHOICE" = "auto" ]; then
    if [ -t 0 ]; then
        echo "请选择 TPU 模式："
        echo "1) AIPC (PCIe 模式)"
        echo "2) AirBox (SoC 模式)"
        read -r -p "请输入选择 (1 或 2，默认 2): " selection
    else
        selection=""
    fi
    case ${selection:-2} in
        1)
            TPU_MODE_CHOICE="pcie"
            ;;
        2)
            TPU_MODE_CHOICE="soc"
            ;;
        *)
            echo "无效选择，默认使用 SoC 模式" >&2
            TPU_MODE_CHOICE="soc"
            ;;
    esac
fi

case $TPU_MODE_CHOICE in
    pcie)
        echo "已选择 AIPC (PCIe 模式)"
        uv pip install "https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/tools/py310/sophon_arm_pcie-3.8.0-py3-none-any.whl"
        ;;
    soc)
        echo "已选择 AirBox (SoC 模式)"
        uv pip install "https://modelscope.cn/models/wlc952/aigchub_models/resolve/master/tools/py310/sophon_arm-3.8.0-py3-none-any.whl"
        ;;
    *)
        echo "UNTOOL_MODE=$TPU_MODE_CHOICE 无效，支持值: soc, pcie" >&2
        exit 1
        ;;
esac

export UNTOOL_MODE=$TPU_MODE_CHOICE

if ! grep -q "export UNTOOL_MODE=" "$HOME/.bashrc" 2>/dev/null; then
    echo "export UNTOOL_MODE=$TPU_MODE_CHOICE" >>"$HOME/.bashrc"
    echo "已将 UNTOOL_MODE=$TPU_MODE_CHOICE 添加到 ~/.bashrc"
else
    sed -i "s/export UNTOOL_MODE=.*/export UNTOOL_MODE=$TPU_MODE_CHOICE/" "$HOME/.bashrc"
    echo "已更新 ~/.bashrc 中的 UNTOOL_MODE=$TPU_MODE_CHOICE"
fi

echo "环境初始化完成！如需使 shell 立即生效，请运行 'source ~/.bashrc'。"