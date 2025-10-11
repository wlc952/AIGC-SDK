#!/bin/bash
set -e
set -u
set -o pipefail

SDK_TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# 确保提供了至少一个应用名称
if [ $# -eq 0 ]; then
    echo "Usage: $0 app_name [app_name2 ...]"
    exit 1
fi

# 定义存放仓库的目录
repo_dir="repo"

# 遍历所有提供的应用名称
for app_name in "$@"; do
    echo "Processing $app_name..."

    if [ ! -d "$repo_dir/$app_name" ]; then
        echo "Repository $app_name not found in $repo_dir"
        exit 1
    fi

    pushd "$repo_dir/$app_name"

    # 执行仓库中的下载脚本
    if [ -f "download.sh" ]; then
        echo "Running download.sh for $app_name..."
        chmod +x download.sh
        ./download.sh
    fi

    popd
done

echo "All specified apps processed."