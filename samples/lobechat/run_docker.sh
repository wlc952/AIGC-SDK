#!/bin/bash

# 加载Docker镜像
echo "正在加载Docker镜像..."

ARCH=$(uname -m)
if [ "$ARCH" == "x86_64" ]; then
    # 使用更可靠的方式提取镜像ID
    LOAD_OUTPUT=$(docker load -i lobechat_x86.tar)
    echo "$LOAD_OUTPUT"
    # 正确提取镜像ID - 完整保留sha256:前缀
    IMAGE_ID=$(echo "$LOAD_OUTPUT" | grep -o "sha256:[a-f0-9]\{64\}")
else
    # 使用更可靠的方式提取镜像ID
    LOAD_OUTPUT=$(docker load -i lobechat_aarch64.tar)
    echo "$LOAD_OUTPUT"
    # 正确提取镜像ID - 完整保留sha256:前缀
    IMAGE_ID=$(echo "$LOAD_OUTPUT" | grep -o "sha256:[a-f0-9]\{64\}")
fi

# 确保我们有镜像ID
if [ -z "$IMAGE_ID" ]; then
    echo "错误：无法获取Docker镜像ID，请检查加载过程。"
    exit 1
fi

echo "Docker镜像加载完成，镜像ID: $IMAGE_ID"

# 清理已存在的同名容器（如果存在）
if docker ps -a --format '{{.Names}}' | grep -q '^lobe-chat$'; then
    echo "检测到已存在的容器 lobe-chat，正在清理..."
    docker stop lobe-chat >/dev/null 2>&1
    docker rm lobe-chat >/dev/null 2>&1
fi

# 运行容器（按需添加参数）
echo "正在启动容器 lobe-chat..."
docker run -d -p 3210:3210 \
    -e OPENAI_API_KEY=sk-xxxx \
    -e OPENAI_PROXY_URL=http://localhost:8000/v1 \
    -e ACCESS_CODE=lobe66 \
    --name lobe-chat \
    "$IMAGE_ID"

# 验证容器状态
if docker ps --format '{{.Names}}' | grep -q '^lobe-chat$'; then
    echo "容器 lobe-chat 启动成功！"
    echo "使用以下命令查看日志：docker logs lobe-chat"
else
    echo "警告：容器启动异常，请检查日志：docker logs lobe-chat"
    exit 1
fi
