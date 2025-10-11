import argparse
import asyncio
import base64
import json
import os
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import httpx
import yaml
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from untool import MiniCPMVPipeline

from api.base_api import BaseAPIRouter, sdk_abs_path


app_name = "minicpmv"

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

model_name = os.getenv("MODEL_NAME", "minicpm3o")

model_config = config["minicpmv"].get(model_name)
if model_config is None:
    raise ValueError(
        f"模型名称 {model_name} 不存在, 请检查配置文件 config.yml。可用的模型名称: {list(config['minicpmv'].keys())}"
    )

model_path = os.path.join(sdk_abs_path, model_config["model_path"])
token_path = os.path.join(sdk_abs_path, model_config["token_path"])


class AppInitializationRouter(BaseAPIRouter):
    async def init_app(self):
        args = argparse.Namespace(
            model_path=model_path,
            tokenizer_path=token_path,
            devid=0,
            generation_mode="greedy",
            enable_history=False,
        )
        self.register_model("minicpmv", MiniCPMVPipeline(args))
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


class ImageURL(BaseModel):
    url: str


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = Field(None, alias="image_url")
    image: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[List[MessageContent], str]


class ChatRequest(BaseModel):
    model: str = Field(model_name, description="model name")
    messages: List[ChatMessage] = Field(..., description="Message")
    stream: bool = Field(False, description="Stream response")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    slm = router.require_model("minicpmv")
    slm.input_str = ""
    slm.image_str = []

    tmp_dir = Path(sdk_abs_path) / "tmpdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 处理输入消息
    cleanup_paths: list[Path] = []
    message = request.messages[-1]
    if isinstance(message.content, list):
        for item in message.content:
            if item.type == "text" and item.text is not None:
                slm.input_str = item.text
            elif item.type == "image_url" and item.image_url is not None:
                image_url = item.image_url.url
                try:
                    if image_url.startswith("data:"):
                        try:
                            base64_data = image_url.split(",", 1)[1]
                        except IndexError as exc:
                            raise HTTPException(status_code=400, detail="图片数据URL格式错误") from exc
                        img_bytes = base64.b64decode(base64_data)
                    else:
                        async with httpx.AsyncClient(timeout=30) as client:
                            resp = await client.get(image_url)
                            resp.raise_for_status()
                        img_bytes = resp.content

                    with Image.open(BytesIO(img_bytes)) as img:
                        ext = (img.format or "jpeg").lower()
                        save_path = tmp_dir / f"{uuid.uuid4()}.{ext}"
                        img.convert("RGB").save(save_path)

                    slm.image_str.append(str(save_path))
                    cleanup_paths.append(save_path)
                except (httpx.HTTPError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=f"图片下载失败: {exc}") from exc
                except HTTPException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise HTTPException(status_code=400, detail=f"图片处理失败: {exc}") from exc
    elif isinstance(message.content, str):
        slm.input_str = message.content
    else:
        raise HTTPException(status_code=400, detail="Invalid content type")

    # 编码处理
    if slm.image_str:
        missing_images = [x for x in slm.image_str if not os.path.exists(x)]
        if missing_images:
            print(f"Missing images: {missing_images}")
            slm.encode()
        else:
            slm.patch_num = len(slm.image_str)
            slm.encode_with_image()
    else:
        slm.encode()

    # 计算prompt tokens
    prompt_tokens = len(slm.input_ids)
    if prompt_tokens >= slm.model.SEQLEN:
        cleanup(cleanup_paths)
        return JSONResponse(
            {"error": "Input length exceeds maximum sequence length."}, status_code=400
        )
    max_tokens = slm.model.SEQLEN - prompt_tokens

    created = int(time.time())
    comp_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:

        async def event_stream():
            completion_tokens = 0
            # ----- (0) assistant role 首包 -----
            head = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"

            # ----- (1) 逐 token 增量包 -----
            token = slm.model.forward_first(
                slm.input_ids, slm.pixel_values, slm.image_offsets, slm.patch_num
            )
            buf: list[int] = []

            while token not in slm.ID_EOS and completion_tokens < max_tokens:
                completion_tokens += 1
                buf.append(token)
                piece = slm.tokenizer.decode(buf, skip_special_tokens=True)
                if "�" not in piece:
                    chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    buf.clear()

                token = slm.model.forward_next()
                await asyncio.sleep(0)

            # ----- (2) 尾包 stop + usage -----
            tail = {
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        response = StreamingResponse(event_stream(), media_type="text/event-stream")
        response.background = cleanup_background(cleanup_paths)
        return response

    else:
        buf: list[int] = []
        completion_tokens = 0
        token = slm.model.forward_first(
            slm.input_ids, slm.pixel_values, slm.image_offsets, slm.patch_num
        )

        while token not in slm.ID_EOS and completion_tokens < max_tokens:
            completion_tokens += 1
            buf.append(token)
            token = slm.model.forward_next()

        answer = slm.tokenizer.decode(buf, skip_special_tokens=True)
        data = {
            "id": comp_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        cleanup(cleanup_paths)
        return JSONResponse(data)


def cleanup(paths: List[Path]):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_background(paths: List[Path]):
    return BackgroundTask(cleanup, list(paths))
