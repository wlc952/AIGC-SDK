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
import numpy as np
import torch
import yaml
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from qwen_vl_utils import process_vision_info

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "qwen2_5vl"

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

model_name = os.getenv("MODEL_NAME", "qwen2.5-vl-3b")

model_config = config["qwen2_5vl"].get(model_name)
if model_config is None:
    raise ValueError(
        f"模型名称 {model_name} 不存在, 请检查配置文件 config.yml。可用的模型名称: {list(config['qwen2_5vl'].keys())}"
    )

model_path = os.path.join(sdk_abs_path, model_config["model_path"])
token_path = os.path.join(sdk_abs_path, model_config["token_path"])


class AppInitializationRouter(BaseAPIRouter):
    dir = f"{sdk_abs_path}/repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        args = argparse.Namespace(
            model_path=model_path,
            config_path=token_path,
            devid=0,
        )

        from repo.qwen2_5vl.python_demo.pipeline import Qwen2_5VL

        pipeline = Qwen2_5VL(args)
        self.register_model("qwen2_5vl", pipeline)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


class ImageURL(BaseModel):
    url: str


class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageURL] = Field(None, alias="image_url")
    image: Optional[str] = None
    video: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class ChatMessage(BaseModel):
    role: str
    content: Union[List[MessageContent], str]


class ChatRequest(BaseModel):
    model: str = Field(model_name, description="Model name")
    messages: List[ChatMessage] = Field(..., description="Message")
    stream: bool = Field(False, description="Stream response")


@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    pipeline = router.require_model("qwen2_5vl")
    pipeline.ID_EOS = [pipeline.ID_IM_END, pipeline.ID_END]

    tmp_dir = Path(sdk_abs_path) / "tmpdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cleanup_paths: List[Path] = []
    std_content: list[dict] = []
    media_type = ""

    message = request.messages[-1]

    async def handle_image_source(raw: str, client: httpx.AsyncClient) -> str:
        if raw.startswith("data:"):
            try:
                base64_data = raw.split(",", 1)[1]
            except IndexError as exc:
                raise HTTPException(status_code=400, detail="图片数据URL格式错误") from exc
            img_bytes = base64.b64decode(base64_data)
        else:
            resp = await client.get(raw)
            resp.raise_for_status()
            img_bytes = resp.content

        try:
            with Image.open(BytesIO(img_bytes)) as img:
                processed = resize_image(img)
                processed_format = processed.format or "PNG"
                save_path = tmp_dir / f"{uuid.uuid4()}.{processed_format.lower()}"
                processed.save(save_path, format=processed_format, optimize=True)
                processed.close()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"图片处理失败: {exc}") from exc

        cleanup_paths.append(save_path)
        return str(save_path)

    async def handle_video_source(raw: str, client: httpx.AsyncClient) -> str:
        if raw.startswith("data:"):
            try:
                base64_data = raw.split(",", 1)[1]
            except IndexError as exc:
                raise HTTPException(status_code=400, detail="视频数据URL格式错误") from exc
            video_bytes = base64.b64decode(base64_data)
            ext = ".mp4"
        else:
            resp = await client.get(raw)
            resp.raise_for_status()
            video_bytes = resp.content
            ext = Path(raw.split("?")[0]).suffix or ".mp4"
        save_path = tmp_dir / f"{uuid.uuid4()}{ext}"
        save_path.write_bytes(video_bytes)
        cleanup_paths.append(save_path)
        return str(save_path)

    try:
        if isinstance(message.content, list):
            async with httpx.AsyncClient(timeout=30) as client:
                for item in message.content:
                    if item.type == "text" and item.text is not None:
                        std_content.append({"type": "text", "text": item.text})
                    elif item.type in {"image_url", "image"}:
                        source = item.image_url.url if item.image_url else item.image
                        if not source:
                            raise HTTPException(status_code=400, detail="图片地址缺失")
                        save_path = await handle_image_source(source, client)
                        std_content.append({"type": "image", "image": save_path})
                        media_type = "image"
                    elif item.type == "video" and item.video is not None:
                        save_path = await handle_video_source(item.video, client)
                        std_content.append(
                            {
                                "type": "video",
                                "video": save_path,
                                "fps": 1.0,
                                "min_pixels": 64 * 28 * 28,
                                "max_pixels": pipeline.model.MAX_PIXELS // 4,
                            }
                        )
                        media_type = "video"
                    else:
                        raise HTTPException(status_code=400, detail=f"不支持的消息类型: {item.type}")
        elif isinstance(message.content, str):
            std_content.append({"type": "text", "text": message.content})
        else:
            raise HTTPException(status_code=400, detail="Invalid content type")
    except Exception:
        cleanup(cleanup_paths)
        raise

    std_messages = [{"role": "user", "content": std_content}]

    text = pipeline.processor.apply_chat_template(
        std_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(std_messages)
    inputs = pipeline.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    token_len = inputs.input_ids.numel()

    # Chat
    first_start = time.time()
    pipeline.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
    if media_type == "image":
        vit_token_list = torch.where(inputs.input_ids == pipeline.ID_IMAGE_PAD)[1].tolist()
        vit_offset = vit_token_list[0]
        pipeline.vit_process_image(inputs, vit_offset)
        position_ids = pipeline.get_rope_index(
            inputs.input_ids, inputs.image_grid_thw, pipeline.ID_IMAGE_PAD
        )
        pipeline.max_posid = int(position_ids.max())
        ftoken = pipeline.forward_prefill(position_ids.numpy())
    elif media_type == "video":
        vit_token_list = torch.where(inputs.input_ids == pipeline.ID_VIDEO_PAD)[1].tolist()
        vit_offset = vit_token_list[0]
        pipeline.vit_process_video(inputs, vit_offset)
        position_ids = pipeline.get_rope_index(
            inputs.input_ids, inputs.video_grid_thw, pipeline.ID_VIDEO_PAD
        )
        pipeline.max_posid = int(position_ids.max())
        ftoken = pipeline.forward_prefill(position_ids.numpy())
    else:
        position_ids = 3 * [i for i in range(token_len)]
        pipeline.max_posid = token_len - 1
        ftoken = pipeline.forward_prefill(np.array(position_ids, dtype=np.int32))
    first_end = time.time()
    first_duration = first_end - first_start
    print(f"FTL: {first_duration:.3f} s")

    prompt_tokens = token_len
    if prompt_tokens >= pipeline.model.MAX_INPUT_LENGTH:
        cleanup(cleanup_paths)
        return JSONResponse(
            {
                "error": "The maximum question length should be shorter than {} but we get {} instead.".format(
                    pipeline.model.MAX_INPUT_LENGTH, prompt_tokens
                )
            },
            status_code=400,
        )
    max_tokens = pipeline.model.SEQLEN - prompt_tokens

    created = int(time.time())
    comp_id = f"chatcmpl-{uuid.uuid4().hex}"

    # Following tokens
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

            token = ftoken
            full_word_tokens = []

            while token not in pipeline.ID_EOS and completion_tokens < max_tokens:
                full_word_tokens.append(token)
                word = pipeline.tokenizer.decode(
                    full_word_tokens, skip_special_tokens=True
                )
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = pipeline.tokenizer.decode(
                            [token, token], skip_special_tokens=True
                        )[len(pre_word) :]
                    chunk = {
                        "id": comp_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    full_word_tokens = []
                pipeline.max_posid += 1
                position_ids = np.array(
                    [pipeline.max_posid, pipeline.max_posid, pipeline.max_posid], dtype=np.int32
                )
                token = pipeline.model.forward_next(position_ids)
                completion_tokens += 1
                await asyncio.sleep(0)
            pipeline.history_max_posid = pipeline.max_posid + 2
            next_end = time.time()
            next_duration = next_end - first_end
            tps = completion_tokens / next_duration
            print(f"TPS: {tps:.3f} token/s")

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
        response.background = BackgroundTask(cleanup, list(cleanup_paths))
        return response

    else:
        full_word_tokens = []
        completion_tokens = 0
        token = ftoken

        while token not in pipeline.ID_EOS and completion_tokens < max_tokens:
            completion_tokens += 1
            full_word_tokens.append(token)
            pipeline.max_posid += 1
            position_ids = np.array(
                [pipeline.max_posid, pipeline.max_posid, pipeline.max_posid], dtype=np.int32
            )
            token = pipeline.model.forward_next(position_ids)

        answer = pipeline.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
        pipeline.history_max_posid = pipeline.max_posid + 2
        next_end = time.time()
        next_duration = next_end - first_end
        tps = completion_tokens / next_duration
        print(f"TPS: {tps:.3f} token/s")

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


def resize_image(img: Image.Image) -> Image.Image:
    max_width, max_height = 560, 560
    if img.width <= max_width and img.height <= max_height:
        return img.convert("RGB") if img.mode not in {"RGB", "RGBA"} else img

    ratio = min(max_width / img.width, max_height / img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    resized = img
    if ratio < 0.5:
        intermediate_width = max(1, int(img.width * 0.5))
        intermediate_height = max(1, int(img.height * 0.5))
        resized = resized.resize(
            (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
        )
    resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
    if resized.mode not in {"RGB", "RGBA"}:
        resized = resized.convert("RGB")
    return resized


def cleanup(paths: List[Path]):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
