import os
import time
import json
import yaml
import uuid
import base64
import asyncio
import argparse
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse
from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "qwen3vl"

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

model_name = os.getenv("MODEL_NAME", "qwen3-vl-4b")

model_config = config["qwen3vl"].get(model_name)
if model_config is None:
    raise ValueError(
        f"模型名称 {model_name} 不存在, 请检查配置文件 config.yml。可用的模型名称: {list(config['qwen3vl'].keys())}"
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

        from repo.qwen3vl.python_demo.pipeline import Qwen3_VL

        self.llm_model = Qwen3_VL(args)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}

    async def destroy_app(self):
        del self.llm_model


router = AppInitializationRouter(app_name=app_name)


class ChatRequest(BaseModel):
    model: str = Field(model_name, description="Model name")
    messages: list = Field(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        description="Message",
    )
    stream: bool = Field(False, description="Stream response")


@router.post("/v1/chat/completions")
@change_dir(router.dir)
async def chat_completions(request: ChatRequest):
    slm = router.llm_model
    slm.ID_EOS = [slm.ID_IM_END, slm.ID_END]

    tmp_dir = f"{sdk_abs_path}/tmpdir"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_dir = os.path.abspath(tmp_dir)

    std_content = []
    media_type = ""

    if isinstance(request.messages[-1]["content"], list):
        content = request.messages[-1]["content"]
        for x in content:
            if x["type"] == "text":
                std_content.append(x)
            elif x["type"] == "image_url":
                image_url = x["image_url"]["url"]
                if image_url.startswith("data:"):
                    save_path = await base64_decode(image_url, tmp_dir)
                else:
                    save_path = await url_download(image_url, tmp_dir)
                std_content.append({"type": "image", "image": save_path})
                media_type = "image"
            elif x["type"] == "image":
                image_url = x["image"]
                if image_url.startswith("data:"):
                    save_path = await base64_decode(image_url, tmp_dir)
                else:
                    save_path = await url_download(image_url, tmp_dir)
                std_content.append({"type": "image", "image": save_path})
                media_type = "image"
            elif x["type"] == "video":
                video_url = x["video"]
                if isinstance(video_url, str):
                    save_path = await url_download(video_url, tmp_dir)
                    std_content.append(
                        {
                            "type": "video",
                            "video": save_path,
                            "fps": 1.0,
                            "min_pixels": 64 * 32 * 32,
                            "max_pixels": slm.model.MAX_PIXELS // 4,
                        }
                    )
                    media_type = "video"
    elif isinstance(request.messages[-1]["content"], str):
        std_content.append({"type": "text", "text": request.messages[-1]["content"]})
    else:
        return JSONResponse({"error": "Invalid content type"}, status_code=400)

    std_messages = [{"role": "user", "content": std_content}]
    inputs = slm.processor.apply_chat_template(std_messages,
                                            tokenize=True,
                                            add_generation_prompt=True,
                                            return_dict=True,
                                            return_tensors="pt")
    token_len = inputs.input_ids.numel()

    # Chat
    first_start = time.time()
    slm.model.forward_embed(inputs.input_ids.squeeze(0).tolist())
    if media_type == "image":
        vit_token_list = torch.where(inputs.input_ids == slm.ID_IMAGE_PAD)[1].tolist()
        vit_offset = vit_token_list[0]
        slm.vit_process_image(inputs, vit_offset)
        position_ids = slm.get_rope_index(
            inputs.input_ids, inputs.image_grid_thw, slm.ID_IMAGE_PAD
        )
        slm.max_posid = int(position_ids.max())
        ftoken = slm.forward_prefill(position_ids.numpy())
    elif media_type == "video":
        vit_token_list = torch.where(inputs.input_ids == slm.ID_VIDEO_PAD)[1].tolist()
        vit_offset = vit_token_list[0]
        slm.vit_process_video(inputs, vit_offset)
        position_ids = slm.get_rope_index(
            inputs.input_ids, inputs.video_grid_thw, slm.ID_VIDEO_PAD
        )
        slm.max_posid = int(position_ids.max())
        ftoken = slm.forward_prefill(position_ids.numpy())
    else:
        position_ids = 3 * [i for i in range(token_len)]
        slm.max_posid = token_len - 1
        ftoken = slm.forward_prefill(np.array(position_ids, dtype=np.int32))
    first_end = time.time()
    first_duration = first_end - first_start
    print(f"FTL: {first_duration:.3f} s")

    prompt_tokens = token_len
    if prompt_tokens >= slm.model.MAX_INPUT_LENGTH:
        return JSONResponse(
            {
                "error": "The maximum question length should be shorter than {} but we get {} instead.".format(
                    slm.model.MAX_INPUT_LENGTH, prompt_tokens
                )
            },
            status_code=400,
        )
    max_tokens = slm.model.SEQLEN - prompt_tokens

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

            while token not in slm.ID_EOS and completion_tokens < max_tokens:
                full_word_tokens.append(token)
                word = slm.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = slm.tokenizer.decode(
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
                slm.max_posid += 1
                position_ids = np.array(
                    [slm.max_posid, slm.max_posid, slm.max_posid], dtype=np.int32
                )
                token = slm.model.forward_next(position_ids)
                completion_tokens += 1
                await asyncio.sleep(0)
            slm.history_max_posid = slm.max_posid + 2
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

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    else:
        full_word_tokens = []
        completion_tokens = 0
        token = ftoken

        while token not in slm.ID_EOS and completion_tokens < max_tokens:
            completion_tokens += 1
            full_word_tokens.append(token)
            slm.max_posid += 1
            position_ids = np.array(
                [slm.max_posid, slm.max_posid, slm.max_posid], dtype=np.int32
            )
            token = slm.model.forward_next(position_ids)

        answer = slm.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
        slm.history_max_posid = slm.max_posid + 2
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
        return JSONResponse(data)


async def url_download(url, dir):
    src_ext = os.path.splitext(url.split("/")[-1])[1]
    ext = src_ext if src_ext else ".jpg"
    temp_path = os.path.join(dir, f"temp_{uuid.uuid4()}{ext}")
    final_path = os.path.join(dir, f"{uuid.uuid4()}{ext}")
    os.system(f"wget {url} -O {temp_path}")
    try:
        img = Image.open(temp_path)
        
        max_width, max_height = 560, 560
        if img.width > max_width or img.height > max_height:
            ratio = min(max_width / img.width, max_height / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)

            # 如果缩放比例很小，使用两步缩放以提高质量
            if ratio < 0.5:
                intermediate_width = int(img.width * 0.5)
                intermediate_height = int(img.height * 0.5)
                img = img.resize(
                    (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
                )
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        final_path_png = os.path.join(dir, f"{uuid.uuid4()}.png")
        img.save(final_path_png, format="PNG", optimize=True)
        img.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return final_path_png

    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        if os.path.exists(temp_path):
            os.rename(temp_path, final_path)
            return final_path
        return final_path


async def base64_decode(image_url, tmp_dir):
    base64_data = image_url.split(",")[1]
    img_bytes = base64.b64decode(base64_data)
    img = Image.open(BytesIO(img_bytes))

    max_width, max_height = 560, 560
    if img.width > max_width or img.height > max_height:
        ratio = min(max_width / img.width, max_height / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)

        # 如果缩放比例很小，使用两步缩放以提高质量
        if ratio < 0.5:
            intermediate_width = int(img.width * 0.5)
            intermediate_height = int(img.height * 0.5)
            img = img.resize(
                (intermediate_width, intermediate_height), Image.Resampling.LANCZOS
            )
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    save_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
    img.save(save_path, format="PNG", optimize=True)
    return save_path
