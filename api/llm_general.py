import os
import time
import json
import yaml
import uuid
import asyncio
import argparse
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List

from untool import EngineLLM
from untool.bindings.wrapper import llm_forward_first, llm_forward_next, llm_get_seq_len

from api.base_api import BaseAPIRouter, sdk_abs_path

app_name = "llm_general"

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

model_name = os.getenv("MODEL_NAME", "qwen3-1.7b")

model_config = config["llm"].get(model_name)
if model_config is None:
    raise ValueError(
        f"模型名称 {model_name} 不存在, 请检查配置文件 config.yml。可用的模型名称: {list(config['llm'].keys())}"
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
        self.register_model("llm_general", EngineLLM(args))
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    model: str = Field(model_name, description="model name")
    messages: List[ChatMessage] = Field(
        [ChatMessage(role="user", content="hello")], description="Chat history"
    )
    stream: bool = Field(False, description="Stream response")
    enable_thinking: bool = Field(False, description="Enable thinking")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    slm = router.require_model("llm_general")

    chat_messages = [message.model_dump() for message in request.messages]

    input_ids = slm.tokenizer.apply_chat_template(
        chat_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=request.enable_thinking,
    )
    prompt_tokens = len(input_ids)
    SEQLEN = llm_get_seq_len(slm.llmbase)
    if prompt_tokens >= SEQLEN:
        return JSONResponse(
            {"error": "Input length exceeds maximum sequence length."}, status_code=400
        )
    max_tokens = SEQLEN - prompt_tokens

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
            token = llm_forward_first(slm.llmbase, input_ids, prompt_tokens)
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

                token = llm_forward_next(slm.llmbase)
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

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    else:
        buf: list[int] = []
        completion_tokens = 0
        token = llm_forward_first(slm.llmbase, input_ids, prompt_tokens)

        while token not in slm.ID_EOS and completion_tokens < max_tokens:
            completion_tokens += 1
            buf.append(token)
            token = llm_forward_next(slm.llmbase)

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
        return JSONResponse(data)
