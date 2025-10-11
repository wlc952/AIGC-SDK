import asyncio
import shutil
from typing import Optional

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from repo.bmwhisper.transcribe import transcribe
from starlette import status

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "bmwhisper"


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        from repo.bmwhisper import load_model

        args = {}
        args["model_name"] = "base"
        args["bmodel_dir"] = f"{sdk_abs_path}/bmodels/{app_name}"
        args["beam_size"] = 5
        args["padding_size"] = 448
        args["dev_id"] = 0
        model = load_model(args)
        self.register_model("whisper", model)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


### ASR；兼容openai api，audio/transcriptions
@router.post("/v1/audio/transcriptions")
@change_dir(router.dir)
async def whisper(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    word_timestamps: Optional[bool] = Form(False),
):
    model = router.require_model("whisper")

    # init whisper parameters
    language = None if language in ["", "string"] else language
    prompt = None if prompt in ["", "string"] else prompt

    # Load the audio
    audio = await load_audio(file)

    args = {
        "verbose": True,
        "task": "transcribe",
        "language": language,
        "best_of": 5,
        "beam_size": 5,
        "patience": None,
        "length_penalty": None,
        "suppress_tokens": "-1",
        "initial_prompt": prompt,
        "condition_on_previous_text": True,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": word_timestamps,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "padding_size": 448,
 
    }

    # Transcribe the audio
    result = transcribe(model, audio, **args)

    # 响应格式
    if response_format == "text":
        return PlainTextResponse(content=result["text"])
    else:
        return JSONResponse(content=result)


async def load_audio(file: UploadFile, sr: int = 16000):
    if shutil.which("ffmpeg") is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ffmpeg 未安装，无法进行音频格式转换",
        )

    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未检测到有效的音频内容",
        )

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:1",
    ]

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(data)

    if process.returncode != 0:
        error_message = stderr.decode(errors="ignore") or "ffmpeg 转码失败"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message.strip(),
        )

    return np.frombuffer(stdout, np.int16).astype(np.float32) / 32768.0


#### 测试命令
# curl http://localhost:8000/v0/audio/transcriptions \
#   -F 'file=@/data/AigcHub-TPU/repo/datasets/test/demo.wav;type=audio/wav' \
#   -F 'model=base'
