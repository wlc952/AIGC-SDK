import asyncio
import json
import re
import time
from pathlib import Path
from typing import List, Optional

from asyncio.subprocess import PIPE, STDOUT

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path


app_name = "sherpa"


async def run_sherpa_command(command: List[str]) -> dict:
    pattern = re.compile(r"\{.*?\}")
    result_payload = {"text": ""}

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=PIPE,
        stderr=STDOUT,
    )

    assert process.stdout is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace")
        matches = pattern.findall(decoded)
        for match in matches:
            try:
                result_payload = json.loads(match)
            except json.JSONDecodeError:
                continue

    returncode = await process.wait()
    if returncode != 0:
        raise RuntimeError(f"Sherpa 推理进程退出码异常: {returncode}")

    return result_payload


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        command = [
            "build/bin/sherpa-onnx",
            f"--tokens={sdk_abs_path}/bmodels/sherpa/tokens.txt",
            f"--zipformer2-ctc-model={sdk_abs_path}/bmodels/sherpa/zipformer2_ctc_F32.bmodel",
        ]
        self.register_model("sherpa", command)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


### ASR；兼容openai api，audio/transcriptions
@router.post("/v1/audio/transcriptions")
@change_dir(router.dir)
async def sherpa(
    file: UploadFile = File(...),
    response_format: Optional[str] = Form("text"),
):
    if response_format not in {"text", "json"}:
        raise HTTPException(status_code=400, detail="response_format 仅支持 text 或 json")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传的音频为空")

    tmp_dir = Path(sdk_abs_path) / "tmpdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_tmp_path = tmp_dir / f"sherpa_{int(time.time())}.wav"
    file_tmp_path.write_bytes(payload)

    command = list(router.require_model("sherpa"))
    command_with_audio = [*command, str(file_tmp_path)]

    audio_start_time = time.time()
    try:
        result = await run_sherpa_command(command_with_audio)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Sherpa 执行失败: {exc}") from exc
    finally:
        file_tmp_path.unlink(missing_ok=True)
        await file.close()

    total_time = time.time() - audio_start_time
    print(f"Total time: {total_time}")

    if response_format == "text":
        return PlainTextResponse(content=result.get("text", ""))

    return JSONResponse(content=result)


# #### 测试命令
# curl http://localhost:8000/v1/audio/transcriptions \
#   -F 'file=@/data/TEST.wav;type=audio/wav' \
#   -F 'response_format=json'
