import base64
import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import soundfile as sf
import yaml
from fastapi import HTTPException, Response
from fastapi import File, UploadFile
from pydantic import BaseModel, Field
from pydub import AudioSegment

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "emotivoice"

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

VOICE = config["emotivoice"]["voice"]


def convert(
    src_wav,
    tgt_wav,
    tone_color_converter,
    get_se,
    save_path="./temp/output.wav",
    encode_message="",
):
    try:
        # extract the tone color features of the source speaker and target speaker
        source_se, _ = get_se(
            src_wav, tone_color_converter, target_dir="processed", vad=True
        )
        target_se, _ = get_se(
            tgt_wav, tone_color_converter, target_dir="processed", vad=True
        )
    except Exception as e:
        return {"error": f"Failed to extract speaker embedding: {e}"}
    tone_color_converter.convert(
        audio_src_path=src_wav,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )
    return save_path


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        from repo.emotivoice.demo_page import get_models

        models, tone_color_converter, g2p, lexicon = get_models()
        self.register_model(
            "emotivoice",
            {
                "models": models,
                "tone_color_converter": tone_color_converter,
                "g2p": g2p,
                "lexicon": lexicon,
            },
        )
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


### 音色转换；兼容openai api，audio/translation
@router.post("/v1/audio/translation")
@change_dir(router.dir)
async def voice_changer(
    file: UploadFile = File(...),
    ref_file: UploadFile = File(...),  # 比 OpenAI 多一个参数
):
    from repo.emotivoice.tone_color_conversion import get_se
    import aiofiles

    resources = router.require_model("emotivoice")
    src_tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", dir=f"{sdk_abs_path}/tmpdir", delete=False
    )
    tgt_tmp = tempfile.NamedTemporaryFile(
        suffix=".wav", dir=f"{sdk_abs_path}/tmpdir", delete=False
    )
    src_tmp.close()
    tgt_tmp.close()
    cleanup: list[Path] = []

    try:
        async with aiofiles.open(src_tmp.name, "wb") as buffer:
            await buffer.write(await file.read())
        cleanup.append(Path(src_tmp.name))

        async with aiofiles.open(tgt_tmp.name, "wb") as buffer:
            await buffer.write(await ref_file.read())
        cleanup.append(Path(tgt_tmp.name))

        save_path = convert(
            src_wav=src_tmp.name,
            tgt_wav=tgt_tmp.name,
            tone_color_converter=resources["tone_color_converter"],
            get_se=get_se,
            encode_message="Airbox",
        )
        if isinstance(save_path, dict):
            raise HTTPException(status_code=500, detail=save_path["error"])

        with open(save_path, "rb") as audio_file:
            audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode()
        return {"text": audio_base64, "info": "text is the base64 encoded audio"}
    finally:
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


### 文本转语音 以及 音色克隆；兼容openai api，audio/speech
class TTSRequest(BaseModel):
    input: str = Field(..., description="要转换为语音的文本")
    response_format: Optional[str] = Field("wav", description="音频格式")

    emotion: Optional[str] = Field(
        "", description="情感提示（可选）【普通、生气、开心、惊讶、悲伤、厌恶、恐惧】"
    )
    audio_path: Optional[str] = Field("", description="要参考的目标音色路径（可选）")


@router.post("/v1/audio/speech")
@change_dir(router.dir)
async def text_to_speech(request: TTSRequest):
    from repo.emotivoice.demo_page import tts
    from repo.emotivoice.tone_color_conversion import get_se

    allowed_formats = {"wav", "mp3", "flac", "ogg"}
    response_format = (request.response_format or "wav").lower()
    if response_format not in allowed_formats:
        raise HTTPException(status_code=400, detail="不支持的音频格式")

    resources = router.require_model("emotivoice")
    voice = os.getenv("VOICE", VOICE)

    tmp_output = Path(sdk_abs_path) / "tmpdir" / f"{uuid.uuid4()}.wav"
    tmp_output.parent.mkdir(parents=True, exist_ok=True)
    cleanup_paths: list[Path] = [tmp_output]

    tts(
        request.input,
        request.emotion,
        voice,
        str(tmp_output),
        resources["models"],
        resources["g2p"],
        resources["lexicon"],
    )

    save_path = tmp_output
    if request.audio_path:
        target_path = Path(request.audio_path)
        if not target_path.exists():
            raise HTTPException(status_code=400, detail="参考音色文件不存在")

        converted_output = Path(sdk_abs_path) / "tmpdir" / f"{uuid.uuid4()}.wav"
        cleanup_paths.append(converted_output)

        converted_path = convert(
            src_wav=str(save_path),
            tgt_wav=str(target_path),
            tone_color_converter=resources["tone_color_converter"],
            get_se=get_se,
            save_path=str(converted_output),
            encode_message="Airbox",
        )
        if isinstance(converted_path, dict):
            raise HTTPException(status_code=500, detail=converted_path["error"])

        save_path = Path(converted_path)
        cleanup_paths.append(save_path)

    np_audio, sr = sf.read(str(save_path))
    wav_buffer = io.BytesIO()
    sf.write(file=wav_buffer, data=np_audio, samplerate=sr, format="WAV")
    wav_buffer.seek(0)

    buffer = wav_buffer
    if response_format != "wav":
        wav_audio = AudioSegment.from_wav(wav_buffer)
        wav_audio.frame_rate = sr
        buffer = io.BytesIO()
        wav_audio.export(buffer, format=response_format)

    try:
        return Response(
            content=buffer.getvalue(),
            media_type=f"audio/{response_format}",
            headers={"Cache-Control": "no-store"},
        )
    finally:
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


# 测试用指令
# curl http://0.0.0.0:8000/v1/audio/speech -H "Content-Type: application/json" \
# -d '{
#   "input": "大家好啊",
#   "voice": "8051",
#   "response_format": "wav",
#   "model": "emotivoice",
#   "speed": 1,
#   "emotion": "",
#   "audio_path": ""
# }' --output speech.wav
