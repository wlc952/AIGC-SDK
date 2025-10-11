import os, io, asyncio
from pathlib import Path
from fastapi import HTTPException, Response
from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path
from fastapi import File, Form, UploadFile
import soundfile as sf
import logging
from typing import Optional
import uuid

app_name = "indextts"
logging.basicConfig(level=logging.INFO)

SDK_ROOT = Path(sdk_abs_path).resolve()
DEFAULT_REF_AUDIO = (SDK_ROOT / "repo" / "indextts" / "ref.wav").resolve()
TMP_DIR = SDK_ROOT / "tmpdir"


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        from repo.indextts.infer import IndexTTS

        tts_engine = IndexTTS(
            cfg_path=f"{sdk_abs_path}/bmodels/{app_name}/config.yaml",
            model_dir=f"{sdk_abs_path}/bmodels/{app_name}"
        )
        self.register_model("indextts", tts_engine)
        return {"message": f"应用 {self.app_name} 已成功初始化。"}


router = AppInitializationRouter(app_name=app_name)


async def _persist_uploaded_audio(upload: UploadFile) -> Optional[Path]:
    """Save uploaded audio to tmpdir and return its path."""
    file_bytes = await upload.read()
    if not file_bytes:
        return None

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TMP_DIR / f"{uuid.uuid4()}.wav"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    await upload.seek(0)
    return tmp_path.resolve()


def _resolve_local_ref_path(ref_audio_path: Optional[str]) -> Optional[Path]:
    """Validate a client-provided local path and return it when safe."""
    if not ref_audio_path:
        return None

    candidate = Path(ref_audio_path).expanduser()
    try:
        candidate = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        return None

    try:
        # Ensure the path is within the SDK root to avoid directory traversal
        candidate.relative_to(SDK_ROOT)
    except ValueError:
        logging.warning("拒绝访问 SDK 根目录之外的路径: %s", candidate)
        return None

    return candidate if candidate.is_file() else None


@router.post("/v0/audio/speech")
@change_dir(router.dir)
async def indextts(
    text: str = Form("text"),
    ref_audio_file: Optional[UploadFile] = File(None),  # 可选上传文件
    ref_audio_path: Optional[str] = Form(None),  # 客户端可选本地路径
):
    cleanup_paths = []
    try:
        normalized_text = (text or "").strip()
        if not normalized_text:
            raise HTTPException(status_code=400, detail="text 不能为空")

        tts = router.require_model("indextts")

        ref_path: Optional[Path] = None

        if ref_audio_file is not None:
            uploaded_path = await _persist_uploaded_audio(ref_audio_file)
            if uploaded_path is not None:
                ref_path = uploaded_path
                cleanup_paths.append(uploaded_path)
            else:
                logging.warning("收到空的参考音频上传文件，已忽略。")

        if ref_path is None and ref_audio_path:
            ref_path = _resolve_local_ref_path(ref_audio_path)
            if ref_path is None:
                raise HTTPException(status_code=404, detail="无法访问提供的参考音频路径")

        ref_path = ref_path or DEFAULT_REF_AUDIO

        if not ref_path.is_file():
            raise HTTPException(status_code=404, detail=f"参考音频文件不存在: {ref_path}")

        sr, np_audio = await asyncio.to_thread(
            tts.infer,
            audio_prompt=str(ref_path),
            text=normalized_text,
            output_path=None,
            verbose=False,
        )

        # 写入到内存并返回
        wav_buffer = io.BytesIO()
        sf.write(file=wav_buffer, data=np_audio, samplerate=sr, format="WAV")
        wav_buffer.seek(0)

        return Response(content=wav_buffer.getvalue(), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("indextts error")
        return {"error": str(e), "info": "处理过程中出现错误"}
    finally:
        for tmp_path in cleanup_paths:
            try:
                if tmp_path.exists():
                    os.remove(tmp_path)
            except Exception:
                pass

