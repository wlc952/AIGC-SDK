import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "img2txt"


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        from repo.img2txt.img_speaking_pipeline import (
            ImageSpeakingPipeline as ISPipeline,
        )

        self.register_model("img2txt", ISPipeline())
        return {
            "message": f"Application {self.app_name} has been initialized successfully."
        }


router = AppInitializationRouter(app_name=app_name)


### img2txt；兼容openai api，image/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def get_img_caption(
    image: UploadFile = File(...),
):
    pipeline = router.require_model("img2txt")

    tmp_dir = Path(sdk_abs_path) / "tmpdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=tmp_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="未检测到有效的图片内容")

        try:
            with Image.open(BytesIO(image_bytes)) as pil_image:
                pil_image.convert("RGB").save(tmp_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"无法解析上传的图片: {exc}") from exc

        captions, tags = pipeline(str(tmp_path), num_return_sequences=1)
        content = {"data": [{"captions": captions, "tags": tags}]}
        return JSONResponse(content=jsonable_encoder(content), status_code=200)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
