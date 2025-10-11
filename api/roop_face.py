import base64
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path
from repo.roop_face.roop import setup_model, swap_face
from repo.roop_face.roop.inswappertpu import INSwapper

app_name = "roop_face"


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        face_swapper = INSwapper(f"{sdk_abs_path}/bmodels/roop_face")
        restorer = setup_model(f"{sdk_abs_path}/bmodels/roop_face")
        self.register_model("face_swapper", face_swapper)
        self.register_model("restorer", restorer)
        return {
            "message": f"Application {self.app_name} has been initialized successfully."
        }


router = AppInitializationRouter(app_name=app_name)


### 图像变换；兼容openai api，images/variations
@router.post("/v1/images/variations")
@change_dir(router.dir)
async def face_swap(
    image: UploadFile = File(...),
    target_img: UploadFile = File(...),  # 比openai多了一个参数
):
    face_swapper = router.require_model("face_swapper")

    src_image_bytes = await image.read()
    tar_image_bytes = await target_img.read()
    if not src_image_bytes or not tar_image_bytes:
        raise HTTPException(status_code=400, detail="源图像或目标图像为空")

    try:
        with (
            Image.open(BytesIO(src_image_bytes)) as src_image,
            Image.open(BytesIO(tar_image_bytes)) as tar_image,
        ):
            result_image = swap_face(face_swapper, src_image, tar_image)

            buffer = BytesIO()
            result_image.save(buffer, format="JPEG")
            ret_img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"人脸替换失败: {exc}") from exc

    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(
        content=jsonable_encoder(content), media_type="application/json"
    )


### 图像增强；兼容openai api，images/edit
@router.post("/v1/images/edit")
@change_dir(router.dir)
async def face_enhance(
    image: UploadFile = File(...),
    restorer_visibility: Optional[float] = Form(1.0),
):
    if restorer_visibility is None:
        restorer_visibility = 1.0
    restorer_visibility = float(restorer_visibility)
    if not 0.0 <= restorer_visibility <= 1.0:
        raise HTTPException(
            status_code=400, detail="restorer_visibility 需介于 0 和 1 之间"
        )

    restorer = router.require_model("restorer")

    ori_image_bytes = await image.read()
    if not ori_image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像为空")

    try:
        with Image.open(BytesIO(ori_image_bytes)) as ori_image:
            if ori_image.mode != "RGB":
                ori_image = ori_image.convert("RGB")

            numpy_image = np.array(ori_image)
            restored_array = restorer.restore(numpy_image)
            restored_image = Image.fromarray(restored_array)
            result_image = Image.blend(ori_image, restored_image, restorer_visibility)

            buffer = BytesIO()
            result_image.save(buffer, format="JPEG")
            ret_img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"人脸增强失败: {exc}") from exc

    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(
        content=jsonable_encoder(content), media_type="application/json"
    )
