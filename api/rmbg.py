import base64
import os
from typing import Tuple

import cv2
import numpy as np
from fastapi import File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from api.base_api import BaseAPIRouter, sdk_abs_path

from untool import EngineOV

app_name = "rmbg"

model_path = os.path.join(sdk_abs_path, "bmodels/rmbg/rmbg.bmodel")


class AppInitializationRouter(BaseAPIRouter):
    async def init_app(self):
        engine = EngineOV(model_path, device_id=0)
        self.register_model("rmbg", engine)
        return {
            "message": f"Application {self.app_name} has been initialized successfully."
        }


router = AppInitializationRouter(app_name=app_name)


### 图像去背景；兼容openai api，images/edit
@router.post("/v1/images/edit")
async def remove_background(image: UploadFile = File(...)):
    engine = router.require_model("rmbg")

    ori_image_bytes = await image.read()
    if not ori_image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像数据为空")

    nparr = np.frombuffer(ori_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析上传的图像")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_im_size: Tuple[int, int] = img_rgb.shape[:2]
    model_input_size = (1024, 1024)

    image_processed = preprocess_image(img_rgb, model_input_size)

    try:
        result = engine([image_processed])[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"模型推理失败: {exc}") from exc

    result = result.squeeze(0)
    result_image = postprocess_image(result, orig_im_size)

    mask = result_image
    b_channel, g_channel, r_channel = cv2.split(img)
    img_bgra = cv2.merge((b_channel, g_channel, r_channel, mask))

    success, buffer = cv2.imencode(".png", img_bgra)
    if not success:
        raise HTTPException(status_code=500, detail="图像编码失败")

    ret_img_b64 = base64.b64encode(buffer).decode("utf-8")
    content = {"data": [{"b64_json": ret_img_b64}]}

    return JSONResponse(
        content=jsonable_encoder(content), media_type="application/json"
    )


def preprocess_image(im: np.ndarray, model_input_size: Tuple[int, int]) -> np.ndarray:
    # 转换通道顺序 HWC -> CHW
    im_chw = np.transpose(im, (2, 0, 1)).astype(np.float32)

    # 使用OpenCV进行缩放
    c, h, w = im_chw.shape
    im_resized = np.zeros((c, model_input_size[0], model_input_size[1]), dtype=np.uint8)
    for i in range(c):
        im_resized[i] = cv2.resize(
            im_chw[i],
            (model_input_size[1], model_input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # 归一化
    image = im_resized.astype(np.float32) / 255.0

    # 标准化
    for i in range(c):
        image[i] = (image[i] - 0.5) / 1.0

    return image


def postprocess_image(result: np.ndarray, im_size: Tuple[int, int]) -> np.ndarray:
    # 使用OpenCV进行缩放
    c, h, w = result.shape
    result_resized = np.zeros((c, im_size[0], im_size[1]), dtype=np.float32)
    for i in range(c):
        result_resized[i] = cv2.resize(
            result[i], (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR
        )

    # 归一化处理
    ma = np.max(result_resized)
    mi = np.min(result_resized)
    denominator = ma - mi
    if denominator == 0:
        result_norm = np.zeros_like(result_resized)
    else:
        result_norm = (result_resized - mi) / denominator

    # 调整格式并转换到0-255的值域
    im_array = (result_norm * 255).transpose(1, 2, 0).astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array
