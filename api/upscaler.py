import base64
import math
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image

from api.base_api import BaseAPIRouter, sdk_abs_path
from untool import EngineOV

app_name = "upscaler"


class AppInitializationRouter(BaseAPIRouter):
    async def init_app(self):
        model = UpscaleModel(
            model=f"{sdk_abs_path}/bmodels/upscaler/resrgan4x.bmodel", padding=20
        )
        self.register_model("upscaler", model)
        return {
            "message": f"Application {self.app_name} has been initialized successfully."
        }


router = AppInitializationRouter(app_name=app_name)


### 图像超分；兼容openai api，image/edit
@router.post("/v1/images/edit")
async def upscale(
    image: UploadFile = File(...),
    upscale_ratio: Optional[float] = Form(1.0),
):
    model = router.require_model("upscaler")

    if upscale_ratio is None:
        upscale_ratio = 1.0
    try:
        upscale_ratio = float(upscale_ratio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="upscale_ratio 必须为数字") from exc

    if upscale_ratio <= 0:
        raise HTTPException(status_code=400, detail="upscale_ratio 必须大于 0")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="上传的图像为空")

    src_image = None
    pil_res = None
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            src_image = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()

        pil_res = model.extract_and_enhance_tiles(src_image, upscale_ratio=upscale_ratio)

        buffer = BytesIO()
        pil_res.save(buffer, format="JPEG")
        ret_img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"超分失败: {exc}") from exc
    finally:
        await image.close()
        if src_image is not None:
            src_image.close()
        if pil_res is not None and hasattr(pil_res, "close"):
            pil_res.close()

    content = {"data": [{"b64_json": ret_img_b64}]}
    return JSONResponse(
        content=jsonable_encoder(content), media_type="application/json"
    )


def calWeight(d, k):
    """
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    """
    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion2(img1, img2, overlap, left_right=True):
    """
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    """
    # 这里先暂时考虑平行向融合
    wei = calWeight(overlap, 0.05)  # k=5 这里是超参
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    if left_right:  # 左右融合
        assert h1 == h2 and c1 == c2
        img_new = np.zeros((h1, w1 + w2 - overlap, c1))
        img_new[:, :w1, :] = img1
        wei_expand = np.tile(wei, (h1, 1))  # 权重扩增
        wei_expand = np.expand_dims(wei_expand, 2).repeat(3, axis=2)
        img_new[:, w1 - overlap : w1, :] = (1 - wei_expand) * img1[
            :, w1 - overlap : w1, :
        ] + wei_expand * img2[:, :overlap, :]
        img_new[:, w1:, :] = img2[:, overlap:, :]
    else:  # 上下融合
        assert w1 == w2 and c1 == c2
        img_new = np.zeros((h1 + h2 - overlap, w1, c1))
        img_new[:h1, :, :] = img1
        wei = np.reshape(wei, (overlap, 1))
        wei_expand = np.tile(wei, (1, w1))
        wei_expand = np.expand_dims(wei_expand, 2).repeat(3, axis=2)
        img_new[h1 - overlap : h1, :, :] = (1 - wei_expand) * img1[
            h1 - overlap : h1, :, :
        ] + wei_expand * img2[:overlap, :, :]
        img_new[h1:, :, :] = img2[overlap:, :, :]
    return img_new


def imgFusion(img_list, overlap, res_w, res_h):
    print(res_w, res_h)
    pre_v_img = None
    for vi in range(len(img_list)):
        h_img = np.transpose(img_list[vi][0], (1, 2, 0))
        for hi in range(1, len(img_list[vi])):
            new_img = np.transpose(img_list[vi][hi], (1, 2, 0))
            h_img = imgFusion2(
                h_img,
                new_img,
                (h_img.shape[1] + new_img.shape[1] - res_w)
                if (hi == len(img_list[vi]) - 1)
                else overlap,
                True,
            )
        pre_v_img = (
            h_img
            if pre_v_img is None
            else imgFusion2(
                pre_v_img,
                h_img,
                (pre_v_img.shape[0] + h_img.shape[0] - res_h)
                if vi == len(img_list) - 1
                else overlap,
                False,
            )
        )
    return np.transpose(pre_v_img, (2, 0, 1))


class UpscaleModel:
    def __init__(
        self,
        tile_size=(196, 196),
        padding=4,
        upscale_rate=2,
        model=None,
        model_size=(200, 200),
        device_id=0,
    ):
        self.tile_size = tile_size
        self.padding = padding
        self.upscale_rate = upscale_rate
        if model is None:
            print("use default upscaler model")
            model = "./models/other/resrgan4x.bmodel"
        self.model = EngineOV(model, device_id=device_id)
        self.model_size = model_size

    def calc_tile_position(self, width, height, col, row):
        # generate mask
        tile_left = col * self.tile_size[0]
        tile_top = row * self.tile_size[1]
        tile_right = (col + 1) * self.tile_size[0] + self.padding
        tile_bottom = (row + 1) * self.tile_size[1] + self.padding
        if tile_right > height:
            tile_right = height
            tile_left = height - self.tile_size[0] - self.padding * 1
        if tile_bottom > width:
            tile_bottom = width
            tile_top = width - self.tile_size[1] - self.padding * 1

        return tile_top, tile_left, tile_bottom, tile_right

    def calc_upscale_tile_position(self, tile_left, tile_top, tile_right, tile_bottom):
        return (
            int(tile_left * self.upscale_rate),
            int(tile_top * self.upscale_rate),
            int(tile_right * self.upscale_rate),
            int(tile_bottom * self.upscale_rate),
        )

    def modelprocess(self, tile):
        ntile = tile.resize(self.model_size)
        # preprocess
        ntile = np.array(ntile).astype(np.float32)
        ntile = ntile / 255
        ntile = np.transpose(ntile, (2, 0, 1))
        ntile = ntile[np.newaxis, :, :, :]

        res = self.model([ntile])[0]
        # extract padding
        res = res[0]
        res = np.transpose(res, (1, 2, 0))
        res = res * 255
        res[res > 255] = 255
        res[res < 0] = 0
        res = res.astype(np.uint8)
        res = Image.fromarray(res)
        res = res.resize(self.target_tile_size)
        return res

    def extract_and_enhance_tiles(self, image, upscale_ratio=2.0):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # 获取图像的宽度和高度
        width, height = image.size
        self.upscale_rate = upscale_ratio
        self.target_tile_size = (
            int((self.tile_size[0] + self.padding * 1) * upscale_ratio),
            int((self.tile_size[1] + self.padding * 1) * upscale_ratio),
        )
        target_width, target_height = (
            int(width * upscale_ratio),
            int(height * upscale_ratio),
        )
        # 计算瓦片的列数和行数
        num_cols = math.ceil((width - self.padding) / self.tile_size[0])
        num_rows = math.ceil((height - self.padding) / self.tile_size[1])

        # 遍历每个瓦片的行和列索引
        img_tiles = []
        for row in range(num_rows):
            img_h_tiles = []
            for col in range(num_cols):
                # 计算瓦片的左上角和右下角坐标
                tile_left, tile_top, tile_right, tile_bottom = self.calc_tile_position(
                    width, height, row, col
                )
                # 裁剪瓦片
                tile = image.crop((tile_left, tile_top, tile_right, tile_bottom))
                # 使用超分辨率模型放大瓦片
                upscaled_tile = self.modelprocess(tile)
                # 将放大后的瓦片粘贴到输出图像上
                # overlap
                ntile = np.array(upscaled_tile).astype(np.float32)
                ntile = np.transpose(ntile, (2, 0, 1))
                img_h_tiles.append(ntile)

            img_tiles.append(img_h_tiles)
        res = imgFusion(
            img_list=img_tiles,
            overlap=int(self.padding * upscale_ratio),
            res_w=target_width,
            res_h=target_height,
        )
        res = Image.fromarray(np.transpose(res, (1, 2, 0)).astype(np.uint8))
        return res
