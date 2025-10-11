import base64
import io
import os
import random
import time
from typing import Dict, List, Optional

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from api.base_api import BaseAPIRouter, change_dir, init_helper, sdk_abs_path

app_name = "sd_lcm"


class AppInitializationRouter(BaseAPIRouter):
    dir = f"repo/{app_name}"

    @init_helper(dir)
    async def init_app(self):
        from sd import StableDiffusionPipeline
        from sd.scheduler import samplers_k_diffusion
        from utils.tools import (
            create_size,
            ratio_resize,
            seed_torch,
            get_model_input_info,
            get_model_path,
        )

        # 初始化模型路径和配置
        self.model_path = get_model_path()
        basenames = list(self.model_path.keys())

        # 获取 ControlNet 列表
        controlnet_dir = "./models/controlnet"
        if os.path.exists(controlnet_dir):
            controlnet_list = os.listdir(controlnet_dir)
            if len(controlnet_list) != 0:
                controlnets = [i.split(".")[0] for i in controlnet_list]
            else:
                controlnets = []
        else:
            controlnets = []

        # 初始化调度器
        scheduler_options = ["LCM", "DDIM", "DPM Solver++"]
        for i in samplers_k_diffusion:
            scheduler_options.append(i[0])

        bad_scheduler = ["DPM Solver++", "DPM fast", "DPM adaptive"]
        for i in bad_scheduler:
            if i in scheduler_options:
                scheduler_options.remove(i)

        # 创建尺寸选项
        size_options = create_size(512, 768)

        # 加载默认模型
        if not basenames:
            raise Exception("No models available")

        # 选择第一个可用的模型作为默认模型
        default_model = basenames[0]

        # 检查模型文件
        if not self._pre_check_model(default_model, check_type=["te", "unet", "vae"]):
            raise Exception(f"Model files for {default_model} are missing")

        # 初始化并加载模型
        print(f"Loading model: {default_model}")
        controlnet_name = controlnets[0] if controlnets else None
        pipe_kwargs = {
            "basic_model": default_model,
            "scheduler": scheduler_options[0],
        }
        if controlnet_name:
            pipe_kwargs["controlnet_name"] = controlnet_name
        pipe = StableDiffusionPipeline(**pipe_kwargs)
        current_model_input_shapes = get_model_input_info(
            pipe.unet.basic_info["stage_info"]
        )

        print(f"Model {default_model} loaded successfully")
        print(f"Supported sizes: {current_model_input_shapes}")

        state: Dict[str, object] = {
            "pipe": pipe,
            "basenames": basenames,
            "scheduler": scheduler_options,
            "controlnets": controlnets,
            "current_model_name": default_model,
            "current_scheduler": scheduler_options[0],
            "controlnet": controlnet_name,
            "current_model_input_shapes": current_model_input_shapes,
            "size_options": size_options,
        }
        self.register_model("sd_lcm", state)

        return {
            "message": f"应用 {self.app_name} 已成功初始化，模型 {default_model} 已加载。"
        }

    def _pre_check_model(self, model_select, check_type=None):
        """内部方法：检查模型文件是否存在"""
        check_pass = True
        model_select_path = os.path.join("models", "basic", model_select)
        te_path = os.path.join(
            model_select_path, self.model_path[model_select]["encoder"]
        )
        unet_path = os.path.join(
            model_select_path, self.model_path[model_select]["unet"]
        )
        vae_de_path = os.path.join(
            model_select_path, self.model_path[model_select]["vae_decoder"]
        )
        vae_en_path = os.path.join(
            model_select_path, self.model_path[model_select]["vae_encoder"]
        )

        if "te" in check_type:
            if not os.path.isfile(te_path):
                check_pass = False
        if "unet" in check_type:
            if not os.path.isfile(unet_path):
                check_pass = False
        if "vae" in check_type:
            if not os.path.exists(vae_en_path) or not os.path.exists(vae_de_path):
                check_pass = False

        return check_pass


router = AppInitializationRouter(app_name=app_name)


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def base64_to_image(base64_str: str) -> Image.Image:
    """将base64字符串转换为PIL图像"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


@router.post("/v1/images/generations")
@change_dir(router.dir)
async def text_to_image(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form("ugly, poor details, bad anatomy"),
    scheduler: Optional[str] = Form("LCM"),
    steps: Optional[int] = Form(20),
    guidance_scale: Optional[float] = Form(0.2),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(768),
    seed: Optional[int] = Form(1),
    enable_prompt_weight: Optional[bool] = Form(False),
):
    """文生图接口"""
    try:
        state = router.require_model("sd_lcm")
        pipe = state["pipe"]
        model_input_shapes = state["current_model_input_shapes"]
        scheduler_options = state["scheduler"]

        # 检查尺寸支持
        target_size = [width, height]
        if target_size not in model_input_shapes:
            raise HTTPException(
                status_code=400,
                detail=f"Size {width}x{height} not supported by current model",
            )

        if scheduler not in scheduler_options:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported scheduler: {scheduler}. 可用选项: {scheduler_options}",
            )

        # 设置随机种子
        if seed is None:
            seed = random.randint(0, 1000000)

        # 调整LCM调度器的步数
        if scheduler == "LCM" and steps > 10:
            steps = 4
        elif scheduler != "LCM" and steps < 15:
            steps = 20

        # 设置尺寸
        pipe.set_height_width(height, width)

        # 生成图像
        img_pil = pipe(
            init_image=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=0.5,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            enable_prompt_weight=enable_prompt_weight,
            seeds=[seed],
        )
        state["current_scheduler"] = scheduler

        # 返回结果
        img_base64 = image_to_base64(img_pil)
        return JSONResponse(
            content={"created": int(time.time()), "data": [{"b64_json": img_base64}]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/edits")
@change_dir(router.dir)
async def image_to_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form("ugly, poor details, bad anatomy"),
    scheduler: Optional[str] = Form("LCM"),
    steps: Optional[int] = Form(4),
    strength: Optional[float] = Form(0.5),
    guidance_scale: Optional[float] = Form(0.0),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(768),
    seed: Optional[int] = Form(None),
    enable_prompt_weight: Optional[bool] = Form(False),
):
    """图生图接口"""
    try:
        state = router.require_model("sd_lcm")
        pipe = state["pipe"]
        model_input_shapes = state["current_model_input_shapes"]
        scheduler_options = state["scheduler"]

        # 加载输入图像
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="上传的图像为空")

        target_size = [width, height]
        from utils.tools import ratio_resize

        with Image.open(io.BytesIO(image_bytes)) as original_image:
            prepared_image = ratio_resize(original_image, target_size)
            input_image = prepared_image.copy() if prepared_image is original_image else prepared_image

        # 调整图像尺寸
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # 检查尺寸支持
        if target_size not in model_input_shapes:
            raise HTTPException(
                status_code=400,
                detail=f"Size {width}x{height} not supported by current model",
            )

        if scheduler not in scheduler_options:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported scheduler: {scheduler}. 可用选项: {scheduler_options}",
            )

        # 设置随机种子
        if seed is None:
            seed = random.randint(0, 1000000)

        # 调整LCM调度器的步数
        if scheduler == "LCM" and steps > 10:
            steps = 4
        elif scheduler != "LCM" and steps < 15:
            steps = 20

        # 设置尺寸
        pipe.set_height_width(height, width)

        # 生成图像
        img_pil = pipe(
            init_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=strength,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            enable_prompt_weight=enable_prompt_weight,
            seeds=[seed],
        )
        state["current_scheduler"] = scheduler

        # 返回结果
        img_base64 = image_to_base64(img_pil)
        return JSONResponse(
            content={"created": int(time.time()), "data": [{"b64_json": img_base64}]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "input_image" in locals() and hasattr(input_image, "close"):
            input_image.close()


@router.get("/v1/models")
@change_dir(router.dir)
async def list_models():
    """获取可用模型列表"""
    state = router.require_model("state")
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "sd_lcm",
                }
                for model in state["basenames"]
            ],
        }
    )


@router.get("/v1/current_model")
@change_dir(router.dir)
async def get_current_model():
    """获取当前加载的模型信息"""
    state = router.require_model("state")
    return JSONResponse(
        content={
            "current_model": state["current_model_name"],
            "current_controlnet": state["controlnet"],
            "supported_sizes": state["current_model_input_shapes"],
            "available_schedulers": state["scheduler"],
        }
    )


@router.get("/v1/schedulers")
@change_dir(router.dir)
async def list_schedulers():
    """获取可用调度器列表"""
    state = router.require_model("state")
    return JSONResponse(content={"object": "list", "data": state["scheduler"]})


@router.get("/v1/controlnets")
@change_dir(router.dir)
async def list_controlnets():
    """获取可用ControlNet列表"""
    state = router.require_model("state")
    return JSONResponse(content={"object": "list", "data": state["controlnets"]})


#### 测试命令
# 文生图：
# curl -X POST "http://localhost:8000/v1/images/generations" \
#   -F "prompt=1girl, ponytail, white hair, purple eyes, medium breasts, collarbone, flowers and petals, landscape, background, rose, abstract" \
#   -F "negative_prompt=ugly, poor details, bad anatomy" \
#   -F "scheduler=LCM" \
#   -F "steps=4" \
#   -F "width=512" \
#   -F "height=768"

# 图生图：
# curl -X POST "http://localhost:8000/v1/images/edits" \
#   -F "image=@/path/to/your/image.jpg" \
#   -F "prompt=upper body photo, fashion photography of cute Hatsune Miku" \
#   -F "negative_prompt=ugly, poor details, bad anatomy" \
#   -F "strength=0.5" \
#   -F "scheduler=DPM++ 2M SDE Karras" \
#   -F "steps=20"
