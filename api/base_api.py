# 创建一个类继承APIRouter，并且在 init 的时候调用一些初始化加载模型的逻辑

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware
from abc import ABC, abstractmethod
from functools import wraps
import sys

sdk_abs_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class InitMiddleware(BaseHTTPMiddleware):
    """在懒加载模式下保障路由只初始化一次，并对并发请求加锁。"""

    def __init__(self, app, routers):
        super().__init__(app)
        self.routers = routers  # 这是一个列表，包含所有需要初始化的 router 实例
        # 初始化锁映射，避免多个请求同时触发重复初始化
        self.init_locks = {id(router): asyncio.Lock() for router in routers}

    async def dispatch(self, request: Request, call_next):
        # 检查每个 router 是否已初始化
        for router in self.routers:
            if not router.initialized:
                # 获取对应 router 的锁
                lock = self.init_locks[id(router)]
                async with lock:
                    # 双重检查是否已初始化
                    if not router.initialized:
                        try:
                            await router.ensure_initialized()  # 执行初始化
                        except Exception as e:
                            router.logger.exception(
                                "Initialization failed inside middleware"
                            )
                            # 如果初始化失败，返回错误响应
                            return JSONResponse(
                                status_code=500,
                                content={
                                    "message": f"Initialization failed for {router.app_name}: {str(e)}"
                                },
                            )
        # 所有 router 都已初始化，处理请求
        response = await call_next(request)
        return response


class BaseAPIRouter(APIRouter, ABC):
    def __init__(self, app_name: str):
        super().__init__()
        self.app_name = app_name
        self.dir = f"{sdk_abs_path}/repo/{app_name}"
        self.models = {}
        self.initialized = False
        self._init_lock = asyncio.Lock()
        self.logger = logging.getLogger(f"aigc.router.{app_name}")

    @abstractmethod
    async def init_app(self):
        """子类实现具体的模型加载或资源准备逻辑。"""
        pass

    async def destroy_app(self):
        """Override if extra cleanup is required."""
        self.models.clear()

    async def ensure_initialized(self):
        """并发安全地执行初始化过程，确保只运行一次。"""
        if self.initialized:
            return
        async with self._init_lock:
            if self.initialized:
                return
            self.logger.info("Initializing application router")
            await self.init_app()
            self.initialized = True
            self.logger.info("Application router initialized")

    async def shutdown(self):
        """释放路由占用的资源，并重置状态，便于后续再次加载。"""
        if not self.initialized:
            return
        self.logger.info("Shutting down application router")
        try:
            await self.destroy_app()
        finally:
            self.models.clear()
            self.initialized = False
            self.logger.info("Application router shut down")

    def register_model(self, name: str, resource):
        """登记资源句柄，供业务处理函数在运行期获取。"""
        self.models[name] = resource

    def require_model(self, name: str):
        """获取已登记的资源，若未初始化或缺失则抛出 HTTP 异常。"""
        if not self.initialized:
            raise HTTPException(
                status_code=503,
                detail=f"Router '{self.app_name}' is not initialized yet",
            )
        try:
            return self.models[name]
        except KeyError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Resource '{name}' is unavailable in router '{self.app_name}'",
            ) from exc


def change_dir(new_dir):
    """装饰器：在协程执行期间切换到指定目录，结束后切回原目录。"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ori_dir = os.getcwd()
            os.chdir(new_dir)
            try:
                result = await func(*args, **kwargs)
            finally:
                os.chdir(ori_dir)
            return result

        return wrapper

    return decorator


def init_helper(new_dir):
    """装饰器：临时将子模块路径加入 sys.path 并切换目录执行初始化。"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            ori_dir = os.getcwd()
            pkg_path = os.path.join(ori_dir, new_dir)
            path_added = False
            if pkg_path not in sys.path:
                sys.path.append(pkg_path)
                path_added = True
            os.chdir(new_dir)
            try:
                result = await func(*args, **kwargs)
            finally:
                os.chdir(ori_dir)
                if path_added and pkg_path in sys.path:
                    sys.path.remove(pkg_path)
            return result

        return wrapper

    return decorator
