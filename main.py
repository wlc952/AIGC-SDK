import argparse
from argparse import BooleanOptionalAction
import importlib
import logging
import os
import pkgutil
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Sequence, Set, Tuple

from fastapi import FastAPI

import api
from api.base_api import BaseAPIRouter, InitMiddleware


DEFAULT_DESCRIPTION = "Airbox 提供的多模态生成服务集合"
LOGGER_NAME = "aigc"
INIT_OPENAPI_TAG = {"name": "Init", "description": "应用初始化相关操作"}


def configure_logging(log_level: Optional[str]) -> None:
    """根据传入的日志级别配置根 logger 及常用子 logger。"""
    level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        )
    else:
        root_logger.setLevel(level)
    logging.getLogger(LOGGER_NAME).setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)


def _port_type(value: str) -> int:
    """命令行端口参数解析器，确保端口处于合法范围。"""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be an integer") from exc
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def discover_available_modules() -> List[str]:
    """扫描 api 包下的所有路由模块名称，排除基础模块。"""
    modules: List[str] = []
    for module_info in pkgutil.iter_modules(api.__path__):
        if module_info.ispkg:
            continue
        if module_info.name == "base_api":
            continue
        modules.append(module_info.name)
    modules.sort()
    return modules


def resolve_module_names(requested: Optional[Sequence[str]] = None) -> List[str]:
    """解析待加载模块列表；若未指定则回退到自动发现。"""
    if requested:
        seen: Set[str] = set()
        ordered: List[str] = []
        for name in requested:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    discovered = discover_available_modules()
    if not discovered:
        raise RuntimeError("No router modules discovered under the 'api' package.")
    return discovered


def load_router_modules(
    module_names: Sequence[str], *, strict: bool
) -> Tuple[List[BaseAPIRouter], List[str], Dict[str, str]]:
    """按名称导入路由模块，返回成功实例与跳过列表及错误信息。"""
    routers: List[BaseAPIRouter] = []
    skipped: List[str] = []
    load_errors: Dict[str, str] = {}
    logger = logging.getLogger(LOGGER_NAME)
    for module_name in module_names:
        try:
            module = importlib.import_module(f"api.{module_name}")
        except ModuleNotFoundError as exc:
            if strict:
                raise ModuleNotFoundError(
                    f"No router module named '{module_name}' found under the api package"
                ) from exc
            logger.warning(
                "Skipping router module '%s' due to import error: %s", module_name, exc
            )
            skipped.append(module_name)
            load_errors[module_name] = str(exc)
            continue

        router = getattr(module, "router", None)
        if not isinstance(router, BaseAPIRouter):
            message = f"Module 'api.{module_name}' does not expose a BaseAPIRouter instance named 'router'"
            if strict:
                raise TypeError(message)
            logger.warning("Skipping router module '%s': %s", module_name, message)
            skipped.append(module_name)
            load_errors[module_name] = message
            continue
        routers.append(router)
    return routers, skipped, load_errors


def build_openapi_tags(routers: Sequence[BaseAPIRouter]) -> List[Dict[str, str]]:
    """构建用于 OpenAPI 文档的标签列表。"""
    tags: List[Dict[str, str]] = [INIT_OPENAPI_TAG]
    seen: Set[str] = set()
    for router in routers:
        if router.app_name in seen:
            continue
        tags.append(
            {
                "name": router.app_name,
                "description": f"{router.app_name} API",
            }
        )
        seen.add(router.app_name)
    return tags


async def initialize_routers(routers: Sequence[BaseAPIRouter]) -> None:
    """依次初始化所有 router，失败时抛出错误以阻止启动。"""
    for router in routers:
        try:
            await router.ensure_initialized()
        except Exception:
            router.logger.exception(
                "Router initialization failed during application startup"
            )
            raise


async def shutdown_routers(routers: Sequence[BaseAPIRouter]) -> None:
    """按逆序调用 router 的关停逻辑，避免资源泄露。"""
    for router in reversed(list(routers)):
        try:
            await router.shutdown()
        except Exception:
            router.logger.exception("Router shutdown encountered an error")


def create_app(
    routers: Sequence[BaseAPIRouter],
    *,
    lazy_init: bool,
    load_errors: Optional[Dict[str, str]] = None,
) -> FastAPI:
    # lazy_init 为 True 时表示延迟加载，只有在收到首次请求时才初始化模型资源；
    # 为 False 则在应用启动阶段即完成所有路由的初始化，适合对就绪时间有要求的场景。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if not lazy_init:
                await initialize_routers(routers)
            yield
        finally:
            await shutdown_routers(routers)

    app = FastAPI(
        title="Airbox AIGC Hub",
        description=DEFAULT_DESCRIPTION,
        version=os.getenv("AIGC_VERSION", "1.0.0"),
        openapi_tags=build_openapi_tags(routers),
        lifespan=lifespan,
    )

    app.state.router_registry = {router.app_name: router for router in routers}
    app.state.load_errors = load_errors or {}

    if lazy_init:
        app.add_middleware(InitMiddleware, routers=list(routers))

    for router in routers:
        app.include_router(router, tags=[router.app_name])

    @app.get("/", tags=["Init"])
    async def read_root():
        settings = getattr(app.state, "settings", {})
        return {
            "message": "Welcome to Airbox AIGC Hub",
            "modules": list(app.state.router_registry.keys()),
            "skipped": settings.get("skipped_modules", []),
            "load_errors": settings.get("load_errors", {}),
        }

    return app


def build_app(
    module_names: Optional[Sequence[str]] = None,
    *,
    lazy_init: Optional[bool] = None,
    log_level: Optional[str] = None,
    strict_imports: Optional[bool] = None,
) -> FastAPI:
    """创建并配置 FastAPI 应用，按参数控制模块加载与懒加载策略。"""
    configure_logging(log_level)

    resolved_names = resolve_module_names(module_names)
    requested_explicitly = module_names is not None
    strict = (
        strict_imports if strict_imports is not None else bool(requested_explicitly)
    )

    routers, skipped_modules, load_errors = load_router_modules(
        resolved_names, strict=strict
    )

    if not routers and strict:
        raise RuntimeError(
            "No router modules could be initialized. Check module names or dependencies."
        )

    if skipped_modules:
        logging.getLogger(LOGGER_NAME).warning(
            "Skipped router modules due to import errors: %s",
            ", ".join(skipped_modules),
        )

    lazy = lazy_init if lazy_init is not None else False

    app = create_app(routers, lazy_init=lazy, load_errors=load_errors)
    app.state.settings = {
        "requested_modules": list(resolved_names),
        "active_modules": [router.app_name for router in routers],
        "skipped_modules": skipped_modules,
        "lazy_init": lazy,
        "log_level": (log_level or "INFO").upper(),
        "strict_imports": strict,
        "load_errors": load_errors,
    }
    return app


def parse_args(argv: Optional[Sequence[str]] = None):
    """解析命令行参数，支持 host/port 等常用运行选项。"""
    parser = argparse.ArgumentParser(description="Run the Airbox AIGC Hub API server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind",
    )
    parser.add_argument(
        "--port",
        type=_port_type,
        default=8000,
        help="Port to expose the HTTP server",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--lazy-init",
        action=BooleanOptionalAction,
        default=False,
        help="Delay model initialization until the first request",
    )
    parser.add_argument(
        "module_names",
        nargs="*",
        help="Optional explicit list of router module names to load",
    )
    return parser.parse_args(argv)


if __name__ != "__main__":
    app = build_app(strict_imports=False)
else:
    app = None


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口：构建应用并交给 uvicorn 运行。"""
    args = parse_args(argv)
    runtime_app = build_app(
        module_names=args.module_names or None,
        lazy_init=args.lazy_init,
        log_level=args.log_level,
        strict_imports=True,
    )

    import uvicorn

    uvicorn.run(
        runtime_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
