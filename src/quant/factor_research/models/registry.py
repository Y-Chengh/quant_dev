"""模型工厂的注册、自动发现和命令行构建入口。"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from typing import TypeVar

from .base import DirectionModelFactory

FactoryType = TypeVar("FactoryType", bound=type[DirectionModelFactory])
MODEL_FACTORY_TYPES: dict[str, type[DirectionModelFactory]] = {}
DEFAULT_MODEL = "simple_decision_tree"


def register_model_factory(factory_type: FactoryType) -> FactoryType:
    """以唯一名称注册模型工厂类，同时保留被装饰的类。"""
    if not isinstance(factory_type, type) or not issubclass(
        factory_type, DirectionModelFactory
    ):
        raise TypeError("register_model_factory 只能装饰 DirectionModelFactory 子类")
    name = getattr(factory_type, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"模型工厂 {factory_type.__name__} 必须定义非空 name")
    if name in MODEL_FACTORY_TYPES:
        existing = MODEL_FACTORY_TYPES[name]
        raise ValueError(
            f"模型名称 {name!r} 重复: "
            f"{existing.__module__}.{existing.__name__} 与 "
            f"{factory_type.__module__}.{factory_type.__name__}"
        )
    MODEL_FACTORY_TYPES[name] = factory_type
    return factory_type


def _discover_model_factories() -> None:
    """自动导入 models 包下的具体模型模块，触发工厂注册。"""
    package_name = __package__
    package = importlib.import_module(package_name)
    ignored = {"base", "registry"}
    modules = sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name)
    for module in modules:
        if not module.ispkg and module.name not in ignored:
            importlib.import_module(f"{package_name}.{module.name}")


def available_models() -> list[str]:
    """惰性发现模型模块，并返回当前已注册的模型名称。"""
    _discover_model_factories()
    return sorted(MODEL_FACTORY_TYPES)


def add_model_selection_argument(parser: argparse.ArgumentParser) -> None:
    """只注册通用的模型选择参数，供两阶段解析共同使用。"""
    parser.add_argument(
        "--model",
        choices=available_models(),
        default=DEFAULT_MODEL,
        help=f"方向预测模型；默认 {DEFAULT_MODEL}",
    )


def add_selected_model_arguments(
    parser: argparse.ArgumentParser, model_name: str
) -> None:
    """只注册所选模型的专属参数，避免不同模型的参数名互相冲突。"""
    _discover_model_factories()
    try:
        factory_type = MODEL_FACTORY_TYPES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"未知模型 {model_name!r}；可选模型: {available_models()}"
        ) from exc
    factory_type.add_arguments(parser)


def model_factory_from_args(args: argparse.Namespace) -> DirectionModelFactory:
    """根据 ``args.model`` 选择并构建对应模型工厂。"""
    _discover_model_factories()
    try:
        factory_type = MODEL_FACTORY_TYPES[args.model]
    except KeyError as exc:
        raise ValueError(
            f"未知模型 {args.model!r}；可选模型: {available_models()}"
        ) from exc
    factory = factory_type.from_args(args)
    task = getattr(args, "task", "classification")
    if task not in factory.supported_tasks:
        raise ValueError(
            f"模型 {factory.name!r} 不支持任务 {task!r}；"
            f"支持的任务为 {factory.supported_tasks}"
        )
    return factory
