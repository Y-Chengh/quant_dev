"""数据源的注册、自动发现和命令行构建入口。

结构与 ``factor_research.models.registry`` 完全对齐：主程序不为任何具体数据源
写分支，命令行先解析 ``--data-source``、再只注册所选数据源的专属参数。
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
from typing import TypeVar

from .base import MarketDataSource

SourceType = TypeVar("SourceType", bound=type[MarketDataSource])
DATA_SOURCE_TYPES: dict[str, type[MarketDataSource]] = {}
DEFAULT_DATA_SOURCE = "market_service"


def register_data_source(source_type: SourceType) -> SourceType:
    """以唯一名称注册数据源类，同时保留被装饰的类。

    参数：
        source_type: 待注册的 ``MarketDataSource`` 子类；必须定义非空 ``name``。

    返回：
        原样返回被装饰的类，便于继续使用。
    """
    if not isinstance(source_type, type) or not issubclass(source_type, MarketDataSource):
        raise TypeError("register_data_source 只能装饰 MarketDataSource 子类")
    name = getattr(source_type, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"数据源 {source_type.__name__} 必须定义非空 name")
    if name in DATA_SOURCE_TYPES:
        existing = DATA_SOURCE_TYPES[name]
        raise ValueError(
            f"数据源名称 {name!r} 重复: "
            f"{existing.__module__}.{existing.__name__} 与 "
            f"{source_type.__module__}.{source_type.__name__}"
        )
    DATA_SOURCE_TYPES[name] = source_type
    return source_type


def _discover_data_sources() -> None:
    """自动导入本包下的具体数据源模块，触发注册。

    返回：
        无返回值；``base`` 与 ``registry`` 自身会被跳过。
    """
    package_name = __package__
    package = importlib.import_module(package_name)
    ignored = {"base", "registry"}
    modules = sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name)
    for module in modules:
        if not module.ispkg and module.name not in ignored:
            importlib.import_module(f"{package_name}.{module.name}")


def available_data_sources() -> list[str]:
    """惰性发现数据源模块，并返回当前已注册的数据源名称。

    返回：
        按字典序排列的数据源名列表。
    """
    _discover_data_sources()
    return sorted(DATA_SOURCE_TYPES)


def add_data_source_selection_argument(parser: argparse.ArgumentParser) -> None:
    """只注册通用的数据源选择参数，供两阶段解析共同使用。

    参数：
        parser: 目标解析器。

    返回：
        无返回值。
    """
    parser.add_argument(
        "--data-source",
        choices=available_data_sources(),
        default=DEFAULT_DATA_SOURCE,
        help=f"行情数据源；默认 {DEFAULT_DATA_SOURCE}",
    )


def add_selected_data_source_arguments(
    parser: argparse.ArgumentParser, source_name: str
) -> None:
    """只注册所选数据源的专属参数，避免不同数据源的参数名互相冲突。

    参数：
        parser: 目标解析器。
        source_name: 已选定的数据源名。

    返回：
        无返回值；数据源名未注册时抛出 ``ValueError``。
    """
    _discover_data_sources()
    try:
        source_type = DATA_SOURCE_TYPES[source_name]
    except KeyError as exc:
        raise ValueError(
            f"未知数据源 {source_name!r}；可选数据源: {available_data_sources()}"
        ) from exc
    source_type.add_arguments(parser)


def data_source_from_args(args: argparse.Namespace) -> MarketDataSource:
    """根据 ``args.data_source`` 选择并构建对应数据源。

    参数：
        args: 已解析的命令行参数；缺少 ``data_source`` 字段时使用默认数据源。

    返回：
        构建好的数据源实例。
    """
    _discover_data_sources()
    name = getattr(args, "data_source", DEFAULT_DATA_SOURCE)
    try:
        source_type = DATA_SOURCE_TYPES[name]
    except KeyError as exc:
        raise ValueError(
            f"未知数据源 {name!r}；可选数据源: {available_data_sources()}"
        ) from exc
    return source_type.from_args(args)


def data_source_argument_names(source_name: str) -> set[str]:
    """返回指定数据源声明的参数名，用于识别切换数据源后的失效配置。

    参数：
        source_name: 数据源名。

    返回：
        该数据源通过 ``add_arguments`` 注册的参数 ``dest`` 集合；
        数据源名未注册时返回空集合。
    """
    _discover_data_sources()
    source_type = DATA_SOURCE_TYPES.get(source_name)
    if source_type is None:
        return set()
    parser = argparse.ArgumentParser(add_help=False)
    source_type.add_arguments(parser)
    return {action.dest for action in parser._actions}
