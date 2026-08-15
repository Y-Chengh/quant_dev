"""因子工厂的注册表与自动发现机制。

具体因子类通过 ``@register_factor`` 以其 ``name`` 注册实例；模块加载结束时，
本文件会扫描同包下除 base、registry 外的模块，从而自动导入全部因子文件。
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TypeVar

from .base import FactorFactory


FactoryType = TypeVar("FactoryType", bound=type[FactorFactory])
FACTOR_FACTORIES: dict[str, FactorFactory] = {}


def register_factor(factory_type: FactoryType) -> FactoryType:
    """校验并注册因子工厂类，同时原样返回类以支持装饰器语法。"""
    if not isinstance(factory_type, type) or not issubclass(factory_type, FactorFactory):
        raise TypeError("register_factor 只能装饰 FactorFactory 子类")
    name = getattr(factory_type, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"因子工厂 {factory_type.__name__} 必须定义非空 name")
    if name in FACTOR_FACTORIES:
        existing = FACTOR_FACTORIES[name].__class__
        raise ValueError(
            f"因子名称 {name!r} 重复: {existing.__module__}.{existing.__name__} 与 "
            f"{factory_type.__module__}.{factory_type.__name__}"
        )
    FACTOR_FACTORIES[name] = factory_type()
    return factory_type


def _discover_factories() -> None:
    """按模块名顺序导入包内因子模块，触发各类的注册装饰器。"""
    package_name = __package__
    package = importlib.import_module(package_name)
    ignored = {"base", "registry"}
    modules = sorted(pkgutil.iter_modules(package.__path__), key=lambda module: module.name)
    for module in modules:
        if not module.ispkg and module.name not in ignored:
            importlib.import_module(f"{package_name}.{module.name}")


def get_factor_factory(name: str) -> FactorFactory:
    """按唯一因子名称返回已实例化的工厂，不存在时给出可选名称。"""
    try:
        return FACTOR_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"未知因子 {name!r}；可选因子: {sorted(FACTOR_FACTORIES)}") from exc


# 导入 registry 即完成因子发现，调用方不需要手动维护因子模块列表。
_discover_factories()
