from __future__ import annotations

import importlib
import pkgutil
from typing import TypeVar

from .base import FactorFactory


FactoryType = TypeVar("FactoryType", bound=type[FactorFactory])
FACTOR_FACTORIES: dict[str, FactorFactory] = {}


def register_factor(factory_type: FactoryType) -> FactoryType:
    """Register a factor factory class while preserving the decorated class."""
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
    package_name = __package__
    package = importlib.import_module(package_name)
    ignored = {"base", "registry"}
    modules = sorted(pkgutil.iter_modules(package.__path__), key=lambda module: module.name)
    for module in modules:
        if not module.ispkg and module.name not in ignored:
            importlib.import_module(f"{package_name}.{module.name}")


def get_factor_factory(name: str) -> FactorFactory:
    try:
        return FACTOR_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"未知因子 {name!r}；可选因子: {sorted(FACTOR_FACTORIES)}") from exc


_discover_factories()
