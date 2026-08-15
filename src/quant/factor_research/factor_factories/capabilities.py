"""判断因子工厂对行情频率的要求。

**为什么用 ``getattr`` 而不是在基类上加属性**：``FactorCache`` 的实现指纹会哈希
``factor_factories/base.py`` 的文件字节，往基类里加任何一行都会作废全部约三十个
因子的历史缓存。因此 ``requires_intraday`` 只声明在真正需要日内行情的那几个具体
因子模块里，这里统一用 ``getattr`` 读取，未声明者按 ``False`` 处理。
"""

from __future__ import annotations

from .base import FactorFactory
from .registry import FACTOR_FACTORIES

#: 类属性名；具体因子模块通过声明它来标记自己依赖日内行情。
REQUIRES_INTRADAY_ATTRIBUTE = "requires_intraday"


def requires_intraday(factory: FactorFactory | type[FactorFactory]) -> bool:
    """判断一个因子工厂是否依赖日内（分钟）行情。

    参数：
        factory: 因子工厂实例或类。

    返回：
        工厂声明了 ``requires_intraday = True`` 时返回 ``True``；未声明按
        ``False`` 处理，即认为该因子只用日频数据就能算。
    """
    return bool(getattr(factory, REQUIRES_INTRADAY_ATTRIBUTE, False))


def intraday_factor_names() -> frozenset[str]:
    """返回全部依赖日内行情的已注册因子名。

    返回：
        因子名的不可变集合；日频数据源必须排除这些因子。
    """
    return frozenset(
        name for name, factory_type in FACTOR_FACTORIES.items() if requires_intraday(factory_type)
    )


def daily_capable_factor_names() -> frozenset[str]:
    """返回只用日频数据就能计算的已注册因子名。

    返回：
        因子名的不可变集合，即全部已注册因子减去依赖日内行情的那些。
    """
    return frozenset(FACTOR_FACTORIES) - intraday_factor_names()
