"""判断因子工厂对行情频率的要求，以及它对价格尺度的齐次度。

**为什么用 ``getattr`` 而不是在基类上加属性**：``FactorCache`` 的实现指纹会哈希
``factor_factories/base.py`` 的文件字节，往基类里加任何一行都会作废全部约三十个
因子的历史缓存。因此 ``requires_intraday`` 与 ``price_homogeneity`` 都只声明在
真正需要它的那几个具体因子模块里，这里统一用 ``getattr`` 读取，未声明者分别按
``False`` 和 ``0`` 处理。

关于 ``price_homogeneity``：把窗口内全部价格乘以同一个正常数 ``c`` 后，齐次度为
``k`` 的因子满足 ``g(c·P) = c**k · g(P)``。

这里的扰动是**逐证券的常数**乘子，对应「更换前复权基准日（anchor）」这一类变化，
因为 ``qfq(s | anchor) = hfq(s) / hfq(anchor)``，分母对该证券是常数。它**不**等价于
「不复权换成后复权」：``hfq(code, t)`` 在时间上是分段常数，窗口跨过除权日时会跳一
档，所以即使 ``k == 0``，`--adjust` 选 ``none`` 还是 ``hfq`` 算出来的因子也不相同
（且只有复权后的那个才对）。

- ``k == 0``：因子只由价格之间的比值决定，换任何复权基准日取值都不变，可以安全
  进入横截面比较。绝大多数因子都属于这一类，因此取它作为缺省值。
- ``k != 0``：因子带绝对价格量纲。后复权系数是**逐证券**的，于是这类因子在横截面
  上会主要由「上市时长 × 分红历史」决定，而不是由信号决定，必须显式声明。
- ``None``：因子不满足任何齐次度，通常是在同一个表达式里混用了一次齐次的价格和
  零次齐次的收益率类中间量。这类因子没有解析换算式，只能个案判断它在实际取值
  范围内是否稳定。
"""

from __future__ import annotations

from .base import FactorFactory
from .registry import FACTOR_FACTORIES

#: 类属性名；具体因子模块通过声明它来标记自己依赖日内行情。
REQUIRES_INTRADAY_ATTRIBUTE = "requires_intraday"

#: 类属性名；具体因子模块通过声明它来标记自身对价格尺度的齐次度。
PRICE_HOMOGENEITY_ATTRIBUTE = "price_homogeneity"


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


def price_homogeneity(factory: FactorFactory | type[FactorFactory]) -> int | None:
    """返回一个因子工厂对价格的齐次度。

    参数：
        factory: 因子工厂实例或类。

    返回：
        工厂声明的整数齐次度 ``k``，含义是把窗口内全部价格乘以正常数 ``c`` 后
        ``g(c·P) = c**k · g(P)``；未声明按 ``0``（尺度无关）处理。声明为
        ``None`` 表示该因子不满足任何齐次度。

    异常：
        TypeError: 声明值既不是 ``None`` 也不是整数（``bool`` 也视为非法，
            避免把 ``True`` 误当成 ``k = 1``）。
    """
    value = getattr(factory, PRICE_HOMOGENEITY_ATTRIBUTE, 0)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        name = getattr(factory, "name", factory)
        raise TypeError(
            f"因子 {name!r} 的 {PRICE_HOMOGENEITY_ATTRIBUTE} 必须是整数或 None，"
            f"当前为 {value!r}"
        )
    return value


def dimensional_factor_names() -> frozenset[str]:
    """返回全部带绝对价格量纲的已注册因子名，即齐次度为非零整数的那些。

    后复权系数逐证券不同，因此这些因子在横截面上会混入「上市时长 × 分红历史」
    这一与信号无关的成分；把它们直接投进横截面排名或分组前必须先处理量纲。

    返回：
        因子名的不可变集合；不含声明为 ``None`` 的非齐次因子，后者见
        :func:`non_homogeneous_factor_names`。
    """
    return frozenset(
        name
        for name, factory in FACTOR_FACTORIES.items()
        if (degree := price_homogeneity(factory)) is not None and degree != 0
    )


def non_homogeneous_factor_names() -> frozenset[str]:
    """返回全部声明为不满足任何齐次度的已注册因子名。

    与 :func:`dimensional_factor_names` 的区别在于：那一类有确定的量纲、可以按
    ``c**k`` 解析换算；这一类没有解析形式，只能逐个评估它在实际取值范围内是否
    稳定，因此不能一概认为必须先处理量纲才能进入横截面。

    返回：
        声明 ``price_homogeneity = None`` 的因子名不可变集合。
    """
    return frozenset(
        name
        for name, factory in FACTOR_FACTORIES.items()
        if price_homogeneity(factory) is None
    )
