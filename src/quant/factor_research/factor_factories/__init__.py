"""因子工厂包的公共入口。

导入本包时，registry 会自动发现各因子模块并实例化注册；调用方可通过
``get_factor_factory(name)`` 按名称取得对应工厂并执行因子计算。
"""

from .registry import FACTOR_FACTORIES, get_factor_factory, register_factor

__all__ = ["FACTOR_FACTORIES", "get_factor_factory", "register_factor"]
