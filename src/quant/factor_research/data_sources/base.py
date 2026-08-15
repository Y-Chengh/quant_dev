"""行情数据源的统一抽象。

数据源封装「从哪里取行情、取到的是什么频率、以及怎么把它变成日频因子表」。
主实验只依赖这个抽象，不为任何具体数据源写分支——与模型工厂的约定一致。
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

#: 分钟频数据源标识；可以计算依赖日内行情的因子。
FREQUENCY_INTRADAY = "5m"

#: 日频数据源标识；只能计算日频因子。
FREQUENCY_DAILY = "1d"


class MarketDataSource(ABC):
    """行情数据源的统一抽象：元数据、证券池、行情装载与日频特征构建。"""

    #: 数据源唯一名称，同时是 ``--data-source`` 的取值。
    name: str

    #: 行情频率，取值为 ``FREQUENCY_INTRADAY`` 或 ``FREQUENCY_DAILY``。
    frequency: str

    #: 是否提供日内行情；为 ``False`` 时依赖 ``intraday_values`` 的因子不可用。
    provides_intraday: bool

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """注册本数据源的专属命令行参数。

        参数：
            parser: 目标解析器。实现方只应注册自己独有的参数，不同数据源之间
                允许出现同名参数，因为主程序会先解析 ``--data-source``、
                再只注册所选数据源的参数。

        返回：
            无返回值；缺省实现不注册任何参数。
        """

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> MarketDataSource:
        """按已解析的命令行参数构建数据源实例。

        参数：
            args: 命令行参数命名空间，至少包含本数据源通过
                ``add_arguments`` 注册的字段。

        返回：
            可直接使用的数据源实例。
        """

    @abstractmethod
    def metadata(self) -> dict:
        """返回数据集的时间范围等元信息。

        返回：
            至少包含 ``first_time`` 与 ``last_time`` 两个键的字典，
            供主实验裁剪研究窗口使用。
        """

    @abstractmethod
    def list_symbols(self, limit: int) -> list[str]:
        """返回缺省研究证券池。

        参数：
            limit: 最多返回多少只证券。

        返回：
            证券代码列表。
        """

    @abstractmethod
    def load_bars(self, codes: Sequence[str], start, end) -> pd.DataFrame:
        """装载指定证券与区间的行情。

        参数：
            codes: 证券代码序列。
            start: 起始时间，包含该时刻或该日。
            end: 结束时间，包含该时刻或该日。

        返回：
            已通过对应频率契约校验的行情表。
        """

    @abstractmethod
    def build_features(
        self,
        bars: pd.DataFrame,
        feature_columns,
        cache_dir,
        factor_expressions,
    ) -> pd.DataFrame:
        """把行情表转换为带因子列的日频表。

        参数：
            bars: ``load_bars`` 的返回值。
            feature_columns: 要计算的注册因子名序列；``None`` 表示使用该数据源
                支持的默认因子集合。
            cache_dir: 因子缓存根目录；``None`` 表示不读写缓存。
            factor_expressions: 运行时 DSL 因子表达式序列，可为空。

        返回：
            按交易日与证券排序的日频因子表。
        """

    def cache_namespace(self) -> str:
        """返回因子缓存的命名空间，用于隔离不同数据源与不同口径的缓存。

        返回：
            相对缓存根目录的子路径；空串表示直接使用缓存根目录，
            这样既有的 5 分钟缓存路径保持逐字节不变。
        """
        return ""

    def describe(self) -> str:
        """返回一句话描述，写进运行日志便于复盘。

        返回：
            含数据源名与频率的中文描述。
        """
        return f"数据源 {self.name}（{self.frequency}）"
