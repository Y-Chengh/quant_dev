"""基于本地 5 分钟行情库的数据源，即项目原有的默认行为。

这里刻意逐字沿用原来的调用链：``MarketDataClient`` → ``load_market_service``
→ ``build_daily_features``，并且 ``cache_namespace`` 返回空串，
因此既有实验的代码路径和因子缓存路径都逐字节不变。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quant.config import default_market_database

from ..data import load_market_service
from ..factors import build_daily_features
from .base import FREQUENCY_INTRADAY, MarketDataSource
from .registry import register_data_source


@register_data_source
class MarketServiceSource(MarketDataSource):
    """从 ``market.duckdb`` 的 ``bars_5m`` 视图读取 5 分钟行情。"""

    name = "market_service"
    frequency = FREQUENCY_INTRADAY
    provides_intraday = True

    def __init__(self, database: Path) -> None:
        """绑定 5 分钟行情库。

        参数：
            database: ``market.duckdb`` 文件路径。

        返回：
            无返回值；DuckDB 客户端在首次使用时惰性建立。
        """
        self.database = database
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """注册 5 分钟数据源的专属参数。

        参数：
            parser: 目标解析器；新增 ``--database``。

        返回：
            无返回值。
        """
        parser.add_argument(
            "--database",
            type=Path,
            default=default_market_database(),
            help="market.duckdb 路径；缺省按 MARKET_DB_PATH 环境变量解析",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MarketServiceSource:
        """按命令行参数构建 5 分钟数据源。

        参数：
            args: 命令行参数命名空间；读取其中的 ``database`` 字段，
                缺失时回落到默认行情库路径。

        返回：
            绑定好数据库路径的数据源实例。
        """
        database = getattr(args, "database", None) or default_market_database()
        return cls(Path(database))

    def client(self):
        """返回惰性创建的行情客户端。

        把 ``duckdb`` 的导入推迟到真正取数时，``--help`` 与参数校验不必付出
        加载代价，缺少依赖时的报错也更贴近使用现场。

        返回：
            ``MarketDataClient`` 实例。
        """
        if self._client is None:
            from quant.market_data.client import MarketDataClient

            self._client = MarketDataClient(self.database)
        return self._client

    def metadata(self) -> dict:
        """返回 5 分钟库的时间范围与规模。

        返回：
            含 ``first_time``、``last_time``、``rows``、``symbols`` 的字典。
        """
        return self.client().get_metadata()

    def list_symbols(self, limit: int) -> list[str]:
        """返回代码表中的前若干只证券。

        参数：
            limit: 最多返回多少只证券。

        返回：
            证券代码列表。注意它只反映「库里有没有行情」，不是无幸存者偏差的池子。
        """
        return self.client().search_symbols("", limit=limit)

    def load_bars(self, codes: Sequence[str], start, end) -> pd.DataFrame:
        """读取指定证券与区间的 5 分钟行情。

        参数：
            codes: 证券代码序列。
            start: 起始时间，包含该时刻。
            end: 结束时间，包含该时刻。

        返回：
            已通过分钟行情契约校验的行情表。
        """
        return load_market_service(self.client(), codes, start, end)

    def build_features(
        self,
        bars: pd.DataFrame,
        feature_columns,
        cache_dir,
        factor_expressions,
    ) -> pd.DataFrame:
        """按分钟行情构建日频因子表。

        参数：
            bars: ``load_bars`` 返回的分钟行情表。
            feature_columns: 要计算的注册因子名序列；``None`` 表示默认全部因子。
            cache_dir: 因子缓存根目录；``None`` 表示不使用缓存。
            factor_expressions: 运行时 DSL 因子表达式序列。

        返回：
            日频因子表，与改动前的实现完全一致。
        """
        return build_daily_features(
            bars,
            feature_columns=feature_columns,
            cache_dir=cache_dir,
            factor_expressions=factor_expressions,
        )
