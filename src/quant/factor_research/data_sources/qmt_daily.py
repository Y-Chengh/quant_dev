"""基于大 QMT 日线库的数据源。

与 5 分钟数据源的三点关键差异：

1. 只有日频行情，因此依赖 ``intraday_values`` 的八个因子不可用。
2. 库里存的是不复权价，复权在读取时按 ``--adjust`` 计算；研究缺省用后复权，
   因为它的历史取值不随新的分红送转改变，既不引入未来信息，也不会让因子缓存
   在每次除权后整体失效。
3. 构造时会先跑一次轻量增量检查，把新下载的交易日自动并进库里。
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from quant.config import default_qmt_daily_database

from ..data import validate_daily_bars
from ..factors import build_daily_features_from_daily
from .base import FREQUENCY_DAILY, MarketDataSource
from .registry import register_data_source

logger = logging.getLogger(__name__)

#: ``--adjust`` 的可选值，与 ``AdjustMode`` 的取值保持一致。
ADJUST_CHOICES = ("none", "qfq", "hfq")

#: 研究缺省复权口径。
DEFAULT_ADJUST = "hfq"


@register_data_source
class QmtDailySource(MarketDataSource):
    """从 ``qmt_daily.duckdb`` 的 ``bars_1d`` 视图读取日线行情。"""

    name = "qmt_daily"
    frequency = FREQUENCY_DAILY
    provides_intraday = False

    def __init__(
        self,
        database: Path,
        adjust: str = DEFAULT_ADJUST,
        include_suspended: bool = False,
        adjust_anchor: date | None = None,
        adjust_volume: bool = False,
    ) -> None:
        """绑定日线库并记录复权口径。

        参数：
            database: ``qmt_daily.duckdb`` 文件路径。
            adjust: 复权口径，取值见 ``ADJUST_CHOICES``。
            include_suspended: 是否保留停牌日；缺省剔除，避免假零收益污染因子。
            adjust_anchor: 前复权基准日；``None`` 时使用库中记录的基准日。
            adjust_volume: 是否同步反向调整成交量；缺省保留原始股数。

        返回：
            无返回值；DuckDB 客户端在首次使用时惰性建立。
        """
        self.database = database
        self.adjust = adjust
        self.include_suspended = include_suspended
        self.adjust_anchor = adjust_anchor
        self.adjust_volume = adjust_volume
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """注册日线数据源的专属参数。

        参数：
            parser: 目标解析器；新增日线库路径、复权口径、停牌处理等参数，
                并一并挂上日线库增量同步的通用开关。

        返回：
            无返回值。
        """
        from quant.market_data.daily.ingest.cli_support import add_sync_arguments

        parser.add_argument(
            "--daily-database",
            type=Path,
            default=default_qmt_daily_database(),
            help="qmt_daily.duckdb 路径；缺省按 QMT_DAILY_DB_PATH 环境变量解析",
        )
        parser.add_argument(
            "--adjust",
            choices=ADJUST_CHOICES,
            default=DEFAULT_ADJUST,
            help=(
                "复权口径：hfq 后复权（默认，历史取值稳定），"
                "qfq 前复权（会随新分红重算历史价），none 不复权"
            ),
        )
        parser.add_argument(
            "--adjust-anchor",
            help="前复权基准日，格式 YYYY-MM-DD；缺省使用库中记录的基准日",
        )
        parser.add_argument(
            "--include-suspended",
            action="store_true",
            help="保留停牌日行情；缺省剔除，避免停牌日的平价零量造出假零收益",
        )
        parser.add_argument(
            "--adjust-volume",
            action="store_true",
            help="复权时同步反向调整成交量；缺省保留原始股数",
        )
        add_sync_arguments(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> QmtDailySource:
        """按命令行参数构建日线数据源，并在此之前完成自动增量同步。

        同步必须在建立读连接**之前**完成：DuckDB 是单写多读，先持有读连接会
        挡住随后的写入。

        参数：
            args: 命令行参数命名空间，含本数据源与同步开关注册的字段。

        返回：
            绑定好数据库路径与复权口径的数据源实例。
        """
        from quant.market_data.daily.ingest.cli_support import sync_from_args

        database = Path(getattr(args, "daily_database", None) or default_qmt_daily_database())
        report = sync_from_args(args, database=database)
        if report is not None:
            logger.info("[qmt_daily] 增量同步: %s %s", report.status, report.message)
        anchor_text = getattr(args, "adjust_anchor", None)
        adjust = getattr(args, "adjust", DEFAULT_ADJUST)
        anchor = pd.Timestamp(anchor_text).date() if anchor_text else None
        if adjust == "qfq" and anchor is None:
            # 前复权的基准日必须在此处定死并进入缓存命名空间。库里记录的
            # adjust_anchor_date 每同步一次就前移一天，若放任它当缺省值，
            # 同一条命令在两天之间会算出不同的历史价，却共用同一份因子缓存。
            from quant.market_data.daily import DailyMarketClient

            recorded = DailyMarketClient(database).get_metadata().get("adjust_anchor_date", "")
            if not recorded:
                raise ValueError(
                    "使用 --adjust qfq 时必须显式指定 --adjust-anchor：日线库尚未记录基准日"
                )
            anchor = pd.Timestamp(recorded).date()
            logger.warning(
                "[qmt_daily] --adjust qfq 未指定基准日，本次固定为库中记录的 %s；"
                "下次同步后该值会前移，建议显式传 --adjust-anchor 以保证可复现",
                anchor,
            )
        return cls(
            database=database,
            adjust=adjust,
            include_suspended=bool(getattr(args, "include_suspended", False)),
            adjust_anchor=anchor,
            adjust_volume=bool(getattr(args, "adjust_volume", False)),
        )

    def client(self):
        """返回惰性创建的日线客户端。

        返回：
            ``DailyMarketClient`` 实例。
        """
        if self._client is None:
            from quant.market_data.daily import DailyMarketClient

            self._client = DailyMarketClient(self.database)
        return self._client

    def metadata(self) -> dict:
        """返回日线库的时间范围与规模。

        返回：
            含 ``first_time``、``last_time``、``rows``、``symbols`` 的字典；
            前两项是交易日零点的 ISO 文本，形状与 5 分钟库对齐。
        """
        return self.client().get_metadata()

    def list_symbols(self, limit: int) -> list[str]:
        """返回库覆盖区间内**任一时点**上市过的证券。

        走 ``instruments`` 的上市退市信息而不是「库里有行情的代码」，并且用整个
        库区间而不是某一天的截面：只取最后一天的截面会漏掉在那之前就已退市的
        证券，正好把幸存者偏差重新引回来。

        注意本接口没有研究窗口参数，因此只能按库区间给出一个偏保守（偏大）的
        池子；需要严格按研究窗口取池时，请直接用
        ``DailyMarketClient.list_universe_over_window``。

        参数：
            limit: 最多返回多少只证券。

        返回：
            按代码升序排列的证券代码列表。
        """
        client = self.client()
        metadata = client.get_metadata()
        first, last = metadata.get("first_date"), metadata.get("last_date")
        if first is None or last is None:
            return client.search_symbols("", limit=limit)
        return client.list_universe_over_window(
            pd.Timestamp(first).date(), pd.Timestamp(last).date(), limit=limit
        )

    def load_bars(self, codes: Sequence[str], start, end) -> pd.DataFrame:
        """读取指定证券与区间的日线行情并按配置复权。

        参数：
            codes: 证券代码序列。
            start: 起始时间；只取日期部分，包含该日。
            end: 结束时间；只取日期部分，包含该日。

        返回：
            已通过日频行情契约校验的行情表。
        """
        from quant.market_data.daily import AdjustMode

        bars = self.client().get_klines_1d(
            codes,
            pd.Timestamp(start).date(),
            pd.Timestamp(end).date(),
            adjust=AdjustMode(self.adjust),
            include_suspended=self.include_suspended,
            adjust_anchor=self.adjust_anchor,
            adjust_volume=self.adjust_volume,
        )
        return validate_daily_bars(bars)

    def build_features(
        self,
        bars: pd.DataFrame,
        feature_columns,
        cache_dir,
        factor_expressions,
    ) -> pd.DataFrame:
        """按日频行情构建因子表。

        参数：
            bars: ``load_bars`` 返回的日频行情表。
            feature_columns: 要计算的注册因子名序列；``None`` 表示使用全部
                日频可算因子，此时依赖分钟行情的因子会被自动跳过并告警。
            cache_dir: 因子缓存根目录；``None`` 表示不使用缓存。
            factor_expressions: 运行时 DSL 因子表达式序列。

        返回：
            日频因子表。
        """
        return build_daily_features_from_daily(
            bars,
            feature_columns=feature_columns,
            cache_dir=cache_dir,
            factor_expressions=factor_expressions,
            cache_namespace=self.cache_namespace(),
            source_name=self.name,
        )

    def cache_namespace(self) -> str:
        """返回隔离本数据源因子缓存的命名空间。

        复权口径与前复权基准日都进命名空间：不同口径下同一天的价格不同，
        缓存绝不能互相命中。

        返回：
            形如 ``qmt_daily/hfq/-/no_susp/rawvol`` 的相对路径。
        """
        anchor = self.adjust_anchor.isoformat() if self.adjust_anchor else "-"
        suspended = "with_susp" if self.include_suspended else "no_susp"
        volume = "adjvol" if self.adjust_volume else "rawvol"
        return f"{self.name}/{self.adjust}/{anchor}/{suspended}/{volume}"
