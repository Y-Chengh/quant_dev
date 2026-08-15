"""日线库合法性审计的编排层。

按固定顺序驱动各 mixin：先装载参照数据，再依次跑完整性、成交状态、覆盖率、
价格连续性和可选的交叉校验，最后汇总落盘。全程只读，绝不修改日线库。
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from quant.config import default_market_database
from quant.qmt_downloader.self_check import AuditResult

from ..daily.database import DailyMarketDatabase
from .base import _DailyCheckerState
from .config import DailyCheckConfig
from .cross_5m import _CrossCheckMixin
from .loading import _ReferenceLoadingMixin
from .rules_coverage import _CoverageRulesMixin
from .rules_integrity import _IntegrityRulesMixin
from .rules_prices import _PriceRulesMixin
from .rules_volume import _VolumeRulesMixin
from .summary import _SummaryMixin

logger = logging.getLogger(__name__)


class DailyMarketChecker(
    _ReferenceLoadingMixin,
    _IntegrityRulesMixin,
    _VolumeRulesMixin,
    _CoverageRulesMixin,
    _PriceRulesMixin,
    _CrossCheckMixin,
    _SummaryMixin,
    _DailyCheckerState,
):
    """对入库后的日线库执行全样本合法性审计。"""

    def __init__(self, config: DailyCheckConfig) -> None:
        """初始化审计器并准备空的结果容器。

        参数：
            config: 审计配置与阈值，见 :class:`DailyCheckConfig`。

        返回：
            无返回值；实际检查在 ``run()`` 中执行。
        """
        self.config = config
        self.issues = []
        self.calendar = ()
        self.calendar_is_authoritative = False
        self.instruments = pd.DataFrame()
        self.codes = ()
        self.coverage_by_date = pd.DataFrame()
        self.coverage_by_symbol = pd.DataFrame()
        self.missing_spans = pd.DataFrame()
        self.adjust_audit = pd.DataFrame()
        self.price_limit_violations = pd.DataFrame()
        self.cross_check_diff = pd.DataFrame()

    def run(self) -> AuditResult:
        """执行全部审计规则并写出报告。

        返回：
            含摘要、问题明细与报告目录的 ``AuditResult``；存在 ERROR 时其
            ``exit_code`` 为 1。
        """
        started_at = datetime.now()
        repository = DailyMarketDatabase(self.config.database)
        with repository.connect() as connection:
            self._load_scope(connection)
            logger.info(
                "[market-check] 审计区间 %s 至 %s，证券 %d 只",
                self.start_date,
                self.end_date,
                len(self.codes),
            )
            self._prepare_scope_views(connection)
            self._load_calendar(connection)
            self._load_instruments(connection)

            logger.info("[market-check] 检查行级完整性")
            self._check_integrity(connection)
            logger.info("[market-check] 检查成交状态")
            multiplier = self._check_volume(connection)
            logger.info("[market-check] 检查覆盖率与缺失区间")
            self._check_coverage(connection)
            logger.info("[market-check] 检查价格连续性与除权自洽性")
            self._check_prices(connection)

            extras = {"volume_multiplier": multiplier, "cross_check_5m": self.config.cross_check_5m}
            if self.config.cross_check_5m:
                logger.info("[market-check] 与 5 分钟库交叉校验")
                market_database = (
                    self.config.market_database
                    if self.config.market_database is not None
                    else default_market_database()
                )
                self._check_cross_5m(connection, market_database, multiplier)

        report_dir = self.config.resolved_report_dir(started_at)
        result = self._build_result(report_dir, started_at, extras)
        logger.info(
            "[market-check] 完成: errors=%d warnings=%d 报告=%s",
            result.summary["errors"],
            result.summary["warnings"],
            report_dir,
        )
        return result

    def _prepare_scope_views(self, connection) -> None:
        """建立限定审计范围的临时视图。

        参数：
            connection: 只读日线库连接。临时视图存在于内存中的 ``temp`` 模式，
                只读连接同样可以创建，不会写入库文件。

        返回：
            无返回值；建立 ``scoped_codes`` 与 ``scoped_bars`` 两个临时视图。
        """
        connection.register("scoped_codes_frame", pd.DataFrame({"code": list(self.codes)}))
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW scoped_codes AS SELECT code FROM scoped_codes_frame"
        )
        clauses = [f"trade_date BETWEEN DATE '{self.start_date}' AND DATE '{self.end_date}'"]
        if self.config.codes:
            clauses.append("code IN (SELECT code FROM scoped_codes)")
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW scoped_bars AS "
            "SELECT * FROM bars_1d WHERE {0}".format(" AND ".join(clauses))
        )


def run_daily_market_check(config: DailyCheckConfig) -> AuditResult:
    """对日线库执行一次合法性审计。

    参数：
        config: 审计配置与阈值。

    返回：
        含摘要、问题明细与报告目录的 ``AuditResult``。
    """
    return DailyMarketChecker(config).run()
