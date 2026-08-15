"""与 5 分钟行情库的交叉校验。

把 ``market.duckdb`` 以只读方式 ``ATTACH`` 进来，把 5 分钟 bar 聚合成日频后与
``bars_1d`` 逐行比对。两个库来自完全不同的数据源（iFinD 与大 QMT），因此对得上
是很强的正确性证据，对不上则说明至少有一侧有问题。

几个必须处理的边界：

- 两库的证券宇宙和时间范围都不一样，只比交集，否则会刷屏。
- 5 分钟库同样不复权，因此差异不是复权造成的假象。
- 停牌日没有 5 分钟 bar，不能算作「日线多出来的行」。
- 成交量单位可能不同，用标定出的倍数换算后按相对误差比较。
"""

from __future__ import annotations

import pandas as pd

from .base import _DailyCheckerState

#: 每类差异最多列举多少个样例。
_SAMPLE_LIMIT = 20

_DIFF_COLUMNS = [
    "code",
    "trade_date",
    "daily_close",
    "intraday_close",
    "daily_volume",
    "intraday_volume",
    "reason",
]


class _CrossCheckMixin(_DailyCheckerState):
    """用 5 分钟聚合结果校验日线。"""

    def _check_cross_5m(self, connection, market_database, multiplier: float) -> None:
        """执行 5 分钟与日线的交叉校验。

        参数：
            connection: 只读日线库连接，已建好 ``scoped_bars`` 临时视图。
            market_database: 5 分钟库 ``market.duckdb`` 路径。
            multiplier: 日线成交量的单位倍数，用于折算到 5 分钟库的口径。

        返回：
            无返回值；差异明细写入 ``cross_check_diff``。
        """
        self.cross_check_diff = pd.DataFrame(columns=_DIFF_COLUMNS)
        path = str(market_database)
        try:
            connection.execute("ATTACH '{0}' AS m (READ_ONLY)".format(path.replace("'", "''")))
        except Exception as error:
            self._add_issue(
                "CROSS_5M_UNAVAILABLE",
                "INFO",
                "无法附加 5 分钟行情库，跳过交叉校验",
                expected=f"可只读打开 {path}",
                actual=str(error),
                possible_causes="该库不存在，或正被其它进程以读写方式占用",
                suggested_action="确认 5 分钟库路径与占用情况后重试",
            )
            return
        try:
            self._compare(connection, multiplier)
        finally:
            connection.execute("DETACH m")

    def _compare(self, connection, multiplier: float) -> None:
        """比对聚合后的 5 分钟行情与日线。

        参数：
            connection: 已附加 ``m`` 库的只读连接。
            multiplier: 日线成交量的单位倍数。

        返回：
            无返回值。
        """
        tolerance = self.config.cross_check_tolerance
        diff = connection.execute(
            """
            WITH agg AS (
                SELECT code, CAST(trade_time AS DATE) AS trade_date,
                       arg_max(close, trade_time) AS close,
                       sum(volume) AS volume
                FROM m.bars_5m
                WHERE CAST(trade_time AS DATE) BETWEEN ? AND ?
                GROUP BY 1, 2
            ),
            overlap AS (
                SELECT code FROM (SELECT DISTINCT code FROM agg)
                INTERSECT
                SELECT code FROM (SELECT DISTINCT code FROM scoped_bars)
            ),
            joined AS (
                SELECT COALESCE(d.code, a.code) AS code,
                       COALESCE(d.trade_date, a.trade_date) AS trade_date,
                       d.close AS daily_close, a.close AS intraday_close,
                       d.volume AS daily_volume, a.volume AS intraday_volume,
                       COALESCE(d.suspend_flag, 0) AS suspend_flag
                FROM scoped_bars d
                FULL OUTER JOIN agg a ON a.code = d.code AND a.trade_date = d.trade_date
                WHERE COALESCE(d.code, a.code) IN (SELECT code FROM overlap)
            )
            SELECT code, trade_date, daily_close, intraday_close,
                   daily_volume, intraday_volume,
                   CASE
                       WHEN daily_close IS NULL THEN 'missing_daily'
                       WHEN intraday_close IS NULL AND suspend_flag = 0 THEN 'missing_intraday'
                       WHEN intraday_close IS NULL THEN 'suspended'
                       WHEN abs(daily_close - intraday_close)
                            > ? * greatest(abs(daily_close), 1e-9) THEN 'close_mismatch'
                       WHEN abs(daily_volume * ? - intraday_volume)
                            > ? * greatest(intraday_volume, 1) THEN 'volume_mismatch'
                       ELSE 'ok'
                   END AS reason
            FROM joined
            ORDER BY code, trade_date
            """,
            [self.start_date, self.end_date, tolerance, multiplier, tolerance],
        ).df()
        if diff.empty:
            return
        diff = diff[~diff["reason"].isin(("ok", "suspended"))]
        self.cross_check_diff = diff.loc[:, _DIFF_COLUMNS].reset_index(drop=True)
        if diff.empty:
            return
        descriptions = {
            "missing_daily": (
                "CROSS_5M_MISSING_DAILY",
                "5 分钟库有该交易日行情，日线库没有",
                "日线库应覆盖同一 (证券, 交易日)",
            ),
            "missing_intraday": (
                "CROSS_5M_MISSING_INTRADAY",
                "日线库有非停牌行情，5 分钟库没有",
                "5 分钟库应覆盖同一 (证券, 交易日)",
            ),
            "close_mismatch": (
                "CROSS_5M_CLOSE_MISMATCH",
                "日线收盘价与 5 分钟聚合收盘价不一致",
                f"两者相对误差不超过 {tolerance:g}",
            ),
            "volume_mismatch": (
                "CROSS_5M_VOLUME_MISMATCH",
                "日线成交量与 5 分钟聚合成交量不一致",
                f"换算单位后相对误差不超过 {tolerance:g}",
            ),
        }
        for reason, group in diff.groupby("reason"):
            issue_code, message, expected = descriptions[str(reason)]
            for row in group.head(_SAMPLE_LIMIT).itertuples(index=False):
                self._add_issue(
                    issue_code,
                    "WARNING",
                    f"{message}（共 {len(group)} 处）",
                    code=str(row.code),
                    date=str(row.trade_date),
                    expected=expected,
                    actual=f"daily_close={row.daily_close} intraday_close={row.intraday_close} daily_volume={row.daily_volume} "
                    f"intraday_volume={row.intraday_volume}",
                    evidence="5 分钟库与日线库来自不同数据源，均为不复权口径",
                    possible_causes="其中一侧的数据缺失或有误；也可能两库的成交量单位不同",
                    suggested_action="以两库交集为准逐条核对，必要时补下载对应交易日",
                )
