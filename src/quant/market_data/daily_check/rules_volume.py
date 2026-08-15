"""成交状态校验：停牌与成交量、成交额的一致性，以及成交量单位标定。

大 QMT 没有明确说明 ``volume`` 的单位是股还是手，因此这里用
``median(amount / (volume * close))`` 标定一次：结果接近 1 说明是股，接近 100
说明是手。该系数会写进报告，交叉校验成交量时也依赖它。
"""

from __future__ import annotations

from .base import _DailyCheckerState

#: 每条规则最多列举多少个样例。
_SAMPLE_LIMIT = 20

#: 成交量单位的候选倍数：1 表示股，100 表示手。
_VOLUME_MULTIPLIER_CANDIDATES = (1.0, 100.0)

#: 标定值与候选倍数的最大允许相对偏差。
_MULTIPLIER_TOLERANCE = 0.2


class _VolumeRulesMixin(_DailyCheckerState):
    """校验停牌与成交的一致性，并标定成交量单位。"""

    def _check_volume(self, connection) -> float:
        """执行全部成交状态规则并返回标定出的成交量倍数。

        参数：
            connection: 只读日线库连接，已建好 ``scoped_bars`` 临时视图。

        返回：
            成交量单位倍数，1.0 表示股、100.0 表示手；无法标定时返回 1.0
            并记一条 WARNING。
        """
        self._check_active_zero_turnover(connection)
        self._check_suspended_turnover(connection)
        multiplier = self._calibrate_volume_multiplier(connection)
        self._check_vwap_range(connection, multiplier)
        return multiplier

    def _check_active_zero_turnover(self, connection) -> None:
        """检查非停牌日是否出现零成交。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        for issue_code, condition, field, message in (
            (
                "DAILY_ACTIVE_ZERO_VOLUME",
                "COALESCE(suspend_flag, 0) = 0 AND COALESCE(volume, 0) <= 0",
                "volume",
                "非停牌日成交量为零",
            ),
            (
                "DAILY_ACTIVE_ZERO_AMOUNT",
                "COALESCE(suspend_flag, 0) = 0 AND COALESCE(amount, 0) <= 0 "
                "AND COALESCE(volume, 0) > 0",
                "amount",
                "非停牌日有成交量却没有成交额",
            ),
        ):
            total = connection.execute(
                f"SELECT count(*) FROM scoped_bars WHERE {condition}"
            ).fetchone()[0]
            if not total:
                continue
            rows = connection.execute(
                f"""
                SELECT code, trade_date, volume, amount FROM scoped_bars
                WHERE {condition} ORDER BY code, trade_date LIMIT ?
                """,
                [_SAMPLE_LIMIT],
            ).fetchall()
            for code, trade_date, volume, amount in rows:
                self._add_issue(
                    issue_code,
                    "ERROR",
                    f"{message}（共 {total} 行）",
                    code=str(code),
                    date=str(trade_date),
                    field=field,
                    expected="suspend_flag=0 时成交量与成交额都应大于 0",
                    actual=f"volume={volume} amount={amount}",
                    possible_causes="停牌标记缺失，或该日行情本身不完整",
                    suggested_action="核对源分区该行的 suspendFlag 与成交数据",
                )

    def _check_suspended_turnover(self, connection) -> None:
        """检查停牌日是否出现成交。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        condition = "COALESCE(suspend_flag, 0) <> 0 AND (COALESCE(volume, 0) > 0 " \
                    "OR COALESCE(amount, 0) > 0)"
        total = connection.execute(
            f"SELECT count(*) FROM scoped_bars WHERE {condition}"
        ).fetchone()[0]
        if not total:
            return
        rows = connection.execute(
            f"""
            SELECT code, trade_date, volume, amount FROM scoped_bars
            WHERE {condition} ORDER BY code, trade_date LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for code, trade_date, volume, amount in rows:
            self._add_issue(
                "DAILY_SUSPENDED_WITH_TURNOVER",
                "ERROR",
                f"停牌日却有成交（共 {total} 行）",
                code=str(code),
                date=str(trade_date),
                expected="suspend_flag=1 时成交量与成交额都应为 0",
                actual=f"volume={volume} amount={amount}",
                possible_causes="QMT 的停牌标记与成交数据来自不同快照",
                suggested_action="以成交数据为准判断是否真的停牌，必要时重新下载该日",
            )

    def _calibrate_volume_multiplier(self, connection) -> float:
        """标定成交量单位：股还是手。

        参数：
            connection: 只读日线库连接。

        返回：
            最接近的候选倍数；标定值离两个候选都太远时返回 1.0 并记一条 WARNING。
        """
        row = connection.execute(
            """
            SELECT median(amount / (volume * close)) FROM scoped_bars
            WHERE COALESCE(suspend_flag, 0) = 0
              AND volume > 0 AND close > 0 AND amount > 0
            """
        ).fetchone()
        if not row or row[0] is None:
            return 1.0
        estimated = float(row[0])
        best = min(
            _VOLUME_MULTIPLIER_CANDIDATES,
            key=lambda candidate: abs(estimated / candidate - 1.0),
        )
        if abs(estimated / best - 1.0) > _MULTIPLIER_TOLERANCE:
            self._add_issue(
                "DAILY_VOLUME_UNIT_AMBIGUOUS",
                "WARNING",
                "无法确定成交量单位是股还是手",
                field="volume",
                expected="amount/(volume*close) 应接近 1（股）或 100（手）",
                actual=f"{estimated:.4f}",
                possible_causes="成交额或成交量口径与预期不同，或数据本身有误",
                suggested_action="人工确认单位后再解读成交量相关的因子与校验",
            )
            return 1.0
        return best

    def _check_vwap_range(self, connection, multiplier: float) -> None:
        """检查由成交额与成交量推出的均价是否落在当日价格区间内。

        参数：
            connection: 只读日线库连接。
            multiplier: ``_calibrate_volume_multiplier`` 标定出的成交量倍数。

        返回：
            无返回值。
        """
        condition = (
            "COALESCE(suspend_flag, 0) = 0 AND volume > 0 AND amount > 0 "
            "AND low > 0 AND high > 0 "
            f"AND (amount / (volume * {multiplier}) < low * 0.99 "
            f"OR amount / (volume * {multiplier}) > high * 1.01)"
        )
        total = connection.execute(
            f"SELECT count(*) FROM scoped_bars WHERE {condition}"
        ).fetchone()[0]
        if not total:
            return
        rows = connection.execute(
            f"""
            SELECT code, trade_date, low, high, amount / (volume * {multiplier}) AS vwap
            FROM scoped_bars WHERE {condition} ORDER BY code, trade_date LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for code, trade_date, low, high, vwap in rows:
            self._add_issue(
                "DAILY_VWAP_OUT_OF_RANGE",
                "WARNING",
                f"成交均价落在当日价格区间之外（共 {total} 行）",
                code=str(code),
                date=str(trade_date),
                expected=f"{low} 至 {high}",
                actual=f"{vwap:.4f}",
                evidence=f"成交量单位倍数按 {multiplier:g} 计算",
                possible_causes="成交额与成交量口径不一致，或该行价格有误",
                suggested_action="核对源数据；若整体偏移，说明成交量单位标定有误",
            )
