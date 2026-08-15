"""价格连续性、除权自洽性与涨跌停校验。

除权因子的口径已在真实数据上标定：``adjustment_factor(t)`` 应当等于
``close(t-1) / pre_close(t)``。实测 2024 年全年 4524 个事件上，两者比值的
10%~90% 分位落在 ``[1.0, 1.000001]``。因此这条关系可以当作硬约束来校验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _DailyCheckerState
from .limits import resolve_limit_ratio

#: 每条规则最多列举多少个样例。
_SAMPLE_LIMIT = 20

#: A 股报价最小变动单位。``pre_close`` 与 ``adjustment_factor`` 在源数据里是各自
#: 独立四舍五入的，两次舍入最多相差一个完整报价单位，因此一切「前收应当等于某个
#: 值」的比较都要容忍一整分钱。实测按纯相对容差 1e-4 判定，2024 年会多出四百余条
#: 纯舍入造成的假阳性。
_PRICE_TICK = 0.01


def _price_tolerance(reference, relative_tolerance: float):
    """计算价格比较的绝对容差。

    参数：
        reference: 参考价格，可以是标量或 ``Series``；容差随价格线性增长的部分
            以它为基数。
        relative_tolerance: 相对容差，来自配置。

    返回：
        与 ``reference`` 同形的绝对容差：一个最小报价单位加上相对容差部分。
    """
    return _PRICE_TICK + relative_tolerance * abs(reference)


class _PriceRulesMixin(_DailyCheckerState):
    """校验前收连续性、除权因子与涨跌停。"""

    def _check_prices(self, connection) -> None:
        """执行全部价格类规则。

        参数：
            connection: 只读日线库连接，已建好 ``scoped_bars`` 临时视图。

        返回：
            无返回值；明细写入 ``adjust_audit`` 与 ``price_limit_violations``。
        """
        continuity = self._load_continuity(connection)
        self._check_pre_close_continuity(connection, continuity)
        self._check_adjust_factor(connection, continuity)
        self._check_corporate_action_without_bar(connection)
        self._check_price_limits(continuity)

    def _load_continuity(self, connection) -> pd.DataFrame:
        """取出每根 K 线及其上一交易日收盘价，并关联当日除权记录。

        必须**包含停牌行**：停牌日也是一行，跳过它会让 ``lag`` 跨过日历空隙而
        算出错误的前收。

        参数：
            connection: 只读日线库连接。

        返回：
            含 ``code``、``trade_date``、``close``、``pre_close``、
            ``previous_close``、``suspend_flag``、``adjustment_factor`` 及各项
            除权字段的表。
        """
        return connection.execute(
            """
            WITH ordered AS (
                SELECT code, trade_date, close, pre_close, suspend_flag,
                       lag(close) OVER (PARTITION BY code ORDER BY trade_date) AS previous_close
                FROM scoped_bars
            )
            SELECT o.*, a.adjustment_factor, a.cash_dividend_per_share,
                   a.bonus_share_per_share, a.capitalization_per_share,
                   a.rights_issue_per_share, a.rights_issue_price
            FROM ordered o
            LEFT JOIN corporate_actions a
              ON a.code = o.code AND a.ex_date = o.trade_date
            ORDER BY o.code, o.trade_date
            """
        ).df()

    def _check_pre_close_continuity(self, connection, continuity: pd.DataFrame) -> None:
        """检查前收与上一交易日收盘是否一致，除权日除外。

        参数：
            connection: 只读日线库连接，仅用于保持各规则签名一致。
            continuity: ``_load_continuity`` 的返回值。

        返回：
            无返回值。
        """
        if continuity.empty:
            return
        previous = pd.to_numeric(continuity["previous_close"], errors="coerce")
        pre_close = pd.to_numeric(continuity["pre_close"], errors="coerce")
        has_action = continuity["adjustment_factor"].notna()
        valid = np.isfinite(previous) & (previous > 0) & np.isfinite(pre_close) & (pre_close > 0)
        gap = (pre_close - previous).abs()
        allowed = _price_tolerance(previous, self.config.pre_close_tolerance)
        broken = valid & ~has_action & (gap > allowed)
        total = int(broken.sum())
        if not total:
            return
        for row in continuity[broken].head(_SAMPLE_LIMIT).itertuples(index=False):
            self._add_issue(
                "DAILY_PRE_CLOSE_MISMATCH",
                "ERROR",
                f"前收与上一交易日收盘不一致，且当日没有除权记录（共 {total} 处）",
                code=str(row.code),
                date=str(row.trade_date),
                field="pre_close",
                expected=f"{row.previous_close}",
                actual=f"{row.pre_close}",
                evidence=f"相对偏差 {abs(row.pre_close - row.previous_close) / row.previous_close:.6f}",
                possible_causes="除权记录缺失，或该证券在此处存在未记录的股本变动",
                suggested_action="补下载该日 corporate_actions 分区后重新同步",
            )

    def _check_adjust_factor(self, connection, continuity: pd.DataFrame) -> None:
        """检查除权因子是否与行情自洽。

        参数：
            connection: 只读日线库连接，仅用于保持各规则签名一致。
            continuity: ``_load_continuity`` 的返回值。

        返回：
            无返回值；明细写入 ``adjust_audit``。
        """
        columns = ["code", "trade_date", "observed", "expected", "deviation"]
        if continuity.empty:
            self.adjust_audit = pd.DataFrame(columns=columns)
            return
        events = continuity[continuity["adjustment_factor"].notna()].copy()
        if events.empty:
            self.adjust_audit = pd.DataFrame(columns=columns)
            return
        previous = pd.to_numeric(events["previous_close"], errors="coerce")
        pre_close = pd.to_numeric(events["pre_close"], errors="coerce")
        factor = pd.to_numeric(events["adjustment_factor"], errors="coerce")
        valid = (
            np.isfinite(previous) & (previous > 0)
            & np.isfinite(pre_close) & (pre_close > 0)
            & np.isfinite(factor) & (factor > 0)
        )
        observed = (previous / pre_close).where(valid)
        events["observed"] = observed
        events["expected"] = factor
        events["deviation"] = ((observed / factor) - 1.0).abs()
        self.adjust_audit = events.loc[:, columns].reset_index(drop=True)

        # 在价格维度而不是比值维度比较：源数据的 pre_close 已按分四舍五入，
        # 直接比比值会把低价股的正常舍入当成因子错误。
        implied_pre_close = (previous / factor).where(valid)
        allowed = _price_tolerance(pre_close, self.config.adjust_factor_tolerance)
        broken = (implied_pre_close - pre_close).abs() > allowed
        total = int(broken.fillna(False).sum())
        if total:
            for row in events[broken.fillna(False)].head(_SAMPLE_LIMIT).itertuples(index=False):
                self._add_issue(
                    "DAILY_ADJUST_FACTOR_MISMATCH",
                    "ERROR",
                    f"除权因子与行情对不上（共 {total} 处）",
                    code=str(row.code),
                    date=str(row.trade_date),
                    field="adjustment_factor",
                    expected=f"close(t-1)/pre_close(t) = {row.observed:.6f}",
                    actual=f"{row.expected:.6f}",
                    evidence=f"相对偏差 {row.deviation:.6f}",
                    possible_causes="QMT 的除权因子与行情来自不同快照，或该次事件被重复记录",
                    suggested_action="以行情推出的比例为准，或重新下载该日 corporate_actions",
                )
        self._check_theoretical_ratio(events)

    def _check_theoretical_ratio(self, events: pd.DataFrame) -> None:
        """用分红送转字段推出的理论比例交叉验证观测比例。

        参数：
            events: 已附带 ``observed`` 列的除权日明细。

        返回：
            无返回值。
        """
        cash = pd.to_numeric(events["cash_dividend_per_share"], errors="coerce").fillna(0.0)
        bonus = pd.to_numeric(events["bonus_share_per_share"], errors="coerce").fillna(0.0)
        capital = pd.to_numeric(events["capitalization_per_share"], errors="coerce").fillna(0.0)
        rights = pd.to_numeric(events["rights_issue_per_share"], errors="coerce").fillna(0.0)
        price = pd.to_numeric(events["rights_issue_price"], errors="coerce").fillna(0.0)
        previous = pd.to_numeric(events["previous_close"], errors="coerce")
        theory_pre_close = (previous - cash + rights * price) / (1.0 + bonus + capital + rights)
        valid = np.isfinite(theory_pre_close) & (theory_pre_close > 0) & (previous > 0)
        theoretical = (previous / theory_pre_close).where(valid)
        # 直接比理论前收与实际前收；容差放宽一档，因为分红送转字段本身常有缺项。
        expected_pre_close = theory_pre_close.where(valid)
        allowed = _price_tolerance(
            expected_pre_close, max(self.config.adjust_factor_tolerance * 10, 1e-3)
        )
        actual_pre_close = pd.to_numeric(events["pre_close"], errors="coerce")
        broken = (expected_pre_close - actual_pre_close).abs() > allowed
        total = int(broken.fillna(False).sum())
        if not total:
            return
        sample = events[broken.fillna(False)].head(_SAMPLE_LIMIT)
        for position, row in enumerate(sample.itertuples(index=False)):
            self._add_issue(
                "DAILY_ADJUST_THEORY_MISMATCH",
                "WARNING",
                f"行情推出的除权比例与分红送转字段推出的理论值不一致（共 {total} 处）",
                code=str(row.code),
                date=str(row.trade_date),
                expected=f"理论比例 {theoretical[sample.index[position]]:.6f}",
                actual=f"观测比例 {row.observed:.6f}",
                possible_causes="除权字段不完整（例如缺少配股信息），或存在特殊股本变动",
                suggested_action="以观测比例为准；如需精确还原，请核对该次公司行为公告",
            )

    def _check_corporate_action_without_bar(self, connection) -> None:
        """检查有除权记录但当日没有 K 线的情况。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        rows = connection.execute(
            """
            SELECT a.code, a.ex_date FROM corporate_actions a
            JOIN scoped_life l ON l.code = a.code
            LEFT JOIN scoped_bars b ON b.code = a.code AND b.trade_date = a.ex_date
            WHERE a.ex_date BETWEEN ? AND ?
              AND a.ex_date >= l.open_date
              AND (l.expire_date IS NULL OR a.ex_date <= l.expire_date)
              AND b.code IS NULL
            ORDER BY a.code, a.ex_date LIMIT ?
            """,
            [self.start_date, self.end_date, _SAMPLE_LIMIT],
        ).fetchall()
        for code, ex_date in rows:
            self._add_issue(
                "DAILY_CORPORATE_ACTION_WITHOUT_BAR",
                "WARNING",
                "存在除权记录但当日没有行情",
                code=str(code),
                date=str(ex_date),
                expected="除权日应有一根 K 线",
                actual="缺少该日行情",
                possible_causes="该日行情缺失，或除权日落在非交易日",
                suggested_action="补下载该交易日行情后重新同步",
            )

    def _check_price_limits(self, continuity: pd.DataFrame) -> None:
        """检查收盘涨跌幅是否越过所属板块的涨跌停限制。

        默认按板块基础幅度校验，**不**按当前 ST 状态收紧：历史风险警示状态不可
        知，收紧会产生大量误报。这样校验的是「涨跌幅有没有超过该板块可能的最大
        幅度」，是必要条件而非精确判定。

        参数：
            continuity: ``_load_continuity`` 的返回值。

        返回：
            无返回值；明细写入 ``price_limit_violations``。
        """
        columns = ["code", "trade_date", "pct_change", "limit_ratio"]
        if continuity.empty or self.instruments.empty:
            self.price_limit_violations = pd.DataFrame(columns=columns)
            return
        frame = continuity[continuity["suspend_flag"].fillna(0) == 0].copy()
        previous = pd.to_numeric(frame["previous_close"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        # 除权日的名义涨跌幅本来就不连续，必须排除。
        frame = frame[frame["adjustment_factor"].isna()]
        previous = previous.loc[frame.index]
        close = close.loc[frame.index]
        valid = np.isfinite(previous) & (previous > 0) & np.isfinite(close) & (close > 0)
        frame = frame[valid]
        if frame.empty:
            self.price_limit_violations = pd.DataFrame(columns=columns)
            return
        frame["pct_change"] = close.loc[frame.index] / previous.loc[frame.index] - 1.0

        boards = self.instruments["board"].astype(str).to_dict()
        st_flags = self.instruments["is_st"].fillna(False).astype(bool).to_dict()
        # 新股豁免必须按交易日计数：遇上国庆长假，5 个交易日会跨越 13 个自然日。
        sessions = self._sessions_since_listing(frame)

        ratios = []
        for position, (code, trade_date) in enumerate(zip(frame["code"], frame["trade_date"])):
            ratios.append(
                resolve_limit_ratio(
                    boards.get(code, "unknown"),
                    st_flags.get(code, False),
                    pd.Timestamp(trade_date).date(),
                    sessions[position],
                    apply_st_limit=self.config.apply_st_limit,
                )
            )
        frame["limit_ratio"] = ratios
        checked = frame[frame["limit_ratio"].notna()].copy()
        violations = checked[
            checked["pct_change"].abs() > checked["limit_ratio"] + self.config.price_limit_tolerance
        ]
        self.price_limit_violations = violations.loc[:, columns].reset_index(drop=True)
        if violations.empty:
            return
        self._report_price_limits(violations)

    def _sessions_since_listing(self, frame: pd.DataFrame) -> list[int | None]:
        """按交易日历计算每一行距该证券上市日经过了几个交易日。

        参数：
            frame: 待判定的行情表，至少含 ``code`` 与 ``trade_date`` 两列。

        返回：
            与 ``frame`` 等长的列表，上市首日为 0；上市日未知或不在交易日历
            覆盖范围内时为 ``None``。
        """
        if not self.calendar:
            return [None] * len(frame)
        ordinals = {value: index for index, value in enumerate(self.calendar)}
        # 上市日本身可能早于审计区间，用二分找出它在日历中的位置。
        calendar_series = pd.Series(list(self.calendar))
        open_dates = self.instruments["open_date"].to_dict()

        listing_ordinal: dict[str, int | None] = {}
        for code, open_date in open_dates.items():
            if open_date is None or pd.isna(open_date):
                listing_ordinal[code] = None
                continue
            listing_ordinal[code] = int(
                calendar_series.searchsorted(pd.Timestamp(open_date).date(), side="left")
            )

        result: list[int | None] = []
        for code, trade_date in zip(frame["code"], frame["trade_date"]):
            base = listing_ordinal.get(code)
            current = ordinals.get(pd.Timestamp(trade_date).date())
            result.append(None if base is None or current is None else current - base)
        return result

    def _report_price_limits(self, violations: pd.DataFrame) -> None:
        """把涨跌停越界明细转成问题记录。

        参数：
            violations: 已筛出的越界行，含 ``code``、``trade_date``、
                ``pct_change`` 与 ``limit_ratio`` 四列。

        返回：
            无返回值。
        """
        level = "ERROR" if self.config.price_limit_level == "error" else "WARNING"
        for row in violations.head(_SAMPLE_LIMIT).itertuples(index=False):
            self._add_issue(
                "DAILY_PRICE_LIMIT_VIOLATION",
                level,
                f"收盘涨跌幅超过板块涨跌停限制（共 {len(violations)} 处）",
                code=str(row.code),
                date=str(row.trade_date),
                expected=f"|涨跌幅| <= {row.limit_ratio:.0%}",
                actual=f"{row.pct_change:.2%}",
                evidence="按板块基础幅度判定，未按当前 ST 状态收紧",
                possible_causes="板块规则在该日与现行口径不同、存在未记录的股本变动，或行情本身有误",
                suggested_action="逐条核对；确需当作硬错误时使用 --price-limit-level error",
            )
