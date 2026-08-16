"""覆盖率与缺失区间校验：上市到退市之间是不是每个交易日都有数据。

关键口径：**停牌日不豁免缺失判定**。大 QMT 的 ``fill_data=True`` 会给停牌日也
写一行（``suspend_flag=1``），入库时又已经按上市日切掉了未上市的填充行，
所以存续期内某个交易日一行都没有，就是真的缺数据。
"""

from __future__ import annotations

import pandas as pd

from .base import _DailyCheckerState

#: 缺失区间最多列举多少条。
_SPAN_SAMPLE_LIMIT = 200

#: 越界数据最多列举多少条。
_SAMPLE_LIMIT = 20


class _CoverageRulesMixin(_DailyCheckerState):
    """统计覆盖率并定位连续缺失区间。"""

    def _check_coverage(self, connection) -> None:
        """执行覆盖率、缺失区间与生命周期越界的全部规则。

        参数：
            connection: 只读日线库连接，已建好 ``scoped_bars`` 临时视图。

        返回：
            无返回值；统计结果写入 ``coverage_by_date``、``coverage_by_symbol``
            与 ``missing_spans``，问题追加到 ``issues``。
        """
        self._prepare_expected_view(connection)
        self._collect_coverage(connection)
        self._collect_missing_spans(connection)
        self._check_lifecycle_bounds(connection)
        self._check_delisted_coverage(connection)

    def _prepare_expected_view(self, connection) -> None:
        """构造「理论上应当存在的 (证券, 交易日)」临时视图。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；建立 ``scoped_calendar``、``scoped_life`` 与
            ``expected_bars`` 三个临时视图。
        """
        calendar = pd.DataFrame({"trade_date": list(self.calendar)})
        connection.register("calendar_frame", calendar)
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW scoped_calendar AS "
            "SELECT CAST(trade_date AS DATE) AS trade_date, "
            "row_number() OVER (ORDER BY trade_date) AS ordinal FROM calendar_frame"
        )
        # open_date 缺失时取审计区间第一天为上市日回退值，口径与
        # ``quant.qmt_downloader.self_check`` 的 ``_build_lifecycle`` 保持一致，
        # 使两套自检对同一只证券算出同样的存续区间与覆盖率分母。
        # CREATE VIEW 语句不支持预备参数，回退日期只来自审计区间日历（非外部输入），
        # 拼接一个带引号的日期字面量是安全的。
        fallback_open_date = (
            f"DATE '{self.calendar[0].isoformat()}'" if self.calendar else "NULL"
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW scoped_life AS
            SELECT i.code, COALESCE(CAST(i.open_date AS DATE), {fallback_open_date}) AS open_date,
                   CAST(i.expire_date AS DATE) AS expire_date
            FROM instruments i
            JOIN (SELECT DISTINCT code FROM scoped_codes) s ON s.code = i.code
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TEMP VIEW expected_bars AS
            SELECT l.code, c.trade_date, c.ordinal
            FROM scoped_life l
            JOIN scoped_calendar c
              ON c.trade_date >= l.open_date
             AND (l.expire_date IS NULL OR c.trade_date <= l.expire_date)
            """
        )

    def _collect_coverage(self, connection) -> None:
        """按交易日与按证券统计应有、实有与覆盖率。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；结果写入 ``coverage_by_date`` 与 ``coverage_by_symbol``。
        """
        self.coverage_by_date = connection.execute(
            """
            SELECT e.trade_date,
                   CAST(count(*) AS BIGINT) AS expected,
                   CAST(count(b.code) AS BIGINT) AS actual,
                   CAST(count(b.code) AS DOUBLE) / nullif(count(*), 0) AS coverage
            FROM expected_bars e
            LEFT JOIN scoped_bars b ON b.code = e.code AND b.trade_date = e.trade_date
            GROUP BY e.trade_date ORDER BY e.trade_date
            """
        ).df()
        self.coverage_by_symbol = connection.execute(
            """
            SELECT e.code,
                   CAST(count(*) AS BIGINT) AS expected,
                   CAST(count(b.code) AS BIGINT) AS actual,
                   CAST(count(*) - count(b.code) AS BIGINT) AS missing,
                   CAST(count(b.code) AS DOUBLE) / nullif(count(*), 0) AS coverage
            FROM expected_bars e
            LEFT JOIN scoped_bars b ON b.code = e.code AND b.trade_date = e.trade_date
            GROUP BY e.code ORDER BY e.code
            """
        ).df()

        threshold = self.config.coverage_error_threshold
        low = self.coverage_by_date[self.coverage_by_date["coverage"] < threshold]
        for row in low.head(_SAMPLE_LIMIT).itertuples(index=False):
            self._add_issue(
                "DAILY_COVERAGE_BELOW_THRESHOLD",
                "ERROR",
                f"单个交易日覆盖率过低（区间内共 {len(low)} 天）",
                date=str(row.trade_date),
                expected=f"覆盖率不低于 {threshold:.0%}",
                actual=f"{row.actual}/{row.expected} = {row.coverage:.2%}",
                possible_causes="该交易日的分区没有入库，或源分区本身不完整",
                suggested_action="确认源目录该日分区完整后重建对应月份分片",
            )
        empty = self.coverage_by_date[self.coverage_by_date["actual"] == 0]
        for row in empty.head(_SAMPLE_LIMIT).itertuples(index=False):
            self._add_issue(
                "DAILY_CALENDAR_DATE_EMPTY",
                "ERROR" if self.calendar_is_authoritative else "WARNING",
                f"交易日历上的交易日在库中没有任何行情（共 {len(empty)} 天）",
                date=str(row.trade_date),
                expected=f"该交易日应有 {row.expected} 只证券的行情",
                actual="0 行",
                possible_causes="该日源分区缺失，或下载器尚未覆盖该日",
                suggested_action="补下载该交易日后重新同步",
            )

    def _collect_missing_spans(self, connection) -> None:
        """把逐日缺失合并为连续缺失区间。

        用「交易日序号减去按证券分组的行号」这一常见技巧分组：同一段连续缺失里
        该差值恒定。序号取自交易日历而非自然日，因此跨节假日不会被误切。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；结果写入 ``missing_spans``。
        """
        self.missing_spans = connection.execute(
            """
            WITH missing AS (
                SELECT e.code, e.trade_date, e.ordinal
                FROM expected_bars e
                LEFT JOIN scoped_bars b ON b.code = e.code AND b.trade_date = e.trade_date
                WHERE b.code IS NULL
            ),
            grouped AS (
                SELECT code, trade_date, ordinal,
                       ordinal - row_number() OVER (PARTITION BY code ORDER BY ordinal) AS island
                FROM missing
            )
            SELECT code, min(trade_date) AS start_date, max(trade_date) AS end_date,
                   CAST(count(*) AS BIGINT) AS days
            FROM grouped GROUP BY code, island
            ORDER BY days DESC, code, start_date
            """
        ).df()
        for row in self.missing_spans.head(_SPAN_SAMPLE_LIMIT).itertuples(index=False):
            self._add_issue(
                "DAILY_MISSING_SPAN",
                "ERROR",
                f"上市至退市区间内存在连续缺失（全库共 {len(self.missing_spans)} 段）",
                code=str(row.code),
                start_date=str(row.start_date),
                end_date=str(row.end_date),
                expected="存续期内每个交易日都应有一行（停牌日也会有 suspend_flag=1 的行）",
                actual=f"连续缺失 {row.days} 个交易日",
                possible_causes="源分区缺失、下载时该证券取数失败，或入库时被生命周期过滤误伤",
                suggested_action="先用 quant-qmt-self-check 确认源侧是否也缺，再决定补下载还是重建分片",
            )

    def _check_lifecycle_bounds(self, connection) -> None:
        """检查是否存在越过上市或退市边界的行情。

        入库时已经按生命周期过滤，正常情况下这里应当一条都查不出；查出来说明
        过滤依据的快照与当前 ``instruments`` 表不一致。上市前 ``suspend_flag=1``
        的停牌占位行不算异常：这是 QMT 对尚未上市证券填充的历史占位数据，不是
        真实行情归属错误，口径与 ``quant.qmt_downloader.self_check`` 的
        ``DATA_BEFORE_LISTING`` 判定保持一致，避免同一份数据在两套自检里一边报错
        一边不报错。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        rows = connection.execute(
            """
            SELECT b.code, b.trade_date, l.open_date, l.expire_date,
                   CASE WHEN b.trade_date < l.open_date THEN 'before' ELSE 'after' END AS side
            FROM scoped_bars b
            JOIN scoped_life l ON l.code = b.code
            WHERE (b.trade_date < l.open_date AND COALESCE(b.suspend_flag, 0) != 1)
               OR (l.expire_date IS NOT NULL AND b.trade_date > l.expire_date)
            ORDER BY b.code, b.trade_date LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for code, trade_date, open_date, expire_date, side in rows:
            self._add_issue(
                "DAILY_DATA_BEFORE_LISTING" if side == "before" else "DAILY_DATA_AFTER_DELISTING",
                "ERROR",
                "行情落在证券存续期之外",
                code=str(code),
                date=str(trade_date),
                expected=f"上市日 {open_date} 至退市日 {expire_date} 之间",
                actual=str(trade_date),
                possible_causes="入库时依据的 instrument_info 快照与当前表不一致",
                suggested_action="用 quant-build-daily-store --rebuild-all 按最新快照重建",
            )
        unknown = connection.execute(
            """
            SELECT DISTINCT b.code FROM scoped_bars b
            LEFT JOIN instruments i ON i.code = b.code
            WHERE i.code IS NULL ORDER BY b.code LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for (code,) in unknown:
            self._add_issue(
                "DAILY_LIFECYCLE_UNKNOWN",
                "WARNING",
                "库中有行情但 instruments 里没有该证券，无法判定其存续期",
                code=str(code),
                expected="instruments 覆盖 bars_1d 中的全部代码",
                actual="缺少该代码",
                possible_causes="证券已从 QMT 证券池中移除，而历史分片仍保留其行情",
                suggested_action="重新导出 instrument_info，或接受该证券不参与覆盖率统计",
            )

    def _check_delisted_coverage(self, connection) -> None:
        """检查窗口内退市的证券是否确实有行情。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        rows = connection.execute(
            """
            SELECT l.code, l.expire_date FROM scoped_life l
            LEFT JOIN (SELECT DISTINCT code FROM scoped_bars) b ON b.code = l.code
            WHERE l.expire_date IS NOT NULL
              AND l.expire_date BETWEEN ? AND ?
              AND b.code IS NULL
            ORDER BY l.code LIMIT ?
            """,
            [self.start_date, self.end_date, _SAMPLE_LIMIT],
        ).fetchall()
        for code, expire_date in rows:
            self._add_issue(
                "DAILY_DELISTED_SYMBOL_MISSING",
                "WARNING",
                "证券在审计区间内退市，但库中一根 K 线都没有",
                code=str(code),
                date=str(expire_date),
                expected="退市前应有历史行情",
                actual="0 行",
                evidence="缺少退市证券会让回测产生幸存者偏差",
                possible_causes="下载器在该证券退市后才建库，QMT 已不再提供其历史行情",
                suggested_action="如需无偏回测，需要从其它数据源补齐退市证券历史",
            )
