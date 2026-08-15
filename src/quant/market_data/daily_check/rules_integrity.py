"""行级完整性校验：主键、空值、价格关系与字段取值范围。

全部规则都是单条 DuckDB 聚合，不把行情拉进 pandas。
"""

from __future__ import annotations

from .base import _DailyCheckerState

#: 单条规则最多列举多少个样例，避免报告被刷屏。
_SAMPLE_LIMIT = 10


class _IntegrityRulesMixin(_DailyCheckerState):
    """校验日线行本身是否自洽。"""

    def _check_integrity(self, connection) -> None:
        """执行全部行级完整性规则。

        参数：
            connection: 只读日线库连接，已建好 ``scoped_bars`` 临时视图。

        返回：
            无返回值；发现的问题追加到 ``issues``。
        """
        self._check_duplicate_keys(connection)
        self._check_null_fields(connection)
        self._check_price_sanity(connection)
        self._check_quantity_sanity(connection)
        self._check_suspend_flag(connection)
        self._check_code_format(connection)

    def _check_duplicate_keys(self, connection) -> None:
        """检查 ``(code, trade_date)`` 是否唯一。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        rows = connection.execute(
            """
            SELECT code, trade_date, count(*) AS hits
            FROM scoped_bars GROUP BY 1, 2 HAVING count(*) > 1
            ORDER BY 1, 2 LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        total = connection.execute(
            "SELECT count(*) FROM (SELECT 1 FROM scoped_bars GROUP BY code, trade_date "
            "HAVING count(*) > 1)"
        ).fetchone()[0]
        for code, trade_date, hits in rows:
            self._add_issue(
                "DAILY_DUPLICATE_KEY",
                "ERROR",
                f"同一证券在同一交易日出现多行（共 {total} 组重复）",
                code=str(code),
                date=str(trade_date),
                expected="每个 (code, trade_date) 只有一行",
                actual=f"{hits} 行",
                possible_causes="月度分片重写时源分区里存在重复主键，或分片文件残留",
                suggested_action="用 quant-build-daily-store --rebuild-all 重建该月分片",
            )

    def _check_null_fields(self, connection) -> None:
        """检查关键字段是否存在空值。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        total = connection.execute(
            "SELECT count(*) FROM scoped_bars WHERE code IS NULL OR trade_date IS NULL "
            "OR close IS NULL"
        ).fetchone()[0]
        if not total:
            return
        rows = connection.execute(
            """
            SELECT code, trade_date FROM scoped_bars
            WHERE code IS NULL OR trade_date IS NULL OR close IS NULL
            ORDER BY code, trade_date LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for code, trade_date in rows:
            self._add_issue(
                "DAILY_NULL_FIELD",
                "ERROR",
                f"关键字段为空（共 {total} 行）",
                code=str(code),
                date=str(trade_date),
                field="code/trade_date/close",
                expected="三个字段都非空",
                actual="存在空值",
                possible_causes="源 CSV 该行字段缺失，或类型转换失败被置为空",
                suggested_action="检查源分区 data.csv 对应行后重建该月分片",
            )

    def _check_price_sanity(self, connection) -> None:
        """检查价格是否为正以及最高最低价关系是否成立。

        停牌行的开高低收全等于前收，同样应当为正，因此不做豁免。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        self._report_rule(
            connection,
            issue_code="DAILY_NONPOSITIVE_PRICE",
            level="ERROR",
            condition="least(open, high, low, close) <= 0",
            message="存在非正价格",
            expected="开高低收全部大于 0",
            actual_expression="'ohlc=' || open || '/' || high || '/' || low || '/' || close",
            possible_causes="源数据缺失被填成 0，或类型转换出错",
            suggested_action="核对源分区该行后重建该月分片",
        )
        self._report_rule(
            connection,
            issue_code="DAILY_OHLC_VIOLATION",
            level="ERROR",
            condition=(
                "high < greatest(open, close, low) OR low > least(open, close, high)"
            ),
            message="最高价或最低价与开收价矛盾",
            expected="high >= max(open, close, low) 且 low <= min(open, close, high)",
            actual_expression="'ohlc=' || open || '/' || high || '/' || low || '/' || close",
            possible_causes="源行情本身有误；下载器的行级修正未覆盖到该行",
            suggested_action="对照 quant-qmt-self-check 的 line_correct 报告确认源数据",
        )

    def _check_quantity_sanity(self, connection) -> None:
        """检查成交量与成交额是否为负。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        self._report_rule(
            connection,
            issue_code="DAILY_NEGATIVE_QUANTITY",
            level="ERROR",
            condition="volume < 0 OR amount < 0",
            message="成交量或成交额为负",
            expected="volume >= 0 且 amount >= 0",
            actual_expression="'volume=' || volume || ' amount=' || amount",
            possible_causes="源数据溢出或类型转换出错",
            suggested_action="核对源分区该行后重建该月分片",
        )

    def _check_suspend_flag(self, connection) -> None:
        """检查停牌标记是否只取 0 或 1。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        self._report_rule(
            connection,
            issue_code="DAILY_INVALID_SUSPEND_FLAG",
            level="ERROR",
            condition="suspend_flag NOT IN (0, 1)",
            message="停牌标记取值非法",
            expected="suspend_flag 只能是 0 或 1",
            actual_expression="'suspend_flag=' || suspend_flag",
            possible_causes="QMT 返回了预期之外的 suspendFlag 取值",
            suggested_action="确认源数据后决定是否需要扩展标记口径",
        )

    def _check_code_format(self, connection) -> None:
        """检查证券代码是否带交易所后缀。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值。
        """
        self._report_rule(
            connection,
            issue_code="DAILY_CODE_FORMAT_INVALID",
            level="ERROR",
            condition="code IS NOT NULL AND position('.' IN code) = 0",
            message="证券代码缺少交易所后缀",
            expected="形如 000001.SZ",
            actual_expression="code",
            possible_causes="源分区里混入了未规范化的代码",
            suggested_action="核对源分区后重建该月分片",
        )

    def _report_rule(
        self,
        connection,
        *,
        issue_code: str,
        level: str,
        condition: str,
        message: str,
        expected: str,
        actual_expression: str,
        possible_causes: str,
        suggested_action: str,
    ) -> None:
        """按同一套模板执行一条「筛出违规行」的规则。

        参数：
            connection: 只读日线库连接。
            issue_code: 问题编号。
            level: 严重级别。
            condition: 判定违规的 SQL 布尔表达式，可引用 ``scoped_bars`` 的列。
            message: 问题描述模板，会自动附上违规总行数。
            expected: 理论值说明。
            actual_expression: 生成实际值文本的 SQL 表达式。
            possible_causes: 可能原因说明。
            suggested_action: 处理建议。

        返回：
            无返回值。
        """
        total = connection.execute(
            f"SELECT count(*) FROM scoped_bars WHERE {condition}"
        ).fetchone()[0]
        if not total:
            return
        rows = connection.execute(
            f"""
            SELECT code, trade_date, {actual_expression} AS actual_text
            FROM scoped_bars WHERE {condition}
            ORDER BY code, trade_date LIMIT ?
            """,
            [_SAMPLE_LIMIT],
        ).fetchall()
        for code, trade_date, actual_text in rows:
            self._add_issue(
                issue_code,
                level,
                f"{message}（共 {total} 行）",
                code=str(code),
                date=str(trade_date),
                expected=expected,
                actual=str(actual_text),
                possible_causes=possible_causes,
                suggested_action=suggested_action,
            )
