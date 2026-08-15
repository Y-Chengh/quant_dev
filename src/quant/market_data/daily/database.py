"""日线库的只读仓储层：全部 DuckDB SQL 集中在这里。

与现有 ``market_data.database.MarketDatabase`` 保持同样的形状：连接是每次调用
新建的上下文管理器，一律以 ``read_only=True`` 打开，因此可以和其它读进程共存，
也不会在进程内缓存住连接而挡住随后的写入。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from quant.config import default_qmt_daily_database

from ..codes import normalize_security_code

#: ``bars_1d`` 对外返回的固定列顺序。
BAR_QUERY_COLUMNS = (
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "suspend_flag",
)


class DailyMarketDatabase:
    """只读访问 QMT 日线库的 DuckDB 目录文件。"""

    def __init__(self, path: str | Path | None = None) -> None:
        """绑定日线库文件。

        参数：
            path: ``qmt_daily.duckdb`` 文件路径；``None`` 表示按
                ``QMT_DAILY_DB_PATH`` 环境变量解析，未设置时使用内置回退路径。

        返回：
            无返回值；连接在 ``connect()`` 中按需建立。
        """
        self.path = Path(path) if path is not None else default_qmt_daily_database()

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """以只读方式打开一个短生命周期的 DuckDB 连接。

        返回：
            上下文管理器；退出时必定关闭连接，避免在进程内长期持有句柄而与
            后续的增量同步争抢文件锁。
        """
        if not self.path.is_file():
            raise FileNotFoundError(
                f"日线库不存在: {self.path}；请先运行 quant-build-daily-store"
            )
        connection = duckdb.connect(str(self.path), read_only=True)
        try:
            connection.execute("SET threads = 4")
            yield connection
        finally:
            connection.close()

    def metadata(self) -> dict:
        """汇总日线库的时间范围、行数与证券数。

        返回：
            含 ``first_time``、``last_time``、``first_date``、``last_date``、
            ``rows``、``symbols``、``adjust_anchor_date`` 与 ``schema_version``
            的字典。``first_time``/``last_time`` 是为了与 5 分钟库的元数据形状
            对齐，取值为交易日零点的 ISO 文本；库里没有行情时两者为空串。
        """
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT min(min_date), max(max_date), CAST(COALESCE(sum(rows), 0) AS BIGINT)
                FROM monthly_inventory
                """
            ).fetchone()
            symbols = connection.execute("SELECT count(*) FROM symbols").fetchone()[0]
            stored = dict(
                connection.execute("SELECT key, value FROM dataset_metadata").fetchall()
            )
        first, last, rows = row if row else (None, None, 0)
        return {
            "first_date": first,
            "last_date": last,
            "first_time": "" if first is None else f"{first} 00:00:00",
            "last_time": "" if last is None else f"{last} 00:00:00",
            "rows": int(rows or 0),
            "symbols": int(symbols or 0),
            "adjust_anchor_date": str(stored.get("adjust_anchor_date", "")),
            "schema_version": str(stored.get("schema_version", "")),
        }

    def daily_bars(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        *,
        include_suspended: bool = False,
    ) -> pd.DataFrame:
        """按证券与日期区间读取原始不复权日线。

        参数：
            codes: 证券代码序列；空串会被忽略，六位沪深裸代码自动补全后缀。
                传空序列表示不限证券，返回区间内的全市场行情。
            start: 起始交易日，包含该日。
            end: 结束交易日，包含该日。
            include_suspended: 是否保留 ``suspend_flag`` 非零的停牌行。缺省
                ``False``，因为停牌行开高低收全等于前收、成交量为零，留在因子
                输入里会造出成片的假零收益。

        返回：
            按证券代码与交易日升序排列的日线表，列为 ``BAR_QUERY_COLUMNS``；
            库中没有任何分片时返回具有标准列的空表。
        """
        normalized = _normalize_codes(codes)
        clauses = ["trade_date BETWEEN ? AND ?"]
        params: list = [start, end]
        if normalized:
            clauses.append("code IN ({0})".format(",".join("?" for _ in normalized)))
            params.extend(normalized)
        if not include_suspended:
            clauses.append("COALESCE(suspend_flag, 0) = 0")
        with self.connect() as connection:
            if not _has_bars_view(connection):
                return pd.DataFrame(columns=list(BAR_QUERY_COLUMNS))
            return connection.execute(
                """
                SELECT {columns}
                FROM bars_1d
                WHERE {where}
                ORDER BY code, trade_date
                """.format(
                    columns=", ".join(BAR_QUERY_COLUMNS),
                    where=" AND ".join(clauses),
                ),
                params,
            ).df()

    def corporate_actions(
        self,
        codes: Sequence[str] = (),
        end: date | None = None,
    ) -> pd.DataFrame:
        """读取除权送转记录。

        复权因子是从证券上市起累乘的，因此这里刻意**不**按起始日期过滤：
        少读一条早年的除权记录就会让整段历史的复权系数偏掉。

        参数：
            codes: 证券代码序列；空序列表示读取全部证券。
            end: 只取除权日不晚于该日的记录；``None`` 表示不限上界。

        返回：
            按证券代码与除权日升序排列的除权表；无记录时返回空表。
        """
        normalized = _normalize_codes(codes)
        clauses = ["1 = 1"]
        params: list = []
        if normalized:
            clauses.append("code IN ({0})".format(",".join("?" for _ in normalized)))
            params.extend(normalized)
        if end is not None:
            clauses.append("ex_date <= ?")
            params.append(end)
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT code, ex_date, cash_dividend_per_share, bonus_share_per_share,
                       capitalization_per_share, rights_issue_per_share, rights_issue_price,
                       share_reform_flag, adjustment_factor
                FROM corporate_actions
                WHERE {0}
                ORDER BY code, ex_date
                """.format(" AND ".join(clauses)),
                params,
            ).df()

    def trading_dates(self, start: date | None = None, end: date | None = None) -> list[date]:
        """读取交易日历。

        参数：
            start: 起始日期，包含该日；``None`` 表示不限下界。
            end: 结束日期，包含该日；``None`` 表示不限上界。

        返回：
            升序排列的交易日列表；日历表为空时返回空列表。
        """
        clauses = ["1 = 1"]
        params: list = []
        if start is not None:
            clauses.append("trade_date >= ?")
            params.append(start)
        if end is not None:
            clauses.append("trade_date <= ?")
            params.append(end)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT trade_date FROM trading_calendar WHERE {0} ORDER BY trade_date".format(
                    " AND ".join(clauses)
                ),
                params,
            ).fetchall()
        return [row[0] for row in rows]

    def instruments(self, codes: Sequence[str] = ()) -> pd.DataFrame:
        """读取证券静态信息。

        参数：
            codes: 证券代码序列；空序列表示读取全部证券。

        返回：
            含代码、简称、上市日、退市日、交易状态、板块与风险警示标记的表。
        """
        normalized = _normalize_codes(codes)
        clauses = ["1 = 1"]
        params: list = []
        if normalized:
            clauses.append("code IN ({0})".format(",".join("?" for _ in normalized)))
            params.extend(normalized)
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT code, instrument_name, open_date, expire_date,
                       is_trading, instrument_status, board, is_st
                FROM instruments
                WHERE {0}
                ORDER BY code
                """.format(" AND ".join(clauses)),
                params,
            ).df()

    def symbols(self, query: str = "", limit: int = 20) -> list[str]:
        """按代码子串搜索库中有行情的证券。

        与 5 分钟库的 ``search_symbols`` 语义一致，只反映「有没有行情」，
        **不是**无幸存者偏差的证券池；需要无偏池请用 ``universe`` 模块。

        参数：
            query: 代码子串，大小写不敏感；空串表示不过滤。
            limit: 返回条数上限。

        返回：
            按代码升序排列的证券代码列表。
        """
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT code FROM symbols WHERE code LIKE ? ORDER BY code LIMIT ?",
                [f"%{query.upper().strip()}%", limit],
            ).fetchall()
        return [row[0] for row in rows]


def _normalize_codes(codes: Sequence[str]) -> list[str]:
    """去空、去重并补全交易所后缀。

    参数：
        codes: 原始证券代码序列，允许为空或含空串。

    返回：
        保持首次出现顺序的规范化代码列表。
    """
    if not codes:
        return []
    return list(
        dict.fromkeys(
            normalize_security_code(code) for code in codes if str(code).strip()
        )
    )


def _has_bars_view(connection) -> bool:
    """判断目录库里是否已经建好 ``bars_1d`` 视图。

    首次入库前或所有月度分片都被清空时视图会被丢弃，此时查询应当返回空表而不是
    抛出「找不到表」的原始异常。

    参数：
        connection: 已打开的 DuckDB 连接。

    返回：
        视图存在时返回 ``True``。
    """
    row = connection.execute(
        "SELECT count(*) FROM duckdb_views() WHERE view_name = 'bars_1d'"
    ).fetchone()
    return bool(row and row[0])
