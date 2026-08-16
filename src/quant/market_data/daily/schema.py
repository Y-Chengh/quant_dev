"""QMT 日线库的列契约、建表语句以及证券静态属性推导。

本模块是日线库物理结构的唯一定义处：``bars_1d`` 月度 Parquet 的列与类型、
目录库里各张表的 DDL，以及由证券代码和名称推导板块、风险警示状态的规则。
入库层和读取层都只从这里取常量，避免两侧对同一份结构各写一遍。
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

#: 目录库结构版本；改变 Parquet 列或表结构时递增，入库层据此提示重建。
SCHEMA_VERSION = "1"

#: ``bars_1d`` Parquet 的规范列顺序，与 ``BAR_COLUMN_TYPES`` 一一对应。
BAR_COLUMNS: tuple[str, ...] = (
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

#: ``bars_1d`` 各列在 Parquet 中的 DuckDB 类型。
BAR_COLUMN_TYPES: dict[str, str] = {
    "code": "VARCHAR",
    "trade_date": "DATE",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "pre_close": "DOUBLE",
    "volume": "BIGINT",
    "amount": "DOUBLE",
    "suspend_flag": "TINYINT",
}

#: 大 QMT 用于表达「没有这个日期」的哨兵值，全部归一化为 ``None``。
#: 与 ``quant.qmt_downloader.self_check.utils._LIFECYCLE_SENTINELS`` 保持同一份
#: 取值列表；因分层约束（``quant.qmt_downloader`` 不得导入 ``quant`` 的其它子包）
#: 无法共用同一个常量定义，两处修改哨兵集合时必须同步。
LIFECYCLE_SENTINELS: frozenset[str] = frozenset(
    {
        "",
        "nan",
        "nat",
        "none",
        "null",
        "0",
        "99999999",
        "19700101",
        "19700102",
        "19700103",
        "19700104",
        "19700105",
        "19700106",
        "19700427",
        "19700428",
    }
)

#: 板块标识，用于涨跌停口径与证券池过滤。
BOARD_MAIN = "main"
BOARD_STAR = "star"
BOARD_CHINEXT = "chinext"
BOARD_BSE = "bse"
BOARD_UNKNOWN = "unknown"

#: 目录库中除 ``bars_1d`` 视图外的全部表；按依赖顺序建表。
TABLE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS instruments(
        code VARCHAR PRIMARY KEY,
        instrument_name VARCHAR,
        open_date DATE,
        expire_date DATE,
        is_trading BOOLEAN,
        instrument_status VARCHAR,
        board VARCHAR,
        is_st BOOLEAN
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS trading_calendar(
        trade_date DATE PRIMARY KEY,
        calendar_symbol VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corporate_actions(
        code VARCHAR,
        ex_date DATE,
        cash_dividend_per_share DOUBLE,
        bonus_share_per_share DOUBLE,
        capitalization_per_share DOUBLE,
        rights_issue_per_share DOUBLE,
        rights_issue_price DOUBLE,
        share_reform_flag DOUBLE,
        adjustment_factor DOUBLE,
        PRIMARY KEY (code, ex_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_state(
        dataset VARCHAR,
        partition_key VARCHAR,
        marker_mtime_ns HUGEINT,
        marker_size BIGINT,
        data_mtime_ns HUGEINT,
        data_size BIGINT,
        content_sha256 VARCHAR,
        identity_sha256 VARCHAR,
        source_rows BIGINT,
        completed_at VARCHAR,
        target_shard VARCHAR,
        ingested_at TIMESTAMP,
        PRIMARY KEY (dataset, partition_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_runs(
        run_id VARCHAR PRIMARY KEY,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        mode VARCHAR,
        source_root VARCHAR,
        status VARCHAR,
        scanned BIGINT,
        dirty BIGINT,
        removed BIGINT,
        rewritten_shards BIGINT,
        rows_after BIGINT,
        message VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_metadata(
        key VARCHAR PRIMARY KEY,
        value VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monthly_inventory(
        year INTEGER,
        month INTEGER,
        rows BIGINT,
        symbols BIGINT,
        trading_days BIGINT,
        min_date DATE,
        max_date DATE,
        PRIMARY KEY (year, month)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbols(
        code VARCHAR PRIMARY KEY,
        first_date DATE,
        last_date DATE,
        bar_count BIGINT,
        suspended_days BIGINT
    )
    """,
)


def sql_path(path) -> str:
    """把文件系统路径转换为可直接嵌入 SQL 字符串字面量的文本。

    参数：
        path: 任意 ``Path`` 或字符串路径；Windows 反斜杠会转成正斜杠，
            单引号会按 SQL 规则转义为两个单引号。

    返回：
        不含首尾引号的安全路径文本。
    """
    return str(path).replace("\\", "/").replace("'", "''")


def bars_parquet_glob(daily_root) -> str:
    """拼出 ``bars_1d`` 月度分片的 hive 通配路径。

    参数：
        daily_root: 日线库数据根目录，其下为 ``bars_1d/year=*/month=*``。

    返回：
        可直接交给 ``read_parquet`` 的通配字符串。
    """
    return sql_path(Path(daily_root) / "bars_1d" / "year=*" / "month=*" / "bars.parquet")


def create_view_statement(daily_root) -> str:
    """生成重建 ``bars_1d`` 视图的 SQL。

    参数：
        daily_root: 日线库数据根目录。

    返回：
        ``CREATE OR REPLACE VIEW`` 语句；月度分片全部缺失时该视图不可查询，
        因此调用方应在没有任何分片时跳过建视图。
    """
    return (
        f"CREATE OR REPLACE VIEW bars_1d AS SELECT * FROM read_parquet('{bars_parquet_glob(daily_root)}', "
        "hive_partitioning=true, union_by_name=true)"
    )


def apply_schema(connection, daily_root, *, create_view: bool) -> None:
    """在目录库上建齐所有表，并按需重建 ``bars_1d`` 视图。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        daily_root: 日线库数据根目录，用于拼视图的 Parquet 通配路径。
        create_view: 是否重建 ``bars_1d`` 视图。首次入库、还没有任何月度分片时
            必须传 ``False``，否则 DuckDB 会因为通配不到文件而报错。

    返回：
        无返回值；语句全部是幂等的，可重复执行。
    """
    for statement in TABLE_STATEMENTS:
        connection.execute(statement)
    if create_view:
        connection.execute(create_view_statement(daily_root))


def normalize_lifecycle_date(value) -> date | None:
    """把 instrument_info 里的上市或退市日期归一化为真实日期。

    大 QMT 会用 ``0``、``99999999`` 以及 1970 年初的若干个日期表示「没有这个
    日期」，实测快照里同时出现过 ``19700427`` 和 ``19700428``。这些值必须变成
    空值，否则生命周期过滤会把整只证券的行情全部切掉，或者误报「退市后仍有数据」。

    参数：
        value: 原始取值，可能是八位日期字符串、数字或空值。

    返回：
        解析成功的 ``date``；属于哨兵值或无法解析时返回 ``None``。
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in LIFECYCLE_SENTINELS:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 8:
        return None
    if digits[:8] in LIFECYCLE_SENTINELS:
        return None
    try:
        return datetime.strptime(digits[:8], "%Y%m%d").date()
    except ValueError:
        return None


def classify_board(code: str) -> str:
    """根据证券代码前缀判断所属板块。

    板块决定涨跌停幅度，也用于证券池过滤。

    参数：
        code: 带交易所后缀的证券代码，例如 ``688981.SH``；大小写不敏感，
            允许传入不带后缀的六位裸代码。

    返回：
        ``main``/``star``/``chinext``/``bse`` 之一；无法归类时返回 ``unknown``。
    """
    digits = str(code).strip().upper().split(".", 1)[0]
    if not digits.isdigit():
        return BOARD_UNKNOWN
    if digits.startswith(("688", "689")):
        return BOARD_STAR
    if digits.startswith(("300", "301")):
        return BOARD_CHINEXT
    if digits.startswith(("430", "920", "83", "87", "88")):
        return BOARD_BSE
    if digits.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return BOARD_MAIN
    return BOARD_UNKNOWN


def is_risk_warned(instrument_name) -> bool:
    """根据证券名称判断是否处于风险警示或退市整理状态。

    只能反映 ``instrument_info`` 快照的**当前**名称，无法还原历史某一天的 ST
    状态；调用方在做历史涨跌停校验时必须知道这个局限。

    参数：
        instrument_name: 证券简称，例如 ``*ST 中安``；空值按非风险警示处理。

    返回：
        名称中含 ``ST`` 或 ``退`` 时返回 ``True``。
    """
    if instrument_name is None:
        return False
    text = str(instrument_name).strip().upper()
    if not text or text in {"NAN", "NONE"}:
        return False
    return "ST" in text or "退" in text
