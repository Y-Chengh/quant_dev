"""证券生命周期、交易日历与除权送转三张辅助表的装载与增量刷新。

这三张表都很小（分别约 5.2 千、6.4 千和 5.4 万行），因此直接落成 DuckDB 实体表，
按主键做点更新，而不像日线那样走月度 Parquet 分片。

``instruments`` 必须先于 ``bars_1d`` 入库：日线的生命周期过滤依赖它，缺了它就会
把大量「还没上市」的填充行当成真实行情写进库里。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..schema import classify_board, is_risk_warned, normalize_lifecycle_date, sql_path
from .source import (
    DATASET_ACTIONS,
    DATASET_CALENDAR,
    DATASET_INSTRUMENTS,
)

#: 除权送转 CSV 的列类型，显式声明以绕开「只有表头」分区推断不出类型的问题。
_ACTION_CSV_COLUMNS = {
    "code": "VARCHAR",
    "ex_date": "VARCHAR",
    "cash_dividend_per_share": "DOUBLE",
    "bonus_share_per_share": "DOUBLE",
    "capitalization_per_share": "DOUBLE",
    "rights_issue_per_share": "DOUBLE",
    "rights_issue_price": "DOUBLE",
    "share_reform_flag": "DOUBLE",
    "adjustment_factor": "DOUBLE",
}

#: 脏分区超过该数量时改用整目录通配，避免拼出几十万字符的 SQL；同一个阈值也是
#: ``delete_corporate_actions`` 分批下发 ``DELETE`` 时每批的日期个数。
_GLOB_THRESHOLD = 200


class InstrumentSnapshotMissing(RuntimeError):
    """``instrument_info`` 快照缺失或无法解析时抛出。

    这是硬错误：没有生命周期就无法过滤未上市填充行，宁可整个入库失败，
    也不能退化成不过滤地全量写入。
    """


def load_lifecycle_frame(source_root: Path | str) -> pd.DataFrame:
    """读取并归一化证券生命周期快照。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        含 ``code``、``instrument_name``、``open_date``、``expire_date``、
        ``is_trading``、``instrument_status``、``board``、``is_st`` 八列的表。
        ``open_date``/``expire_date`` 已把 QMT 的无日期哨兵归一化为 ``NaT``。
        快照缺失、为空或缺列时抛出 ``InstrumentSnapshotMissing``。
    """
    path = Path(source_root) / DATASET_INSTRUMENTS / "snapshot=latest" / "data.csv"
    if not path.is_file():
        raise InstrumentSnapshotMissing(
            f"证券生命周期快照缺失: {path}；请先用下载器产出 instrument_info"
        )
    try:
        raw = pd.read_csv(str(path), encoding="utf-8-sig", dtype=str)
    except (OSError, ValueError) as error:
        raise InstrumentSnapshotMissing(f"证券生命周期快照无法解析: {path}: {error}")
    if raw.empty or "code" not in raw.columns:
        raise InstrumentSnapshotMissing(f"证券生命周期快照为空或缺少 code 列: {path}")

    frame = pd.DataFrame()
    frame["code"] = raw["code"].astype(str).str.strip().str.upper()
    frame["instrument_name"] = _optional_text(raw, "instrument_name")
    frame["open_date"] = _lifecycle_column(raw, "open_date")
    frame["expire_date"] = _lifecycle_column(raw, "expire_date")
    frame["is_trading"] = _optional_bool(raw, "is_trading")
    frame["instrument_status"] = _optional_text(raw, "instrument_status")
    frame["board"] = frame["code"].map(classify_board)
    frame["is_st"] = frame["instrument_name"].map(is_risk_warned)
    frame = frame[frame["code"].str.len() > 0]
    return frame.drop_duplicates("code").sort_values("code").reset_index(drop=True)


def _optional_text(raw: pd.DataFrame, column: str) -> pd.Series:
    """取出一列文本，缺列时返回等长空串列。

    参数：
        raw: 原始 CSV 数据表。
        column: 目标列名。

    返回：
        去除首尾空白的字符串 ``Series``；原值为空时是空串。
    """
    if column not in raw.columns:
        return pd.Series([""] * len(raw), index=raw.index, dtype=object)
    return raw[column].fillna("").astype(str).str.strip()


def _optional_bool(raw: pd.DataFrame, column: str) -> pd.Series:
    """把 QMT 的布尔文本列转换为可空布尔列。

    实测 ``is_trading`` 整列为空，因此必须容忍全空。

    参数：
        raw: 原始 CSV 数据表。
        column: 目标列名。

    返回：
        ``object`` 类型的 ``Series``，元素为 ``True``/``False``/``None``。
    """
    if column not in raw.columns:
        return pd.Series([None] * len(raw), index=raw.index, dtype=object)
    truthy = {"1", "1.0", "true", "yes", "t"}
    falsy = {"0", "0.0", "false", "no", "f"}

    def convert(value: object) -> bool | None:
        """把单个取值映射为布尔或空值。

        参数：
            value: CSV 中的原始取值，可能是 ``1``/``0``/``true``/``false``
                之类的文本，也可能是空值。

        返回：
            识别为真返回 ``True``，识别为假返回 ``False``，无法识别返回 ``None``。
        """
        text = "" if value is None else str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return None

    return raw[column].map(convert)


def _lifecycle_column(raw: pd.DataFrame, column: str) -> pd.Series:
    """把上市或退市日期列归一化为 ``datetime64`` 列。

    参数：
        raw: 原始 CSV 数据表。
        column: 目标列名，``open_date`` 或 ``expire_date``。

    返回：
        ``datetime64[ns]`` 类型的 ``Series``；哨兵与非法值为 ``NaT``。
    """
    if column not in raw.columns:
        return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    return pd.to_datetime(raw[column].map(normalize_lifecycle_date), errors="coerce")


def refresh_instruments(connection, lifecycle: pd.DataFrame) -> int:
    """用生命周期快照整表重建 ``instruments``。

    参数：
        connection: 以读写方式打开的 DuckDB 连接，调用方负责事务边界。
        lifecycle: ``load_lifecycle_frame`` 的返回值。

    返回：
        写入的行数。
    """
    connection.register("lifecycle_snapshot", lifecycle)
    try:
        connection.execute("DELETE FROM instruments")
        connection.execute(
            """
            INSERT INTO instruments(
                code, instrument_name, open_date, expire_date,
                is_trading, instrument_status, board, is_st)
            SELECT code, instrument_name,
                   CAST(open_date AS DATE), CAST(expire_date AS DATE),
                   CAST(is_trading AS BOOLEAN), instrument_status, board, CAST(is_st AS BOOLEAN)
            FROM lifecycle_snapshot
            """
        )
    finally:
        connection.unregister("lifecycle_snapshot")
    return len(lifecycle)


def refresh_trading_calendar(connection, source_root: Path | str) -> int:
    """把交易日历快照并入 ``trading_calendar``。

    优先读下载器写的 ``trading_calendar/snapshot=latest/data.csv``（两列），
    回退到 ``scripts/qmt_export_calendar.py`` 产出的扁平 ``trading_calendar/data.csv``
    （只有 ``trade_date`` 一列）。日历是只增不减的，因此这里用 ``INSERT OR REPLACE``
    而不是先清表。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        source_root: 大 QMT 落盘根目录。

    返回：
        并入的交易日数量；两个位置都没有日历时返回 0。
    """
    frame = _read_calendar_frame(source_root)
    if frame is None or frame.empty:
        return 0
    connection.register("calendar_snapshot", frame)
    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO trading_calendar(trade_date, calendar_symbol)
            SELECT CAST(trade_date AS DATE), calendar_symbol FROM calendar_snapshot
            """
        )
    finally:
        connection.unregister("calendar_snapshot")
    return len(frame)


def _read_calendar_frame(source_root: Path | str) -> pd.DataFrame | None:
    """读取交易日历 CSV 并归一化为两列表。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        含 ``trade_date``（``datetime64``）与 ``calendar_symbol`` 两列的表；
        两个候选文件都不存在或无法解析时返回 ``None``。
    """
    root = Path(source_root) / DATASET_CALENDAR
    candidates = (root / "snapshot=latest" / "data.csv", root / "data.csv")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = pd.read_csv(str(path), encoding="utf-8-sig", dtype=str)
        except (OSError, ValueError):
            continue
        if raw.empty:
            continue
        column = "trade_date" if "trade_date" in raw.columns else raw.columns[0]
        frame = pd.DataFrame()
        frame["trade_date"] = pd.to_datetime(
            raw[column].astype(str).str.strip(), format="%Y%m%d", errors="coerce"
        )
        frame["calendar_symbol"] = (
            raw["calendar_symbol"].fillna("").astype(str)
            if "calendar_symbol" in raw.columns
            else ""
        )
        frame = frame.dropna(subset=["trade_date"])
        return frame.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)
    return None


def refresh_corporate_actions(
    connection,
    source_root: Path | str,
    ex_dates,
    *,
    rebuild: bool = False,
) -> int:
    """增量刷新 ``corporate_actions``。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        source_root: 大 QMT 落盘根目录。
        ex_dates: 需要重新读取的八位除权日序列。``rebuild`` 为 ``True`` 时忽略。
        rebuild: 是否整表重建。为 ``True`` 时清空全表并按整目录通配重新读取。

    返回：
        写入的记录行数；没有任何待处理分区时返回 0。
    """
    root = Path(source_root) / DATASET_ACTIONS
    values = [] if rebuild else _safe_date_values(ex_dates)
    if not rebuild and not values:
        return 0
    # 脏分区太多时改用整目录通配，此时读到的是全量数据，必须同步降级为整表重建，
    # 否则只删了脏分区却插入全部记录，会在主键上撞车。
    if len(values) > _GLOB_THRESHOLD:
        rebuild = True
        values = []
    if rebuild:
        source = "'{0}'".format(sql_path(root / "ex_date=*" / "data.csv"))
        connection.execute("DELETE FROM corporate_actions")
    else:
        source = _csv_source_expression(root, values)
        # 逐行 executemany 在带主键的表上会退化到每行一次索引维护，这里用一条
        # 集合语句删除，避免脏分区一多就慢到不可接受。
        connection.execute(
            f"DELETE FROM corporate_actions WHERE ex_date IN ({_ex_date_literals(values)})"
        )
    columns = ", ".join(
        f"'{name}': '{kind}'" for name, kind in _ACTION_CSV_COLUMNS.items()
    )
    inserted = connection.execute(
        f"""
        INSERT INTO corporate_actions(
            code, ex_date, cash_dividend_per_share, bonus_share_per_share,
            capitalization_per_share, rights_issue_per_share, rights_issue_price,
            share_reform_flag, adjustment_factor)
        SELECT upper(trim(code)), CAST(strptime(ex_date, '%Y%m%d') AS DATE),
               cash_dividend_per_share, bonus_share_per_share,
               capitalization_per_share, rights_issue_per_share, rights_issue_price,
               share_reform_flag, adjustment_factor
        FROM read_csv({source}, header=true, hive_partitioning=false,
                      union_by_name=true, columns={{{columns}}})
        WHERE code IS NOT NULL AND trim(code) <> ''
          AND ex_date IS NOT NULL AND length(trim(ex_date)) = 8
        """
    ).fetchall()
    return int(inserted[0][0]) if inserted and inserted[0] else 0


def _safe_date_values(values) -> list[str]:
    """筛出可以安全内联进 SQL 的八位日期。

    分区值来自磁盘目录名，理论上可以是任意字符串。这里只放行 ``YYYYMMDD``，
    既挡住注入，也避免一个畸形目录名让 ``strptime`` 报错而回滚整次同步。

    参数：
        values: 待校验的分区值可迭代对象。

    返回：
        去重并升序排列的八位日期列表；不合规的值被静默丢弃。
    """
    return sorted(
        {
            text
            for text in (str(value).strip() for value in values)
            if len(text) == 8 and text.isdigit()
        }
    )


def _ex_date_literals(values) -> str:
    """把八位日期列表拼成 SQL 的 ``IN`` 列表。

    参数：
        values: 已经过 ``_safe_date_values`` 校验的八位日期序列。

    返回：
        逗号分隔的 ``CAST(strptime(...) AS DATE)`` 片段。
    """
    return ", ".join(
        f"CAST(strptime('{value}', '%Y%m%d') AS DATE)" for value in values
    )


def delete_corporate_actions(connection, ex_dates) -> int:
    """删除指定除权日的全部记录。

    用于源侧分区消失的情况：只清掉 ``ingest_state`` 的指纹而留着数据，会让这些
    记录永远留在表里，之后每次复权都被它们带偏。

    删除严格限定在传入的除权日上。分区被整片删除时逐个日期拼 IN 列表会撑出几十
    万字符的 SQL，因此按 ``_GLOB_THRESHOLD`` 分批下发多条 ``DELETE``，而**不能**
    图省事整表清空：整表清空依赖「随后的 refresh 会重新灌回来」，而本次若没有任何
    脏除权分区，``refresh_corporate_actions`` 会直接返回，除权表就此为空且不再恢复，
    此后所有复权系数都退化成 1.0，历史价在除权日出现假跳变。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        ex_dates: 需要清除的八位除权日序列。

    返回：
        传入的除权日数量；其中本来就没有记录的日期不会报错。
    """
    values = _safe_date_values(ex_dates)
    if not values:
        return 0
    for start in range(0, len(values), _GLOB_THRESHOLD):
        chunk = values[start : start + _GLOB_THRESHOLD]
        connection.execute(
            f"DELETE FROM corporate_actions WHERE ex_date IN ({_ex_date_literals(chunk)})"
        )
    return len(values)


def _csv_source_expression(root: Path, partition_values) -> str:
    """为一批分区拼出 ``read_csv`` 的文件列表表达式。

    只列出确实变化的分区文件，避免把未变化的分区一起读回来。调用方负责在分区数
    超过 ``_GLOB_THRESHOLD`` 时改走整表重建，因此这里不再处理通配情况。

    参数：
        root: 数据集目录，例如 ``<源根>/corporate_actions``。
        partition_values: 已排序去重的分区值序列，且长度不超过阈值。

    返回：
        可直接嵌入 ``read_csv(...)`` 第一个参数位置的 SQL 列表片段。
    """
    paths = [
        "'{0}'".format(sql_path(root / f"ex_date={value}" / "data.csv"))
        for value in partition_values
    ]
    return "[{0}]".format(", ".join(paths))
