"""``bars_1d`` 月度 Parquet 分片的重建与原子替换。

一个月对应源目录里 ``kline_1d/date=YYYYMM??`` 一个通配，因此整月重建只需要一条
``COPY``：既天然幂等，也不用处理增量追加的合并语义。实测一个月约 11 万行、
0.13 秒，全量 300 多个月也只在分钟级。

**必须按上市日过滤**：大 QMT 的 ``fill_data=True`` 会给还没上市的证券也造一行
（``suspend_flag=1``、``volume=0``、开高低收全等）。实测 2000-01-04 那天源文件
5209 行里只有 750 行是真实行情。不过滤的话，库会膨胀到三倍，而且每只证券在 IPO
当天都会出现一个由平价段跳到真实价的假跳变，直接污染波动率与收益类因子。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant.qmt_downloader import errata as qmt_errata

from ..schema import sql_path

#: 一次 ``COPY`` 写出的 Parquet 行组大小。
_ROW_GROUP_SIZE = 200_000


@dataclass(frozen=True, slots=True)
class ShardResult:
    """单个月度分片的重建结果。

    参数：
        year: 分片年份。
        month: 分片月份，1 到 12。
        rows: 写入 Parquet 的行数；为 0 表示该月过滤后没有任何行情。
        removed: 是否因为没有数据而删除了原有分片文件。
        path: 最终 Parquet 路径；``removed`` 为 ``True`` 时指向已删除的位置。
    """

    year: int
    month: int
    rows: int
    removed: bool
    path: Path


def shard_directory(daily_root: Path | str, year: int, month: int) -> Path:
    """返回某个月度分片的目录。

    参数：
        daily_root: 日线库数据根目录。
        year: 四位年份。
        month: 月份，1 到 12。

    返回：
        ``<daily_root>/bars_1d/year=YYYY/month=MM`` 路径；本函数不创建目录。
    """
    return Path(daily_root) / "bars_1d" / f"year={year:04d}" / f"month={month:02d}"


def rewrite_month_shard(
    connection,
    source_root: Path | str,
    daily_root: Path | str,
    year: int,
    month: int,
    lifecycle: pd.DataFrame,
    errata: pd.DataFrame | None = None,
) -> ShardResult:
    """从源 CSV 重建一个月的 ``bars_1d`` 分片。

    参数：
        connection: **内存** DuckDB 连接。刻意不用目录库连接，避免在几十秒的
            重活期间独占日线库文件锁而挡住所有读进程。
        source_root: 大 QMT 落盘根目录。
        daily_root: 日线库数据根目录。
        year: 待重建的年份。
        month: 待重建的月份，1 到 12。
        lifecycle: ``load_lifecycle_frame`` 产出的生命周期表，用于过滤未上市与
            已退市的填充行；必须含 ``code``、``open_date``、``expire_date`` 三列。
        errata: ``quant.qmt_downloader.errata.errata_pivot_frame`` 产出的勘误宽表，
            含 ``code``、``trade_date`` 及日线数值字段，未覆盖的字段为
            ``NULL``；``None`` 或空表都表示没有任何勘误记录，此时行为与勘误
            功能加入之前完全一致。落盘（dump）到 Parquet 里的数据已经是应用
            勘误之后的结果。

    返回：
        描述本次重建结果的 ``ShardResult``。该月没有任何源分区、或过滤后为空时，
        原有分片会被删除并返回 ``removed=True``。
    """
    directory = shard_directory(daily_root, year, month)
    target = directory / "bars.parquet"
    pattern = Path(source_root) / "kline_1d" / f"date={year:04d}{month:02d}??" / "data.csv"

    if not _has_source_files(source_root, year, month):
        return _drop_shard(directory, target, year, month)

    if errata is None:
        errata = qmt_errata.errata_pivot_frame({})
    connection.register("lifecycle_filter", lifecycle[["code", "open_date", "expire_date"]])
    connection.register("errata_overrides", errata)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "bars.parquet.part"
    try:
        connection.execute(
            f"""
            COPY (
                SELECT CAST(upper(trim(k.code)) AS VARCHAR)                    AS code,
                       CAST(strptime(k.trade_date, '%Y%m%d') AS DATE)          AS trade_date,
                       CAST(COALESCE(e.open, CAST(k.open AS DOUBLE)) AS DOUBLE)         AS open,
                       CAST(COALESCE(e.high, CAST(k.high AS DOUBLE)) AS DOUBLE)         AS high,
                       CAST(COALESCE(e.low, CAST(k.low AS DOUBLE)) AS DOUBLE)           AS low,
                       CAST(COALESCE(e.close, CAST(k.close AS DOUBLE)) AS DOUBLE)       AS close,
                       CAST(COALESCE(e.pre_close, CAST(k.pre_close AS DOUBLE)) AS DOUBLE)
                                                                               AS pre_close,
                       CAST(round(COALESCE(e.volume, CAST(k.volume AS DOUBLE))) AS BIGINT)
                                                                               AS volume,
                       CAST(COALESCE(e.amount, CAST(k.amount AS DOUBLE)) AS DOUBLE)     AS amount,
                       CAST(COALESCE(e.suspend_flag,
                                     COALESCE(round(CAST(k.suspend_flag AS DOUBLE)), 0)) AS TINYINT)
                                                                               AS suspend_flag
                FROM read_csv('{sql_path(pattern)}', header=true, all_varchar=true,
                              hive_partitioning=false, union_by_name=true) k
                JOIN lifecycle_filter l ON upper(trim(k.code)) = l.code
                LEFT JOIN errata_overrides e
                    ON upper(trim(k.code)) = e.code AND trim(k.trade_date) = e.trade_date
                WHERE k.trade_date IS NOT NULL
                  AND length(trim(k.trade_date)) = 8
                  AND l.open_date IS NOT NULL
                  AND CAST(strptime(k.trade_date, '%Y%m%d') AS DATE) >= CAST(l.open_date AS DATE)
                  AND (l.expire_date IS NULL
                       OR CAST(strptime(k.trade_date, '%Y%m%d') AS DATE) <= CAST(l.expire_date AS DATE))
                ORDER BY trade_date, code
            ) TO '{sql_path(temporary)}'
              (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3, ROW_GROUP_SIZE {_ROW_GROUP_SIZE})
            """
        )
        rows = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{sql_path(temporary)}')"
            ).fetchone()[0]
        )
    finally:
        connection.unregister("lifecycle_filter")
        connection.unregister("errata_overrides")

    if rows == 0:
        temporary.unlink(missing_ok=True)
        return _drop_shard(directory, target, year, month)
    temporary.replace(target)
    return ShardResult(year=year, month=month, rows=rows, removed=False, path=target)


def _has_source_files(source_root: Path | str, year: int, month: int) -> bool:
    """判断某个月在源目录里还有没有日分区。

    参数：
        source_root: 大 QMT 落盘根目录。
        year: 四位年份。
        month: 月份，1 到 12。

    返回：
        至少存在一个 ``date=YYYYMMDD/data.csv`` 时返回 ``True``。
    """
    root = Path(source_root) / "kline_1d"
    prefix = f"date={year:04d}{month:02d}"
    if not root.is_dir():
        return False
    return any((candidate / "data.csv").is_file() for candidate in root.glob(prefix + "??"))


def _drop_shard(directory: Path, target: Path, year: int, month: int) -> ShardResult:
    """删除一个已经没有数据的月度分片。

    参数：
        directory: 分片目录。
        target: 分片 Parquet 文件路径。
        year: 分片年份。
        month: 分片月份。

    返回：
        ``rows=0``、``removed=True`` 的 ``ShardResult``。
    """
    target.unlink(missing_ok=True)
    for leftover in (directory, directory.parent):
        try:
            leftover.rmdir()
        except OSError:
            # 目录非空或不存在都属于正常情况，不需要处理。
            break
    return ShardResult(year=year, month=month, rows=0, removed=True, path=target)


def existing_shards(daily_root: Path | str) -> tuple[tuple[int, int], ...]:
    """列出磁盘上已有的月度分片。

    参数：
        daily_root: 日线库数据根目录。

    返回：
        按时间升序排列的 ``(年, 月)`` 元组；目录不存在时返回空元组。
    """
    root = Path(daily_root) / "bars_1d"
    if not root.is_dir():
        return ()
    months: list[tuple[int, int]] = []
    for year_dir in root.glob("year=*"):
        year_text = year_dir.name.split("=", 1)[-1]
        if not year_text.isdigit():
            continue
        for month_dir in year_dir.glob("month=*"):
            month_text = month_dir.name.split("=", 1)[-1]
            if not month_text.isdigit():
                continue
            if (month_dir / "bars.parquet").is_file():
                months.append((int(year_text), int(month_text)))
    return tuple(sorted(months))


def source_months(source_root: Path | str) -> tuple[tuple[int, int], ...]:
    """列出源目录里 ``kline_1d`` 覆盖的全部年月。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        按时间升序排列的 ``(年, 月)`` 元组；目录不存在时返回空元组。
    """
    root = Path(source_root) / "kline_1d"
    if not root.is_dir():
        return ()
    months: set[tuple[int, int]] = set()
    for candidate in root.glob("date=*"):
        value = candidate.name.split("=", 1)[-1]
        if len(value) == 8 and value.isdigit():
            months.add((int(value[:4]), int(value[4:6])))
    return tuple(sorted(months))
