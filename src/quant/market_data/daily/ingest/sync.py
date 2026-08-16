"""日线库增量同步的编排层。

一次同步的固定顺序，目的是把目录库的写锁窗口压到最短：

1. 抢 ``.sync.lock`` 文件锁，给同一套工具链的其它进程一个明确提示。
2. 只读打开目录库，载入 ``ingest_state`` 与 ``dataset_metadata``，立刻关闭。
3. 无锁跑轻量检查；没有增量就直接返回，全程零写入。
4. 在**内存连接**上重建受影响的月度 Parquet 分片并统计库存。
5. 读写打开目录库，在一个短事务里更新辅助表、指纹、库存与视图。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb

from quant.config import default_qmt_daily_database, default_qmt_errata_path
from quant.qmt_downloader import errata as qmt_errata

from ..schema import SCHEMA_VERSION, apply_schema
from . import aux_tables, catalog, state
from .detect import SourceDelta, detect_changes, watermark_values
from .locking import DailyStoreLockedError, open_catalog, sync_lock
from .shards import existing_shards, rewrite_month_shard, source_months
from .source import DATASET_ACTIONS, DATASET_CALENDAR, resolve_source_root

logger = logging.getLogger(__name__)

#: 同步模式。``auto`` 允许水位短路；``full`` 强制完整扫描；``rebuild`` 整库重建。
SYNC_MODES = ("auto", "full", "rebuild")

#: 重活跑在内存连接上时留给 DuckDB 的最大线程数上限。
_MAX_WORKER_THREADS = 16


def _worker_threads() -> int:
    """决定内存连接可用的 DuckDB 线程数。

    分片重写与库存统计都跑在内存连接上、不持有日线库文件锁，因此可以放开用满
    本机核数——实测 4 线程改为 12 线程，单月重写从 0.313 秒降到 0.245 秒。
    只读查询仍沿用各自 4 线程的既有约定，避免抢占正常查询的资源。

    返回：
        介于 1 与 ``_MAX_WORKER_THREADS`` 之间的线程数。
    """
    return max(1, min(_MAX_WORKER_THREADS, os.cpu_count() or 4))


@dataclass(frozen=True, slots=True)
class DailySyncConfig:
    """一次同步所需的全部输入。

    参数：
        database: ``qmt_daily.duckdb`` 路径；``None`` 表示按
            ``QMT_DAILY_DB_PATH`` 环境变量解析，未设置时用内置回退值。
        source_root: 大 QMT 落盘根目录；``None`` 表示按环境变量与下载器配置解析。
        config_path: 下载器 JSONC 配置路径，仅在需要从配置里取 ``output_root``
            时使用；``None`` 表示用 ``incremental.example.json``。
        mode: 同步模式，取值见 ``SYNC_MODES``。
        verify_hash: 是否对内容变化的分区重算 ``data.csv`` 的 SHA-256。
        dry_run: 只做检查并打印结论，不做任何写入。
        lock_timeout_seconds: 残留同步锁的判定秒数。
        errata_csv: 人工核实的源数据勘误表路径；``None`` 表示用
            ``default_qmt_errata_path()``，文件不存在时视为没有勘误记录。
    """

    database: Path | None = None
    source_root: Path | None = None
    config_path: Path | None = None
    mode: str = "auto"
    verify_hash: bool = False
    dry_run: bool = False
    lock_timeout_seconds: float = 3600.0
    errata_csv: Path | None = None

    def __post_init__(self) -> None:
        """校验同步模式取值。

        返回：
            无返回值；模式非法时抛出 ``ValueError``。
        """
        if self.mode not in SYNC_MODES:
            raise ValueError("mode 必须是 {0} 之一".format(", ".join(SYNC_MODES)))

    def resolved_database(self) -> Path:
        """返回最终使用的日线库文件路径。

        返回：
            显式传入的路径，或按环境变量与内置回退值解析出的路径。
        """
        return Path(self.database) if self.database is not None else default_qmt_daily_database()

    def resolved_daily_root(self) -> Path:
        """返回日线库数据根目录，即库文件所在目录。

        返回：
            ``resolved_database()`` 的父目录。
        """
        return self.resolved_database().parent

    def resolved_source_root(self) -> Path:
        """返回最终使用的大 QMT 落盘根目录。

        返回：
            按「显式传参 > 环境变量 > 下载器配置 > 内置回退」解析出的路径。
        """
        return resolve_source_root(self.source_root, self.config_path)

    def resolved_errata_csv(self) -> Path:
        """返回最终使用的源数据勘误表路径。

        返回：
            显式传入的路径，或 ``default_qmt_errata_path()`` 的默认路径。
        """
        return Path(self.errata_csv) if self.errata_csv is not None else default_qmt_errata_path()


@dataclass(frozen=True, slots=True)
class SyncReport:
    """一次同步的结果摘要。

    参数：
        status: ``up_to_date``（无增量）、``synced``（已入库）、
            ``skipped_locked``（库被占用而跳过）、``dry_run``（只检查未写入）
            或 ``failed``（执行失败）。
        run_id: 本次同步标识；未真正执行时为空串。
        scanned: 扫描到的源分区总数。
        dirty: 内容变化的分区数。
        removed: 源侧消失的分区数。
        rewritten_shards: 重写的月度分片数。
        rows_after: 同步完成后库中的日线总行数；未执行写入时为 ``-1``。
        pending: 无法判定的分区，元素为 ``(dataset, partition_key, 原因)``。
        message: 面向用户的一句话说明。
        elapsed_seconds: 本次同步耗时秒数。
        filtered_totals: 本次同步全部重写月份按过滤原因汇总的丢弃行数，
            ``((原因, 行数), ...)``，原因取值见 ``shards.FILTER_REASONS``；
            未重写任何分片（无增量或 dry-run）时为空元组。
    """

    status: str
    run_id: str = ""
    scanned: int = 0
    dirty: int = 0
    removed: int = 0
    rewritten_shards: int = 0
    rows_after: int = -1
    pending: tuple[tuple[str, str, str], ...] = ()
    message: str = ""
    elapsed_seconds: float = 0.0
    touched_months: tuple[tuple[int, int], ...] = field(default=())
    filtered_totals: tuple[tuple[str, int], ...] = field(default=())


def sync_daily_store(config: DailySyncConfig) -> SyncReport:
    """执行一次日线库增量同步。

    参数：
        config: 同步配置，见 :class:`DailySyncConfig`。

    返回：
        描述本次同步结果的 :class:`SyncReport`。库被其它进程写锁占用时返回
        ``status="skipped_locked"`` 而不抛异常，让调用方可以继续用现有数据运行。
    """
    started_at = datetime.now()
    database = config.resolved_database()
    daily_root = config.resolved_daily_root()
    source_root = config.resolved_source_root()
    try:
        with sync_lock(daily_root, timeout_seconds=config.lock_timeout_seconds):
            return _run_locked(config, database, daily_root, source_root, started_at)
    except DailyStoreLockedError as error:
        logger.warning("日线库正被其它进程写入，本次跳过自动增量同步: %s", error)
        return SyncReport(
            status="skipped_locked",
            message=str(error),
            elapsed_seconds=(datetime.now() - started_at).total_seconds(),
        )


def _run_locked(
    config: DailySyncConfig,
    database: Path,
    daily_root: Path,
    source_root: Path,
    started_at: datetime,
) -> SyncReport:
    """在已经持有同步锁的前提下执行同步主流程。

    参数：
        config: 同步配置。
        database: 目录库文件路径。
        daily_root: 日线库数据根目录。
        source_root: 大 QMT 落盘根目录。
        started_at: 同步开始时间。

    返回：
        本次同步的结果摘要。
    """
    _ensure_schema(database, daily_root)
    with open_catalog(database, read_only=True) as connection:
        fingerprints = state.load_fingerprints(connection)
        metadata = state.read_metadata(connection)

    delta = detect_changes(
        source_root,
        fingerprints,
        metadata,
        mode=config.mode,
        verify_hash=config.verify_hash,
    )
    if config.mode == "rebuild":
        delta = _widen_for_rebuild(delta, source_root, daily_root)

    elapsed = (datetime.now() - started_at).total_seconds()
    if delta.is_empty():
        message = "源目录无增量" + ("（命中水位短路）" if delta.short_circuited else "")
        logger.info("[daily-sync] %s scanned=%d", message, delta.scanned)
        return SyncReport(
            status="up_to_date",
            scanned=delta.scanned,
            pending=delta.pending,
            message=message,
            elapsed_seconds=elapsed,
        )
    if config.dry_run:
        message = f"检测到增量: dirty={len(delta.dirty)} removed={len(delta.removed)} months={len(delta.touched_months())}"
        logger.info("[daily-sync] %s（dry-run，未写入）", message)
        return SyncReport(
            status="dry_run",
            scanned=delta.scanned,
            dirty=len(delta.dirty),
            removed=len(delta.removed),
            pending=delta.pending,
            message=message,
            elapsed_seconds=(datetime.now() - started_at).total_seconds(),
            touched_months=tuple(sorted(delta.touched_months())),
        )
    return _apply_delta(config, database, daily_root, source_root, delta, started_at)


def _apply_delta(
    config: DailySyncConfig,
    database: Path,
    daily_root: Path,
    source_root: Path,
    delta: SourceDelta,
    started_at: datetime,
) -> SyncReport:
    """把检测出的增量真正落到磁盘与目录库。

    参数：
        config: 同步配置。
        database: 目录库文件路径。
        daily_root: 日线库数据根目录。
        source_root: 大 QMT 落盘根目录。
        delta: 轻量检查得到的增量描述。
        started_at: 同步开始时间。

    返回：
        本次同步的结果摘要。
    """
    lifecycle = aux_tables.load_lifecycle_frame(source_root)
    errata_overrides = qmt_errata.load_errata_overrides(config.resolved_errata_csv())
    if errata_overrides:
        logger.info(
            "[daily-sync] 已装载源数据勘误表 %d 条 (证券,交易日) 记录，来自 %s",
            len(errata_overrides),
            config.resolved_errata_csv(),
        )
    errata_frame = qmt_errata.errata_pivot_frame(errata_overrides)
    months = sorted(delta.touched_months())
    memory = duckdb.connect()
    filtered_totals: dict[str, int] = {}
    try:
        memory.execute(f"SET threads = {_worker_threads()}")
        rewritten = 0
        for index, (year, month) in enumerate(months, start=1):
            result = rewrite_month_shard(
                memory, source_root, daily_root, year, month, lifecycle, errata_frame
            )
            rewritten += 1
            breakdown = ", ".join(f"{reason}={count}" for reason, count in result.filtered)
            logger.info(
                "[daily-sync] 重写月度分片 %d/%d %04d-%02d rows=%d%s%s",
                index,
                len(months),
                year,
                month,
                result.rows,
                "（已删除空分片）" if result.removed else "",
                f" filtered=[{breakdown}]" if breakdown else "",
            )
            for reason, count in result.filtered:
                filtered_totals[reason] = filtered_totals.get(reason, 0) + count
        logger.info("[daily-sync] 分片重写完成，开始统计库存")
        inventory = catalog.collect_inventory(memory, daily_root)
        symbols = catalog.collect_symbols(memory, daily_root)
        logger.info(
            "[daily-sync] 库存统计完成: months=%d symbols=%d", len(inventory), len(symbols)
        )
    finally:
        memory.close()

    rows_after = int(inventory["rows"].sum()) if not inventory.empty else 0
    run_id = catalog.make_run_id(started_at)
    dirty_datasets = delta.dirty_datasets()
    rebuild_actions = config.mode == "rebuild"

    logger.info("[daily-sync] 开始写入目录库（此时才会短暂持有写锁）")
    with open_catalog(database, read_only=False) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            aux_tables.refresh_instruments(connection, lifecycle)
            if rebuild_actions or DATASET_CALENDAR in dirty_datasets:
                aux_tables.refresh_trading_calendar(connection, source_root)
            # 源侧消失的除权分区必须显式清掉：只删指纹不删数据的话，
            # 那些记录会永远留在表里，之后每一次复权都会被它们带偏。
            removed_ex_dates = tuple(
                partition_key.split("=", 1)[-1]
                for dataset, partition_key in delta.removed
                if dataset == DATASET_ACTIONS
            )
            if removed_ex_dates:
                aux_tables.delete_corporate_actions(connection, removed_ex_dates)
            if rebuild_actions or DATASET_ACTIONS in dirty_datasets:
                aux_tables.refresh_corporate_actions(
                    connection,
                    source_root,
                    delta.dirty_partition_values(DATASET_ACTIONS),
                    rebuild=rebuild_actions,
                )
            state.upsert_fingerprints(
                connection, list(delta.dirty) + list(delta.refreshed), started_at
            )
            state.delete_fingerprints(connection, delta.removed)
            catalog.refresh_catalog(connection, daily_root, inventory, symbols)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "source_root": str(source_root),
                "adjust_policy": "raw",
                "lifecycle_filter": "instruments",
                "last_sync_run_id": run_id,
            }
            metadata.update(watermark_values(source_root))
            anchor = _max_trade_date(connection)
            if anchor:
                metadata["adjust_anchor_date"] = anchor
            state.write_metadata(connection, metadata)
            finished_at = datetime.now()
            catalog.record_sync_run(
                connection,
                {
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "mode": config.mode,
                    "source_root": str(source_root),
                    "status": "synced",
                    "scanned": delta.scanned,
                    "dirty": len(delta.dirty),
                    "removed": len(delta.removed),
                    "rewritten_shards": rewritten,
                    "rows_after": rows_after,
                    "message": "",
                },
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("CHECKPOINT")

    elapsed = (datetime.now() - started_at).total_seconds()
    filtered_summary = ", ".join(
        f"{reason}={count}" for reason, count in sorted(filtered_totals.items())
    )
    message = f"已入库: dirty={len(delta.dirty)} removed={len(delta.removed)} shards={rewritten} rows={rows_after}"
    logger.info("[daily-sync] %s 用时 %.1fs", message, elapsed)
    if filtered_summary:
        logger.info("[daily-sync] 本次同步累计过滤: %s", filtered_summary)
    if delta.pending:
        logger.warning("[daily-sync] %d 个分区无法判定，已跳过", len(delta.pending))
    return SyncReport(
        status="synced",
        run_id=run_id,
        scanned=delta.scanned,
        dirty=len(delta.dirty),
        removed=len(delta.removed),
        rewritten_shards=rewritten,
        rows_after=rows_after,
        pending=delta.pending,
        message=message,
        filtered_totals=tuple(sorted(filtered_totals.items())),
        elapsed_seconds=elapsed,
        touched_months=tuple(months),
    )


def _ensure_schema(database: Path, daily_root: Path) -> None:
    """确保目录库存在且各张表已建好。

    参数：
        database: 目录库文件路径；父目录会自动创建。
        daily_root: 日线库数据根目录，用于判断能否建 ``bars_1d`` 视图。

    返回：
        无返回值。
    """
    with open_catalog(database, read_only=False) as connection:
        apply_schema(connection, daily_root, create_view=bool(existing_shards(daily_root)))


def _widen_for_rebuild(delta: SourceDelta, source_root: Path, daily_root: Path) -> SourceDelta:
    """整库重建时，把源目录与磁盘上出现过的全部月份都并入待重写集合。

    ``detect_changes`` 在 ``rebuild`` 模式下已经把所有分区标成脏分区，但磁盘上
    可能还留着源侧已经删空的月份，需要一并清理。

    参数：
        delta: 原始检测结果。
        source_root: 大 QMT 落盘根目录。
        daily_root: 日线库数据根目录。

    返回：
        追加了「源侧已消失月份」占位删除记录的新 ``SourceDelta``。
    """
    stale = set(existing_shards(daily_root)) - set(source_months(source_root))
    if not stale:
        return delta
    extra = tuple(
        ("kline_1d", f"date={year:04d}{month:02d}01") for year, month in sorted(stale)
    )
    return SourceDelta(
        dirty=delta.dirty,
        refreshed=delta.refreshed,
        removed=delta.removed + extra,
        pending=delta.pending,
        scanned=delta.scanned,
        short_circuited=delta.short_circuited,
    )


def _max_trade_date(connection) -> str:
    """读取库中最新的交易日，用作前复权的缺省基准日。

    参数：
        connection: 已打开的目录库连接。

    返回：
        形如 ``2026-08-11`` 的日期文本；库中没有任何行情时返回空串。
    """
    row = connection.execute("SELECT max(max_date) FROM monthly_inventory").fetchone()
    if not row or row[0] is None:
        return ""
    return str(row[0])
