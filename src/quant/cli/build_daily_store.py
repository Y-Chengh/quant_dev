"""把大 QMT 落盘的日线 CSV 增量转换为本地日线库的命令行入口。"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from quant.config import default_qmt_daily_database
from quant.market_data.daily.ingest import DailySyncConfig, sync_daily_store
from quant.market_data.daily.ingest.cli_support import add_sync_arguments

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"

#: 缺省日志目录，位于日线库根目录下，与 market_check 报告并列。
LOG_SUBDIR = "reports/build_daily_store"


def build_parser() -> argparse.ArgumentParser:
    """构造日线库构建命令的解析器。

    返回：
        已注册数据库路径、同步开关、试运行与日志等级参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="把大 QMT 落盘的日线 CSV 增量转换为 qmt_daily.duckdb 与月度 Parquet"
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="日线库路径；缺省按 QMT_DAILY_DB_PATH 环境变量解析",
    )
    add_sync_arguments(parser)
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help="整库重建：忽略已有指纹，重写全部月度分片并重建辅助表",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只做增量检查并打印结论，不写入任何数据文件（显式 --log-file 仍会写日志）",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志等级，默认 INFO",
    )
    log_file_group = parser.add_mutually_exclusive_group()
    log_file_group.add_argument(
        "--log-file",
        type=Path,
        help=f"日志与摘要的保存路径；缺省写入 <日线库目录>/{LOG_SUBDIR}/sync_<时间戳>.log"
        "（--dry-run 时缺省不落盘，显式指定本参数仍会写）",
    )
    log_file_group.add_argument(
        "--no-log-file",
        action="store_true",
        help="只输出到终端，不保存日志文件",
    )
    return parser


def _resolve_log_path(
    explicit: Path | None,
    daily_root: Path,
    started_at: datetime,
    *,
    disabled: bool,
    dry_run: bool,
) -> Path | None:
    """决定本次运行的日志文件路径。

    参数：
        explicit: 命令行显式指定的路径；``None`` 表示按缺省规则生成。
        daily_root: 日线库数据根目录，缺省路径以它为基准。
        started_at: 本次运行的开始时间，用于生成带时间戳的文件名。
        disabled: 是否传了 ``--no-log-file``，为真时一律不落盘。
        dry_run: 是否为试运行。试运行承诺「不写入任何文件」，因此缺省不落盘；
            但显式指定 ``--log-file`` 表示用户明确要这个文件，仍然写。

    返回：
        日志文件的完整路径；不落盘时返回 ``None``。父目录尚未创建。
    """
    if disabled:
        return None
    if explicit is not None:
        return Path(explicit)
    if dry_run:
        return None
    stamp = started_at.strftime("%Y%m%d_%H%M%S")
    return daily_root / LOG_SUBDIR / f"sync_{stamp}.log"


def _setup_logging(level: str, log_path: Path | None) -> None:
    """配置根日志，按需同时输出到终端与文件。

    用 ``force=True`` 接管根日志：``basicConfig`` 在根日志已有处理器时是空操作，
    本模块被当作库导入后再调 ``main`` 就会拿不到文件处理器，出现「提示日志已保存
    但文件里没有日志记录」的假象。

    参数：
        level: 日志等级名称，取值见 ``--log-level``。
        log_path: 日志文件路径；``None`` 表示只输出到终端。

    返回：
        无返回值；日志文件的父目录会自动创建，创建或打开失败时抛 ``OSError``。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level),
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def _emit(lines: list[str], text: str) -> None:
    """打印一行摘要并同时留存，供随后写入日志文件。

    参数：
        lines: 收集摘要文本的列表，就地追加。
        text: 本行摘要内容。

    返回：
        无返回值。
    """
    print(text)
    lines.append(text)


def _write_summary(log_path: Path | None, lines: list[str]) -> None:
    """把终端摘要追加到日志文件末尾。

    摘要走 ``print`` 而日志走 ``logging``，两者默认分别落在 stdout 与 stderr；
    这里把摘要补写进同一个文件，保证保存下来的内容与终端所见一致。

    参数：
        log_path: 日志文件路径；``None`` 表示未启用文件日志，直接返回。
        lines: 已经打印到终端的摘要文本。

    返回：
        无返回值；写入失败只提示，不影响命令退出码。
    """
    if log_path is None or not lines:
        return
    # 文件日志处理器仍然打开着，先冲刷再追加，避免两个写入方交错。
    for handler in logging.getLogger().handlers:
        handler.flush()
    try:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    except OSError as error:
        print(f"摘要写入日志文件失败: {error}")


def main(argv: list[str] | None = None) -> int:
    """解析命令行、执行一次同步并打印结果摘要。

    参数：
        argv: 不含程序名的命令行参数列表；缺省时读取当前进程命令行。

    返回：
        同步成功或本来就没有增量返回 0；库被其它进程占用返回 3；
        配置或执行失败返回 2。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    started_at = datetime.now()

    database = args.database if args.database is not None else default_qmt_daily_database()
    log_path = _resolve_log_path(
        args.log_file,
        Path(database).parent,
        started_at,
        disabled=args.no_log_file,
        dry_run=args.dry_run,
    )
    try:
        _setup_logging(args.log_level, log_path)
    except OSError as error:
        # 日志目录不可写不该让整次同步失败，降级为只输出到终端。
        print(f"日志文件无法创建，改为只输出到终端: {error}")
        log_path = None
        _setup_logging(args.log_level, None)

    lines: list[str] = []
    mode = "rebuild" if args.rebuild_all else args.sync_mode
    try:
        report = sync_daily_store(
            DailySyncConfig(
                database=database,
                source_root=args.qmt_output_root,
                config_path=args.qmt_config,
                mode=mode,
                verify_hash=args.sync_verify_hash,
                dry_run=args.dry_run,
            )
        )
    except (OSError, ValueError, RuntimeError) as error:
        _emit(lines, f"日线库同步未完成: {error}")
        if log_path is not None:
            _emit(lines, f"日志已保存: {log_path}")
        _write_summary(log_path, lines)
        return 2

    _emit(
        lines,
        f"日线库同步{report.status}: scanned={report.scanned} dirty={report.dirty} removed={report.removed} shards={report.rewritten_shards} rows={report.rows_after} 用时 {report.elapsed_seconds:.1f}s",
    )
    if report.message:
        _emit(lines, report.message)
    if report.filtered_totals:
        _emit(lines, "本次同步累计过滤（按原因）:")
        for reason, count in report.filtered_totals:
            _emit(lines, f"  {reason}: {count} 行")
    for dataset, partition_key, reason in report.pending[:20]:
        _emit(lines, f"待定分区 {dataset}/{partition_key}: {reason}")
    if len(report.pending) > 20:
        _emit(lines, f"……另有 {len(report.pending) - 20} 个待定分区未列出")
    if log_path is not None:
        _emit(lines, f"日志已保存: {log_path}")
    _write_summary(log_path, lines)
    return 3 if report.status == "skipped_locked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
