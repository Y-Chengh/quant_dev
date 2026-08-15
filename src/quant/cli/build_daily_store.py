"""把大 QMT 落盘的日线 CSV 增量转换为本地日线库的命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.config import default_qmt_daily_database
from quant.market_data.daily.ingest import DailySyncConfig, sync_daily_store
from quant.market_data.daily.ingest.cli_support import add_sync_arguments

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"


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
        help="只做增量检查并打印结论，不写入任何文件",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志等级，默认 INFO",
    )
    return parser


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
    logging.basicConfig(level=getattr(logging, args.log_level), format=DEFAULT_LOG_FORMAT)

    database = args.database if args.database is not None else default_qmt_daily_database()
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
        print(f"日线库同步未完成: {error}")
        return 2

    print(
        f"日线库同步{report.status}: scanned={report.scanned} dirty={report.dirty} removed={report.removed} shards={report.rewritten_shards} rows={report.rows_after} 用时 {report.elapsed_seconds:.1f}s"
    )
    if report.message:
        print(report.message)
    for dataset, partition_key, reason in report.pending[:20]:
        print(f"待定分区 {dataset}/{partition_key}: {reason}")
    if len(report.pending) > 20:
        print(f"……另有 {len(report.pending) - 20} 个待定分区未列出")
    return 3 if report.status == "skipped_locked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
