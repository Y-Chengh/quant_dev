"""同步相关命令行参数的注册与装配。

按 AGENTS.md 的分层要求，参数定义放在库层，命令行模块只负责调用；这样
``python -m quant.cli.build_daily_store``、
``python -m quant.cli.market_check`` 和因子实验的日线数据源
可以共用同一套开关，不会出现三份互相漂移的定义。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .sync import SYNC_MODES, DailySyncConfig, SyncReport, sync_daily_store

logger = logging.getLogger(__name__)


def add_sync_arguments(parser: argparse.ArgumentParser) -> None:
    """注册日线库增量同步的通用参数。

    参数：
        parser: 目标解析器；调用后会新增 ``--qmt-output-root``、
            ``--qmt-config``、``--sync-mode``、``--sync-verify-hash``、
            ``--no-auto-sync`` 五个参数。

    返回：
        无返回值。
    """
    parser.add_argument(
        "--qmt-output-root",
        type=Path,
        help="大 QMT 落盘根目录；缺省按 QMT_OUTPUT_ROOT 环境变量与下载器配置解析",
    )
    parser.add_argument(
        "--qmt-config",
        type=Path,
        help="下载器 JSONC 配置；只用于从中读取 output_root",
    )
    parser.add_argument(
        "--sync-mode",
        choices=SYNC_MODES,
        default="auto",
        help="增量检查力度：auto 允许水位短路，full 强制完整扫描，rebuild 整库重建",
    )
    parser.add_argument(
        "--sync-verify-hash",
        action="store_true",
        help="对内容变化的分区重算 data.csv 的 SHA-256 并与完成标记比对",
    )
    parser.add_argument(
        "--no-auto-sync",
        action="store_true",
        help="跳过启动时的自动增量检查，直接使用现有日线库",
    )


def sync_from_args(
    args: argparse.Namespace,
    *,
    database: Path | None = None,
    dry_run: bool = False,
) -> SyncReport | None:
    """按命令行参数执行一次自动增量同步。

    参数：
        args: 已解析的命令行参数；需含 ``add_sync_arguments`` 注册的字段，
            缺失字段一律按缺省值处理，便于测试构造轻量命名空间。
        database: 目录库路径；``None`` 表示按环境变量与内置回退值解析。
        dry_run: 只检查不写入。

    返回：
        同步结果；``--no-auto-sync`` 时返回 ``None``，表示本次没有做任何检查。
    """
    if getattr(args, "no_auto_sync", False):
        logger.info("[daily-sync] 已通过 --no-auto-sync 跳过自动增量检查")
        return None
    config = DailySyncConfig(
        database=database,
        source_root=getattr(args, "qmt_output_root", None),
        config_path=getattr(args, "qmt_config", None),
        mode=getattr(args, "sync_mode", "auto"),
        verify_hash=getattr(args, "sync_verify_hash", False),
        dry_run=dry_run,
    )
    return sync_daily_store(config)
