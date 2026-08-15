# -*- coding: utf-8 -*-
"""同时面向大 QMT 输出窗口与单次运行日志文件的日志配置。"""

import logging
import re
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: 运行模式到日志文件名片段的映射；未列出的模式按原名清洗后使用。
MODE_LOG_TOKENS = {
    "backfill": "back_fill",
    "incremental": "inc",
    "repair": "repair",
}
_UNSAFE_TOKEN = re.compile(r"[^0-9A-Za-z_]+")


def mode_log_token(mode):
    """把运行模式转换为日志文件名中的模式片段。

    参数：
        mode: 配置中的运行模式，通常为 ``backfill``、``incremental`` 或
            ``repair``。

    返回：
        文件名安全的模式片段；未知模式返回清洗后的原文，空值返回 ``unknown``。
    """
    key = str(mode).strip().lower()
    token = MODE_LOG_TOKENS.get(key)
    if token is not None:
        return token
    cleaned = _UNSAFE_TOKEN.sub("_", key).strip("_")
    return cleaned or "unknown"


def resolve_log_path(log_dir, mode, started_at):
    """为本次运行挑选一个不会覆盖既有日志的文件路径。

    文件名为 ``downloader.<YYYYMMDD>.<HH>.<模式>.log``。文件名只精确到小时，
    同一小时内重复运行时追加从 2 开始的序号，保证每次运行仍然独占一份日志，
    而不是把两次运行混进同一个文件。

    参数：
        log_dir: 日志目录，调用前必须已经创建。
        mode: 运行模式，用于生成文件名中的模式片段。
        started_at: 本次运行的开始时间。

    返回：
        当前尚未被占用的日志文件 ``Path``。
    """
    directory = Path(log_dir)
    stem = "downloader.{0}.{1}.{2}".format(
        started_at.strftime("%Y%m%d"),
        started_at.strftime("%H"),
        mode_log_token(mode),
    )
    candidate = directory / "{0}.log".format(stem)
    index = 2
    while candidate.exists():
        candidate = directory / "{0}.{1}.log".format(stem, index)
        index += 1
    return candidate


def configure_logging(output_root, max_bytes, backup_count, mode, started_at=None):
    """创建终端和 UTF-8 文件双通道日志器，并为本次运行单独建立日志文件。

    每次运行写入独立文件，因此排查某一次全市场回溯时不必再从一份持续追加的
    日志中按时间切分。``max_bytes`` 和 ``backup_count`` 仍然生效，但只在单次
    运行内部轮转，不会跨运行覆盖历史日志。

    参数：
        output_root: 数据根目录；日志写入其 ``logs`` 子目录。
        max_bytes: 单个日志文件轮转前允许的最大字节数。
        backup_count: 同一次运行内保留的历史轮转日志文件数量。
        mode: 运行模式，决定文件名中的 ``back_fill``、``inc`` 或 ``repair``。
        started_at: 本次运行开始时间；缺省取当前本地时间，测试可传入固定时间。

    返回：
        ``(logger, log_path)`` 二元组：名为 ``quant.qmt_downloader`` 的已配置
        日志器，以及本次运行实际写入的日志文件路径。
    """
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_log_path(log_dir, mode, started_at or datetime.now())
    logger = logging.getLogger("quant.qmt_downloader")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, log_path


def log_run_configuration(logger, settings):
    """在日志起始处逐项记录本次运行的全部生效配置。

    逐行输出而不是打印成一个大字典，是为了让 ``配置 batch_size=`` 这类检索能
    直接定位到单项取值，也避免超长单行在大 QMT 输出窗口里被截断。

    参数：
        logger: 已完成配置的日志器。
        settings: ``(名称, 文本值)`` 二元组序列，通常来自
            ``DownloaderConfig.describe()``。

    返回：
        无返回值。
    """
    logger.info("生效配置开始")
    for name, value in settings:
        logger.info("配置 %s=%s", name, value)
    logger.info("生效配置结束")
