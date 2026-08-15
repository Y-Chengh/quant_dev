# -*- coding: utf-8 -*-
"""同时面向大 QMT 输出窗口与外部轮转文件的日志配置。"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(output_root, max_bytes, backup_count):
    """创建终端和 UTF-8 文件双通道日志器。

    参数：
        output_root: 数据根目录；日志写入其 ``logs`` 子目录。
        max_bytes: 单个日志文件轮转前允许的最大字节数。
        backup_count: 保留的历史轮转日志文件数量。

    返回：
        名为 ``quant.qmt_downloader`` 的已配置日志器。
    """
    log_dir = Path(output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
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
        str(log_dir / "downloader.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
