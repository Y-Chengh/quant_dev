# -*- coding: utf-8 -*-
"""为大 QMT 策略入口组装下载器各模块。"""

from .config import DownloaderConfig
from .gateway import QmtGateway
from .logging_setup import configure_logging
from .runner import QmtDailyDownloader
from .state import CheckpointStore
from .storage import DailyPartitionStore


def run_download(context, history_downloader, config_path):
    """从 JSON 配置启动一次完整下载任务。

    参数：
        context: 大 QMT 回调传入的 ``ContextInfo`` 对象。
        history_downloader: 大 QMT 内置全局 ``download_history_data`` 函数。
        config_path: 本地 UTF-8 JSON 配置文件路径。

    返回：
        ``QmtDailyDownloader.run`` 生成的任务摘要字典。
    """
    config = DownloaderConfig.from_json(config_path)
    logger = configure_logging(
        config.output_root, config.log_max_bytes, config.log_backup_count
    )
    store = DailyPartitionStore(config.output_root)
    checkpoints = CheckpointStore(config.output_root / "state" / "downloader_state.sqlite")
    gateway = QmtGateway(
        context,
        history_downloader,
        logger,
        retry_count=config.retry_count,
    )
    downloader = QmtDailyDownloader(config, gateway, store, checkpoints, logger)
    return downloader.run()
