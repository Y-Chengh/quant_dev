# -*- coding: utf-8 -*-
"""为大 QMT 策略入口组装下载器各模块。"""

from .config import DownloaderConfig
from .gateway import QmtGateway
from .logging_setup import configure_logging, log_run_configuration
from .runner import QmtDailyDownloader
from .state import CheckpointStore
from .storage import DailyPartitionStore


def run_download(context, history_downloader, config_path, batch_history_downloader=None):
    """从 JSON 配置启动一次完整下载任务。

    参数：
        context: 大 QMT 回调传入的 ``ContextInfo`` 对象。
        history_downloader: 大 QMT 内置全局 ``download_history_data`` 函数。
        config_path: 本地 UTF-8 JSON 配置文件路径。
        batch_history_downloader: 可选的大 QMT 批量历史下载函数
            ``download_history_data2``；提供后日线下载优先走批量接口，失败自动
            回退逐只下载。配置项 ``download_kline_batch`` 为 ``false`` 时忽略。

    返回：
        ``QmtDailyDownloader.run`` 生成的任务摘要字典。
    """
    config = DownloaderConfig.from_json(config_path)
    logger, log_path = configure_logging(
        config.output_root,
        config.log_max_bytes,
        config.log_backup_count,
        config.mode,
    )
    logger.info("本次运行日志文件=%s", log_path)
    log_run_configuration(logger, config.describe())
    store = DailyPartitionStore(config.output_root)
    checkpoints = CheckpointStore(config.output_root / "state" / "downloader_state.sqlite")
    if batch_history_downloader is not None and not config.download_kline_batch:
        logger.info("配置已禁用日线批量下载，本次逐只补充本地缓存")
        batch_history_downloader = None
    elif batch_history_downloader is None:
        logger.info("未提供日线批量下载接口，本次逐只补充本地缓存")
    gateway = QmtGateway(
        context,
        history_downloader,
        logger,
        retry_count=config.retry_count,
        batch_history_downloader=batch_history_downloader,
    )
    downloader = QmtDailyDownloader(config, gateway, store, checkpoints, logger)
    return downloader.run()
