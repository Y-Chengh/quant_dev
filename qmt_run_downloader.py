# -*- coding: utf-8 -*-
"""大 QMT 内置 Python 策略入口；只需在 QMT 中运行本文件。"""

import sys
import traceback


PROJECT_ROOT = r"C:\Users\win10\Documents\quant"
CONFIG_PATH = r"C:\Users\win10\Documents\quant\qmt_daily_downloader\config.test.json"
_STARTED = False

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qmt_daily_downloader.qmt_entry_support import run_download


def init(C):
    """初始化大 QMT 策略，下载任务实际在 ``after_init`` 中启动。

    参数：
        C: 大 QMT 注入的 ``ContextInfo``；本回调不读取行情也不下单。

    返回：
        无返回值。
    """
    print("QMT daily downloader init")


def after_init(C):
    """在大 QMT 环境准备完成后执行且仅执行一次下载任务。

    参数：
        C: 大 QMT 注入的 ``ContextInfo``，用于调用行情、财务和除权接口。

    返回：
        无返回值；异常会完整打印并继续向 QMT 抛出以显示失败状态。
    """
    global _STARTED
    if _STARTED:
        print("QMT daily downloader already started, skip duplicate callback")
        return
    _STARTED = True
    try:
        summary = run_download(C, download_history_data, CONFIG_PATH)
        print("QMT daily downloader completed: {0}".format(summary))
    except Exception:
        print("QMT daily downloader failed:\n{0}".format(traceback.format_exc()))
        raise


def handlebar(C):
    """保留大 QMT 标准回调，但不在每根行情柱上重复下载。

    参数：
        C: 大 QMT 注入的 ``ContextInfo``；该参数仅用于满足标准策略签名。

    返回：
        无返回值。
    """
    return None
