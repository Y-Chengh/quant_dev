# -*- coding: utf-8 -*-
u"""\u5927 QMT \u5185\u7f6e Python \u7b56\u7565\u5165\u53e3\uff1b\u53ea\u9700\u5728 QMT \u4e2d\u8fd0\u884c\u672c\u6587\u4ef6\u3002"""

import sys
import traceback


PROJECT_ROOT = r"C:\Users\win10\Documents\quant"
CONFIG_PATH = r"C:\Users\win10\Documents\quant\qmt_daily_downloader\config.test.json"
_STARTED = False

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qmt_daily_downloader.qmt_entry_support import run_download


def init(C):
    u"""\u521d\u59cb\u5316\u5927 QMT \u7b56\u7565\uff0c\u4e0b\u8f7d\u4efb\u52a1\u5b9e\u9645\u5728 ``after_init`` \u4e2d\u542f\u52a8\u3002

    \u53c2\u6570\uff1a
        C: \u5927 QMT \u6ce8\u5165\u7684 ``ContextInfo``\uff1b\u672c\u56de\u8c03\u4e0d\u8bfb\u53d6\u884c\u60c5\u4e5f\u4e0d\u4e0b\u5355\u3002

    \u8fd4\u56de\uff1a
        \u65e0\u8fd4\u56de\u503c\u3002
    """
    print("QMT daily downloader init")


def after_init(C):
    u"""\u5728\u5927 QMT \u73af\u5883\u51c6\u5907\u5b8c\u6210\u540e\u6267\u884c\u4e14\u4ec5\u6267\u884c\u4e00\u6b21\u4e0b\u8f7d\u4efb\u52a1\u3002

    \u53c2\u6570\uff1a
        C: \u5927 QMT \u6ce8\u5165\u7684 ``ContextInfo``\uff0c\u7528\u4e8e\u8c03\u7528\u884c\u60c5\u3001\u8d22\u52a1\u548c\u9664\u6743\u63a5\u53e3\u3002

    \u8fd4\u56de\uff1a
        \u65e0\u8fd4\u56de\u503c\uff1b\u5f02\u5e38\u4f1a\u5b8c\u6574\u6253\u5370\u5e76\u7ee7\u7eed\u5411 QMT \u629b\u51fa\u4ee5\u663e\u793a\u5931\u8d25\u72b6\u6001\u3002
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
    u"""\u4fdd\u7559\u5927 QMT \u6807\u51c6\u56de\u8c03\uff0c\u4f46\u4e0d\u5728\u6bcf\u6839\u884c\u60c5\u67f1\u4e0a\u91cd\u590d\u4e0b\u8f7d\u3002

    \u53c2\u6570\uff1a
        C: \u5927 QMT \u6ce8\u5165\u7684 ``ContextInfo``\uff1b\u8be5\u53c2\u6570\u4ec5\u7528\u4e8e\u6ee1\u8db3\u6807\u51c6\u7b56\u7565\u7b7e\u540d\u3002

    \u8fd4\u56de\uff1a
        \u65e0\u8fd4\u56de\u503c\u3002
    """
    return None
