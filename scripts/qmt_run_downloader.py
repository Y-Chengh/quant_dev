# -*- coding: utf-8 -*-
"""Strategy entry point for the miniQMT built-in Python interpreter.

This file must stay pure ASCII. The miniQMT editor stores pasted source as GBK,
so any non-ASCII byte makes the utf-8 coding declaration above fail with
"SyntaxError: (unicode error) 'utf-8' codec can't decode byte ...". Chinese
documentation for this downloader lives in docs/qmt_downloader.md instead.

Usage: open this file in the miniQMT editor and run it WITHOUT the
"start local Python" option, otherwise ContextInfo and the built-in
download_history_data global are not injected.
"""

import sys
import traceback


PROJECT_ROOT = r"C:\Users\win10\Documents\quant"
SOURCE_ROOT = PROJECT_ROOT + r"\src"
CONFIG_PATH = PROJECT_ROOT + r"\configs\qmt_downloader\incremental.example.json"
_STARTED = False

if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from quant.qmt_downloader.qmt_entry_support import run_download


def init(C):
    """Initialize the miniQMT strategy; the download runs in ``after_init``.

    Args:
        C: ContextInfo injected by miniQMT. This callback neither reads market
            data nor places orders.

    Returns:
        None.
    """
    # The interpreter version decides which syntax quant.qmt_downloader may use,
    # so print it up front to make compatibility failures easy to diagnose.
    print("QMT daily downloader init, python {0}".format(sys.version))


def after_init(C):
    """Run the download task exactly once after miniQMT finishes startup.

    Args:
        C: ContextInfo injected by miniQMT, used for market data, financial
            statement and corporate action interfaces.

    Returns:
        None. Exceptions are printed in full and re-raised so miniQMT shows the
        strategy as failed.
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
    """Keep the standard miniQMT callback without downloading on every bar.

    Args:
        C: ContextInfo injected by miniQMT. Only present to satisfy the
            standard strategy signature; it is unused.

    Returns:
        None.
    """
    return None
