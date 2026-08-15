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


def _purge_cached_quant_modules():
    """Drop quant modules left in sys.modules by an earlier strategy run.

    The miniQMT interpreter outlives a single run, so packages imported last
    time keep shadowing the files on disk. Without this purge, editing the
    library silently has no effect until miniQMT itself is restarted -- and a
    changed function signature surfaces as a confusing TypeError instead.

    Returns:
        None.
    """
    stale = [
        name
        for name in sys.modules
        if name == "quant" or name.startswith("quant.")
    ]
    for name in stale:
        del sys.modules[name]
    if stale:
        print("reloaded {0} cached quant modules from disk".format(len(stale)))


_purge_cached_quant_modules()

from quant.qmt_downloader.qmt_entry_support import run_download


def _report_download_entry_points(C):
    """Print every download-like entry point this miniQMT build offers.

    Only runs when no batch downloader could be resolved, so the log stays
    quiet on healthy setups while making it obvious what else could be tried.

    Args:
        C: ContextInfo injected by miniQMT.

    Returns:
        None.
    """
    from_context = [name for name in dir(C) if "download" in name.lower()]
    from_globals = [name for name in globals() if "download" in name.lower()]
    print("ContextInfo download entry points: {0}".format(from_context))
    print("strategy global download entry points: {0}".format(from_globals))


def _resolve_batch_downloader(C):
    """Locate miniQMT's batch history downloader, if this build exposes one.

    Downloading one symbol per call costs a round trip each time and dominates
    the per-batch runtime; download_history_data2 takes the whole code list in
    a single call. Note that xtquant.xtdata is a separate client from the
    built-in interpreter's injected globals: when this script runs WITHOUT
    "start local Python", that client usually cannot reach the market data
    service and its download call fails with a connection error. The gateway
    falls back to per-symbol downloads on the first failure, so an unusable
    batch entry point costs one attempt, not correctness.

    Args:
        C: ContextInfo injected by miniQMT, inspected only for diagnostics.

    Returns:
        A callable taking (code_list, period, start_time, end_time), or None
        when no batch entry point can be resolved.
    """
    injected = globals().get("download_history_data2")
    if callable(injected):
        return injected
    try:
        from xtquant import xtdata
    except Exception as error:
        print("batch history downloader unavailable: {0}".format(error))
        _report_download_entry_points(C)
        return None
    candidate = getattr(xtdata, "download_history_data2", None)
    if not callable(candidate):
        print("xtquant.xtdata exposes no download_history_data2")
        _report_download_entry_points(C)
        return None
    return candidate


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
        summary = run_download(
            C, download_history_data, CONFIG_PATH, _resolve_batch_downloader(C)
        )
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
