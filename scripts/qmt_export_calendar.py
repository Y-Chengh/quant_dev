# -*- coding: utf-8 -*-
"""Export the miniQMT trading calendar to CSV for quant-qmt-self-check.

This file must stay pure ASCII. The miniQMT editor stores pasted source as GBK,
so any non-ASCII byte makes the utf-8 coding declaration above fail with
"SyntaxError: (unicode error) 'utf-8' codec can't decode byte ...".

Prefer the downloader itself: adding "trading_calendar" to the datasets list of
the downloader config writes trading_calendar/snapshot=latest with a completion
marker, which the self check picks up automatically. This standalone export is
for exporting a calendar without running a download at all; it writes the flat
trading_calendar/data.csv fallback path, which has no completion marker.

The self check needs an independent calendar either way: without one the audit
can only reuse the dates it already sees on disk, which cannot detect a whole
trading day whose partition is missing.

Usage: open this file in the miniQMT editor and run it WITHOUT the
"start local Python" option, otherwise ContextInfo is not injected. Writing the
file to OUTPUT_CSV = <output_root>\\trading_calendar\\data.csv lets the self
check pick it up with no --calendar-csv argument.
"""

import os
import sys
import traceback
from datetime import datetime


PROJECT_ROOT = r"C:\Users\win10\Documents\quant"
SOURCE_ROOT = PROJECT_ROOT + r"\src"
OUTPUT_CSV = r"D:\qmt_kline_test1\trading_calendar\data.csv"
CALENDAR_SYMBOL = "000001.SH"
START_DATE = "20000101"
END_DATE = "20260812"
_STARTED = False

if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from quant.qmt_downloader.dates import normalize_date


def _write_calendar_csv(path, dates):
    """Write the trading dates through a temporary file and atomic rename.

    Args:
        path: Final CSV path; the parent directory is created when missing.
        dates: Ascending, deduplicated eight digit trading dates.

    Returns:
        None.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        handle.write("trade_date\n")
        for value in dates:
            handle.write(value + "\n")
    if os.path.exists(path):
        os.remove(path)
    os.rename(temporary, path)


def init(C):
    """Initialize the miniQMT strategy; the export runs in ``after_init``.

    Args:
        C: ContextInfo injected by miniQMT.

    Returns:
        None.
    """
    print("QMT trading calendar export init, python {0}".format(sys.version))


def after_init(C):
    """Read the trading calendar once and save it as a single column CSV.

    Args:
        C: ContextInfo injected by miniQMT; only get_trading_dates is used.

    Returns:
        None. Exceptions are printed in full and re-raised so miniQMT shows the
        strategy as failed.
    """
    global _STARTED
    if _STARTED:
        print("QMT trading calendar export already started, skip callback")
        return
    _STARTED = True
    try:
        span = (
            datetime.strptime(END_DATE, "%Y%m%d")
            - datetime.strptime(START_DATE, "%Y%m%d")
        ).days + 1
        raw = C.get_trading_dates(
            CALENDAR_SYMBOL, START_DATE, END_DATE, max(span, 1), "1d"
        )
        dates = sorted(
            set(
                value
                for value in (normalize_date(item) for item in (raw or []))
                if value is not None
            )
        )
        if not dates:
            raise RuntimeError(
                "get_trading_dates returned no date for {0} {1}..{2}".format(
                    CALENDAR_SYMBOL, START_DATE, END_DATE
                )
            )
        _write_calendar_csv(OUTPUT_CSV, dates)
        print(
            "exported {0} trading dates {1}..{2} to {3}".format(
                len(dates), dates[0], dates[-1], OUTPUT_CSV
            )
        )
    except Exception:
        print(
            "QMT trading calendar export failed:\n{0}".format(traceback.format_exc())
        )
        raise


def handlebar(C):
    """Keep the standard miniQMT callback without exporting on every bar.

    Args:
        C: ContextInfo injected by miniQMT; unused.

    Returns:
        None.
    """
    return None
