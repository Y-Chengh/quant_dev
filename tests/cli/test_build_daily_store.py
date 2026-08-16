"""验证日线库构建命令的结果摘要输出。"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from quant.cli import build_daily_store
from quant.market_data.daily.ingest.sync import SyncReport


def _run_main(report: SyncReport) -> str:
    """用给定的同步报告跑一次命令入口并捕获标准输出。

    参数：
        report: 用来替换真实同步结果的报告对象。

    返回：
        命令打印到标准输出的全部文本。
    """
    buffer = io.StringIO()
    with patch.object(build_daily_store, "sync_daily_store", return_value=report) as stub:
        with redirect_stdout(buffer):
            exit_code = build_daily_store.main([])
    stub.assert_called_once()
    assert exit_code == 0
    return buffer.getvalue()


class BuildDailyStoreSummaryTest(unittest.TestCase):
    """摘要输出应覆盖过滤明细的有无两种情况。"""

    def test_summary_lists_filter_reasons(self) -> None:
        """有过滤记录时应逐条打印原因与行数，不必翻日志。"""
        output = _run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=(("before_listing", 12), ("unknown_code", 3)),
            )
        )
        self.assertIn("本次同步累计过滤（按原因）:", output)
        self.assertIn("before_listing: 12 行", output)
        self.assertIn("unknown_code: 3 行", output)

    def test_summary_omits_section_without_filtering(self) -> None:
        """一行都没过滤时不应打印空的过滤小节。"""
        output = _run_main(SyncReport(status="up_to_date", rows_after=100))
        self.assertNotIn("本次同步累计过滤", output)
