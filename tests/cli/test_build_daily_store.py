"""验证日线库构建命令的结果摘要输出与日志落盘。"""

from __future__ import annotations

import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from quant.cli import build_daily_store
from quant.market_data.daily.ingest.filter_reasons import (
    FILTER_REASONS,
    FilteredSample,
    describe_reason,
)
from quant.market_data.daily.ingest.sync import SyncReport


class BuildDailyStoreSummaryTest(unittest.TestCase):
    """摘要输出与日志文件的行为。"""

    def setUp(self) -> None:
        """每个用例前清空根日志处理器。

        ``logging.basicConfig`` 在根日志已有处理器时是空操作，不清理会让后一个
        用例拿不到自己的文件处理器；同时残留的文件句柄会挡住临时目录清理。

        返回：
            无返回值。
        """
        self.addCleanup(self._reset_logging)
        self._reset_logging()

    def _reset_logging(self) -> None:
        """关闭并移除根日志上的全部处理器。

        返回：
            无返回值。
        """
        root = logging.getLogger()
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)

    def _run_main(
        self,
        report: SyncReport | Exception,
        extra_args: list[str] | None = None,
        expected_exit: int = 0,
    ) -> tuple[str, Path]:
        """用给定的同步报告跑一次命令入口并捕获标准输出。

        参数：
            report: 用来替换真实同步结果的报告对象；传异常实例则让打桩函数抛出它。
            extra_args: 追加到命令行的额外参数；``None`` 表示不追加。
            expected_exit: 期望的退出码。

        返回：
            ``(标准输出文本, 日线库路径)``；日线库位于本用例的临时目录下。
        """
        directory = tempfile.TemporaryDirectory()
        # 先登记临时目录清理、再登记日志清理：cleanup 是后进先出，这样能保证
        # 文件处理器一定先关闭，否则用例失败时会被 Windows 的文件占用错误掩盖。
        self.addCleanup(directory.cleanup)
        self.addCleanup(self._reset_logging)
        database = Path(directory.name) / "qmt_daily.duckdb"
        argv = ["--database", str(database), *(extra_args or [])]
        stub_kwargs = (
            {"side_effect": report} if isinstance(report, Exception) else {"return_value": report}
        )
        buffer = io.StringIO()
        with patch.object(build_daily_store, "sync_daily_store", **stub_kwargs) as stub:
            with redirect_stdout(buffer):
                exit_code = build_daily_store.main(argv)
        stub.assert_called_once()
        self.assertEqual(exit_code, expected_exit)
        self._reset_logging()
        return buffer.getvalue(), database

    def test_summary_lists_filter_reasons(self) -> None:
        """有过滤记录时应逐条打印原因与行数，不必翻日志。"""
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=(("before_listing_padding", 12), ("unknown_code", 3)),
            )
        )
        self.assertIn("本次同步累计过滤（按原因）:", output)
        self.assertIn("before_listing_padding: 12 行", output)
        self.assertIn("unknown_code: 3 行", output)

    def test_summary_prints_zero_count_reasons(self) -> None:
        """计数为 0 的原因也要照常打印，否则看不出这条判据到底查没查。"""
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=tuple((reason, 0) for reason in FILTER_REASONS),
            )
        )
        for reason in FILTER_REASONS:
            self.assertIn(f"{reason}: 0 行", output)

    def test_summary_explains_every_reason_and_gives_verdict(self) -> None:
        """摘要要逐个原因给出判定、含义、处理，并在末尾下结论。"""
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=tuple(
                    (reason, 12 if reason == "before_listing_padding" else 0)
                    for reason in FILTER_REASONS
                ),
            )
        )
        for reason in FILTER_REASONS:
            info = describe_reason(reason)
            self.assertIn(f"判定: {info.condition}", output)
            self.assertIn(f"含义: {info.meaning}", output)
            self.assertIn(f"处理: {info.action}", output)
        self.assertIn("本次过滤全部落在预期原因内，无需人工核对。", output)

    def test_summary_flags_reasons_needing_review(self) -> None:
        """命中需核对的原因时，终端最后一行要点名，不能埋在几十行说明里。"""
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=tuple(
                    (reason, 5 if reason == "before_listing_with_data" else 0)
                    for reason in FILTER_REASONS
                ),
            )
        )
        self.assertIn("需人工核对: before_listing_with_data(5 行)", output)
        self.assertNotIn("全部落在预期原因内", output)

    def test_summary_lists_filtered_samples_under_each_reason(self) -> None:
        """每个原因下面要能直接看到具体被滤掉的行，不必再去翻源 CSV。"""
        sample = FilteredSample(
            reason="before_listing_padding",
            code="600000.SH",
            trade_date="20240102",
            open_date="2024-01-03",
            expire_date=None,
            suspend_flag=1.0,
            volume=0.0,
            close=7.0,
        )
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=(("before_listing_padding", 12),),
                filtered_samples=(("before_listing_padding", (sample,)),),
            )
        )
        self.assertIn("随机样例 1 条:", output)
        self.assertIn(
            "600000.SH 20240102 open=2024-01-03 expire=- suspend=1 volume=0 close=7.0000",
            output,
        )

    def test_summary_omits_sample_line_without_samples(self) -> None:
        """只有计数、没有样例时不应打印空的样例小标题。"""
        output, _ = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=(("before_listing_padding", 12),),
            )
        )
        self.assertIn("before_listing_padding: 12 行", output)
        self.assertNotIn("随机样例", output)

    def test_summary_omits_section_without_filtering(self) -> None:
        """一行都没过滤时不应打印空的过滤小节。"""
        output, _ = self._run_main(SyncReport(status="up_to_date", rows_after=100))
        self.assertNotIn("本次同步累计过滤", output)

    def test_summary_is_saved_and_path_is_printed(self) -> None:
        """摘要要同时落盘，并在终端显示保存路径。"""
        output, database = self._run_main(
            SyncReport(
                status="synced",
                rows_after=100,
                filtered_totals=(("before_listing_padding", 12),),
            )
        )
        log_dir = database.parent / build_daily_store.LOG_SUBDIR
        saved = sorted(log_dir.glob("sync_*.log"))
        self.assertEqual(len(saved), 1)
        self.assertIn(f"日志已保存: {saved[0]}", output)
        content = saved[0].read_text(encoding="utf-8")
        self.assertIn("before_listing_padding: 12 行", content)
        self.assertIn("日线库同步synced", content)

    def test_explicit_log_file_is_used(self) -> None:
        """显式指定路径时应写到该文件，而不是缺省目录。"""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(self._reset_logging)
        target = Path(directory.name) / "nested" / "run.log"
        output, database = self._run_main(
            SyncReport(status="synced", rows_after=100),
            ["--log-file", str(target)],
        )
        self.assertTrue(target.is_file())
        self.assertIn(f"日志已保存: {target}", output)
        self.assertFalse((database.parent / build_daily_store.LOG_SUBDIR).exists())

    def test_no_log_file_writes_nothing(self) -> None:
        """显式关闭文件日志时不应创建任何日志文件。"""
        output, database = self._run_main(
            SyncReport(status="synced", rows_after=100), ["--no-log-file"]
        )
        self.assertNotIn("日志已保存", output)
        self.assertFalse((database.parent / build_daily_store.LOG_SUBDIR).exists())

    def test_dry_run_does_not_write_log_file_by_default(self) -> None:
        """试运行承诺不写入任何文件，缺省不应建出日志目录。"""
        output, database = self._run_main(
            SyncReport(status="dry_run", rows_after=-1), ["--dry-run"]
        )
        self.assertNotIn("日志已保存", output)
        self.assertFalse((database.parent / build_daily_store.LOG_SUBDIR).exists())

    def test_dry_run_honours_explicit_log_file(self) -> None:
        """试运行时显式指定日志路径仍应落盘，用户的明确要求优先。"""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(self._reset_logging)
        target = Path(directory.name) / "dry.log"
        output, _ = self._run_main(
            SyncReport(status="dry_run", rows_after=-1),
            ["--dry-run", "--log-file", str(target)],
        )
        self.assertTrue(target.is_file())
        self.assertIn(f"日志已保存: {target}", output)

    def test_failure_path_saves_summary_and_path(self) -> None:
        """同步抛错返回 2 时，摘要与保存路径同样要落盘。"""
        output, database = self._run_main(
            RuntimeError("源目录不可读"), expected_exit=2
        )
        saved = sorted((database.parent / build_daily_store.LOG_SUBDIR).glob("sync_*.log"))
        self.assertEqual(len(saved), 1)
        self.assertIn("日线库同步未完成: 源目录不可读", output)
        content = saved[0].read_text(encoding="utf-8")
        self.assertIn("日线库同步未完成: 源目录不可读", content)

    def test_locked_store_returns_three_and_saves_summary(self) -> None:
        """库被占用返回 3 时摘要也要落盘。"""
        output, database = self._run_main(
            SyncReport(status="skipped_locked", message="日线库被占用"), expected_exit=3
        )
        saved = sorted((database.parent / build_daily_store.LOG_SUBDIR).glob("sync_*.log"))
        self.assertEqual(len(saved), 1)
        self.assertIn("日线库同步skipped_locked", output)
        self.assertIn("日线库同步skipped_locked", saved[0].read_text(encoding="utf-8"))

    def test_log_file_and_no_log_file_are_mutually_exclusive(self) -> None:
        """两个互斥参数同时出现应直接报错，而不是静默取其一。"""
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                build_daily_store.build_parser().parse_args(
                    ["--log-file", "a.log", "--no-log-file"]
                )
