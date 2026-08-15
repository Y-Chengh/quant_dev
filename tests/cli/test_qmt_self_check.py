"""QMT 全样本自检命令行入口的参数与进度日志配置测试。"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from quant.cli.qmt_self_check import build_parser, configure_progress_logging, main
from quant.qmt_downloader.self_check import LOGGER_NAME


class QmtSelfCheckCliTests(unittest.TestCase):
    """验证进度日志开关的解析、默认值与处理器安装。"""

    def tearDown(self) -> None:
        """移除测试安装的日志处理器并还原级别与传播开关，避免污染其它用例。"""

        logger = logging.getLogger(LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    def test_help_text_renders(self) -> None:
        """帮助文本必须可渲染，防止 help 中的字面百分号触发插值错误。"""

        self.assertIn("--progress-every", build_parser().format_help())

    def test_progress_options_have_expected_defaults(self) -> None:
        """缺省应输出 INFO 级进度、自动选择步长且不额外写日志文件。"""

        args = build_parser().parse_args([])
        self.assertEqual(args.log_level, "INFO")
        self.assertEqual(args.progress_every, 0)
        self.assertIsNone(args.log_file)

    def test_configure_progress_logging_writes_console_and_file(self) -> None:
        """指定日志文件时终端与文件应同时收到进度日志，且重复调用不重复输出。"""

        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "nested" / "self_check.log"
            configure_progress_logging("DEBUG", log_path)
            configure_progress_logging("DEBUG", log_path)
            logger = logging.getLogger(LOGGER_NAME)
            self.assertEqual(len(logger.handlers), 2)
            self.assertEqual(logger.level, logging.DEBUG)
            logger.info("测试进度")
            for handler in logger.handlers:
                handler.flush()
            self.assertIn("测试进度", log_path.read_text(encoding="utf-8"))
            # Windows 上未关闭的文件句柄会让临时目录无法删除，必须先释放处理器。
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def test_unknown_log_level_is_rejected(self) -> None:
        """未知级别必须报错，而不是静默按默认级别运行。"""

        with self.assertRaises(ValueError):
            configure_progress_logging("VERBOSE")

    def test_unusable_log_file_returns_exit_code_two(self) -> None:
        """--log-file 不可用时应按配置错误返回 2，而不是抛出未捕获异常。"""

        with tempfile.TemporaryDirectory() as directory:
            # 传入一个已存在的目录：文件处理器无法在其上创建日志文件。
            self.assertEqual(main(["--log-file", directory]), 2)

    def test_negative_progress_every_exits_with_usage_error(self) -> None:
        """负步长必须在解析阶段报错，而不是进入耗时的扫描后才失败。"""

        with self.assertRaises(SystemExit) as context:
            main(["--progress-every", "-1"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
