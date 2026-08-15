from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from quant.config import (
    MARKET_DATABASE_ENV,
    PROJECT_ROOT_ENV,
    configs_dir,
    default_factor_config_path,
    default_market_database,
    default_qmt_config_path,
    project_root,
)
from quant.config.paths import FALLBACK_MARKET_DATABASE


class ProjectRootTest(unittest.TestCase):
    """验证项目根目录解析的优先级与推算结果。"""

    def test_root_contains_expected_top_level_directories(self):
        """未设置环境变量时，应推算出包含 src 与 configs 的项目根目录。"""
        with patch.dict("os.environ", {}, clear=False) as _:
            with patch.dict("os.environ", {PROJECT_ROOT_ENV: ""}):
                root = project_root()
        self.assertTrue((root / "src" / "quant").is_dir())
        self.assertTrue((root / "configs").is_dir())

    def test_environment_variable_overrides_source_layout(self):
        """设置 QUANT_PROJECT_ROOT 后应完全以环境变量为准。"""
        with patch.dict("os.environ", {PROJECT_ROOT_ENV: r"C:\tmp\other-root"}):
            self.assertEqual(project_root(), Path(r"C:\tmp\other-root").resolve())

    def test_configs_dir_is_relative_to_project_root(self):
        """配置目录必须始终位于项目根目录下的 configs。"""
        with patch.dict("os.environ", {PROJECT_ROOT_ENV: r"C:\tmp\other-root"}):
            self.assertEqual(configs_dir(), Path(r"C:\tmp\other-root").resolve() / "configs")


class MarketDatabasePathTest(unittest.TestCase):
    """验证行情库路径的环境变量优先级。"""

    def test_environment_variable_is_used_when_set(self):
        """设置 MARKET_DB_PATH 后应直接返回该路径。"""
        with patch.dict("os.environ", {MARKET_DATABASE_ENV: r"C:\data\custom.duckdb"}):
            self.assertEqual(default_market_database(), Path(r"C:\data\custom.duckdb"))

    def test_fallback_is_used_when_variable_missing(self):
        """未设置环境变量时应返回内置回退路径。"""
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(default_market_database(), FALLBACK_MARKET_DATABASE)

    def test_empty_variable_falls_back(self):
        """环境变量为空字符串时视为未设置，避免解析出当前目录。"""
        with patch.dict("os.environ", {MARKET_DATABASE_ENV: ""}):
            self.assertEqual(default_market_database(), FALLBACK_MARKET_DATABASE)


class ConfigSampleTest(unittest.TestCase):
    """验证配置样例路径拼接以及仓库中确实存在这些样例。"""

    def test_factor_config_sample_exists(self):
        """因子研究的样例配置必须能按名字解析到真实文件。"""
        path = default_factor_config_path("example.yaml")
        self.assertEqual(path.parent, configs_dir() / "factor_research")
        self.assertTrue(path.is_file())

    def test_qmt_config_samples_exist(self):
        """下载器的两份样例配置必须能按名字解析到真实文件。"""
        for name in ("incremental.example.json", "kline_only.backfill.json"):
            with self.subTest(name=name):
                path = default_qmt_config_path(name)
                self.assertEqual(path.parent, configs_dir() / "qmt_downloader")
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
