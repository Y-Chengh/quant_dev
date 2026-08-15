"""验证数据源注册表、日频因子路径与因子缓存指纹的隔离。"""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant.factor_research.data_sources import (
    DEFAULT_DATA_SOURCE,
    MarketDataSource,
    add_data_source_selection_argument,
    add_selected_data_source_arguments,
    available_data_sources,
    data_source_argument_names,
    data_source_from_args,
    register_data_source,
)
from quant.factor_research.factor_factories.capabilities import (
    daily_capable_factor_names,
    intraday_factor_names,
    requires_intraday,
)
from quant.factor_research.factor_factories.registry import FACTOR_FACTORIES
from quant.factor_research.factors import (
    FactorCache,
    build_daily_features_from_daily,
)


def _daily_bars() -> pd.DataFrame:
    """构造两只证券、二十五个交易日的日频行情。

    行数要足够长，才能让 20 日窗口的因子算出有效值。

    返回：
        含标准日频列的行情表。
    """
    rows = []
    for code, base in (("000001.SZ", 10.0), ("600000.SH", 20.0)):
        for index in range(25):
            close = base + index * 0.1
            rows.append(
                {
                    "code": code,
                    "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
                    "open": close - 0.05,
                    "high": close + 0.08,
                    "low": close - 0.09,
                    "close": close,
                    "volume": 1000 + index,
                    "amount": (1000 + index) * close,
                }
            )
    return pd.DataFrame(rows)


class RegistryTest(unittest.TestCase):
    """注册、发现与命令行装配。"""

    def test_builtin_sources_are_discovered(self) -> None:
        """两个内置数据源都应当能被自动发现。"""
        names = available_data_sources()
        self.assertIn("market_service", names)
        self.assertIn("qmt_daily", names)
        self.assertEqual(DEFAULT_DATA_SOURCE, "market_service")

    def test_duplicate_name_is_rejected(self) -> None:
        """重复的数据源名必须报错，避免静默覆盖。"""

        class Duplicate(MarketDataSource):
            """与内置数据源同名的测试数据源。"""

            name = "market_service"
            frequency = "5m"
            provides_intraday = True

            @classmethod
            def from_args(cls, args):
                """构造实例；本测试不会真正调用。"""
                return cls()

            def metadata(self):
                """返回空元数据；本测试不会真正调用。"""
                return {}

            def list_symbols(self, limit):
                """返回空证券池；本测试不会真正调用。"""
                return []

            def load_bars(self, codes, start, end):
                """返回空行情；本测试不会真正调用。"""
                return pd.DataFrame()

            def build_features(self, bars, feature_columns, cache_dir, factor_expressions):
                """返回空因子表；本测试不会真正调用。"""
                return pd.DataFrame()

        with self.assertRaises(ValueError):
            register_data_source(Duplicate)

    def test_unknown_source_raises(self) -> None:
        """未知数据源名应当报错并列出可选值。"""
        with self.assertRaises(ValueError):
            data_source_from_args(SimpleNamespace(data_source="nope"))

    def test_two_phase_parsing_registers_only_selected_arguments(self) -> None:
        """先解析数据源名，再只注册该数据源的专属参数。"""
        parser = argparse.ArgumentParser(add_help=False)
        add_data_source_selection_argument(parser)
        add_selected_data_source_arguments(parser, "market_service")
        args = parser.parse_args(["--data-source", "market_service"])
        self.assertTrue(hasattr(args, "database"))
        self.assertFalse(hasattr(args, "adjust"))

        other = argparse.ArgumentParser(add_help=False)
        add_data_source_selection_argument(other)
        add_selected_data_source_arguments(other, "qmt_daily")
        parsed = other.parse_args(["--data-source", "qmt_daily", "--adjust", "none"])
        self.assertEqual(parsed.adjust, "none")
        self.assertFalse(hasattr(parsed, "database"))

    def test_argument_names_expose_per_source_fields(self) -> None:
        """能查出各数据源的专属参数名，供 YAML 忽略无关键使用。"""
        self.assertIn("database", data_source_argument_names("market_service"))
        self.assertIn("adjust", data_source_argument_names("qmt_daily"))
        self.assertEqual(data_source_argument_names("nope"), set())


class IntradayCapabilityTest(unittest.TestCase):
    """分钟因子的标记必须与源码实际用法一致。"""

    def test_marked_set_matches_intraday_values_usage(self) -> None:
        """标了 requires_intraday 的因子集合，应恰好等于源码里调用
        ``intraday_values`` 的那些因子。"""
        import inspect

        actual = set()
        for name, factory in FACTOR_FACTORIES.items():
            source = Path(inspect.getfile(factory.__class__)).read_text(encoding="utf-8")
            if "intraday_values(" in source:
                actual.add(name)
        self.assertEqual(actual, set(intraday_factor_names()))

    def test_daily_capable_is_the_complement(self) -> None:
        """日频可算集合应当正好是全部因子减去分钟因子。"""
        self.assertEqual(
            set(FACTOR_FACTORIES) - set(intraday_factor_names()),
            set(daily_capable_factor_names()),
        )
        self.assertTrue(requires_intraday(FACTOR_FACTORIES["realized_vol"]))
        self.assertFalse(requires_intraday(FACTOR_FACTORIES["return_1d"]))


class DailyFeatureBuildTest(unittest.TestCase):
    """日频因子构建路径的行为。"""

    def test_default_set_skips_intraday_factors(self) -> None:
        """不指定因子时自动跳过分钟因子，结果里不应出现它们。"""
        daily = build_daily_features_from_daily(_daily_bars(), source_name="qmt_daily")
        for name in intraday_factor_names():
            self.assertNotIn(name, daily.columns)
        self.assertIn("return_1d", daily.columns)

    def test_explicit_intraday_factor_raises(self) -> None:
        """显式点名分钟因子必须报错，而不是静默返回全 NaN。"""
        with self.assertRaises(ValueError) as context:
            build_daily_features_from_daily(
                _daily_bars(), feature_columns=["return_1d", "realized_vol"],
                source_name="qmt_daily",
            )
        self.assertIn("realized_vol", str(context.exception))

    def test_daily_values_match_manual_calculation(self) -> None:
        """日频路径算出的 return_1d 应与手工计算完全一致。"""
        bars = _daily_bars()
        daily = build_daily_features_from_daily(
            bars, feature_columns=["return_1d"], source_name="qmt_daily"
        )
        subset = daily[daily["code"] == "000001.SZ"].sort_values("trade_date")
        expected = subset["close"].iloc[1] / subset["close"].iloc[0] - 1.0
        self.assertAlmostEqual(float(subset["return_1d"].iloc[1]), expected, places=12)

    def test_groups_do_not_leak_across_codes(self) -> None:
        """每只证券的首个交易日没有前值，收益率必须是缺失。"""
        daily = build_daily_features_from_daily(
            _daily_bars(), feature_columns=["return_1d"], source_name="qmt_daily"
        )
        first = daily.sort_values(["code", "trade_date"]).groupby("code").head(1)
        self.assertTrue(first["return_1d"].isna().all())


class SourceInjectionTest(unittest.TestCase):
    """第三方数据源必须能通过注册表注入主实验，而无需改动实验核心流程。"""

    def test_custom_source_drives_main_without_touching_core(self) -> None:
        """注册一个全新数据源后，主流程应当原样跑通并用上它的数据。"""
        from contextlib import ExitStack
        from datetime import datetime
        from unittest.mock import patch

        from quant.factor_research.data_sources import registry
        from quant.factor_research.factors import build_daily_features_from_daily

        class FixtureSource(MarketDataSource):
            """只返回内存里固定行情的测试数据源。"""

            name = "unit_test_fixture"
            frequency = "1d"
            provides_intraday = False
            loaded: list[str] = []

            @classmethod
            def from_args(cls, args):
                """按命令行参数构建实例；本数据源没有专属参数。"""
                return cls()

            def metadata(self):
                """返回覆盖固定行情区间的元数据。"""
                return {"first_time": "2024-01-02", "last_time": "2024-01-26"}

            def list_symbols(self, limit):
                """返回固定的两只测试证券。"""
                return ["000001.SZ", "600000.SH"][:limit]

            def load_bars(self, codes, start, end):
                """返回内存里预置的日频行情，并记录被请求的证券。"""
                type(self).loaded = list(codes)
                return _daily_bars()

            def build_features(self, bars, feature_columns, cache_dir, factor_expressions):
                """走公共的日频因子构建入口。"""
                return build_daily_features_from_daily(
                    bars,
                    feature_columns=feature_columns,
                    cache_dir=cache_dir,
                    factor_expressions=factor_expressions,
                    source_name=self.name,
                )

        register_data_source(FixtureSource)
        self.addCleanup(registry.DATA_SOURCE_TYPES.pop, FixtureSource.name, None)
        try:
            self.assertIn("unit_test_fixture", available_data_sources())

            source = data_source_from_args(SimpleNamespace(data_source="unit_test_fixture"))
            self.assertIsInstance(source, FixtureSource)

            # 主流程只依赖抽象：拿元数据、取证券池、装载行情、构建因子。
            from quant.cli import factor_demo

            args = SimpleNamespace(
                data_source="unit_test_fixture",
                start="2024-01-02", end="2024-01-26",
                symbol_limit=2, codes=None,
                factors=["return_1d"], factor_expressions=[],
                no_factor_cache=True, factor_cache_dir=Path(".factor_cache"),
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        factor_demo, "resolve_window",
                        return_value=(datetime(2024, 1, 2), datetime(2024, 1, 26)),
                    )
                )
                metadata = source.metadata()
                start, end = factor_demo.resolve_window(metadata, args.start, args.end)
            codes = args.codes or source.list_symbols(args.symbol_limit)
            bars = source.load_bars(codes, start, end)
            daily = source.build_features(
                bars, feature_columns=args.factors, cache_dir=None, factor_expressions=[]
            )
        finally:
            pass

        self.assertEqual(FixtureSource.loaded, ["000001.SZ", "600000.SH"])
        self.assertIn("return_1d", daily.columns)
        self.assertEqual(source.cache_namespace(), "")


class CacheNamespaceTest(unittest.TestCase):
    """因子缓存必须按数据源与复权口径隔离。"""

    def test_default_namespace_keeps_legacy_path(self) -> None:
        """不传命名空间时缓存根目录必须与历史行为逐字节一致。"""
        self.assertEqual(FactorCache(".factor_cache").root, Path(".factor_cache"))

    def test_namespace_isolates_cache_root(self) -> None:
        """不同口径的缓存应落在不同目录下。"""
        hfq = FactorCache(".factor_cache", "qmt_daily/hfq/-/no_susp").root
        none = FactorCache(".factor_cache", "qmt_daily/none/-/no_susp").root
        self.assertNotEqual(hfq, none)
        self.assertTrue(str(hfq).startswith(str(Path(".factor_cache"))))

    def test_daily_and_intraday_paths_never_collide(self) -> None:
        """同一因子在日频与分钟路径上的缓存文件名必须不同。"""
        cache = FactorCache(".factor_cache")
        factory = FACTOR_FACTORIES["return_1d"]
        intraday_path = cache._path(factory, "abc", daily_mode=False)
        daily_path = cache._path(factory, "abc", daily_mode=True)
        self.assertNotEqual(intraday_path, daily_path)

    def test_adjust_mode_enters_input_fingerprint(self) -> None:
        """复权口径不同时输入指纹必须不同，否则缓存会串味。"""
        from quant.factor_research.factors import _daily_input_fingerprint, _prepare_daily_bars

        prepared = _prepare_daily_bars(_daily_bars())
        self.assertNotEqual(
            _daily_input_fingerprint(prepared, "qmt_daily/hfq"),
            _daily_input_fingerprint(prepared, "qmt_daily/none"),
        )


if __name__ == "__main__":
    unittest.main()
