"""校验因子对价格尺度的齐次度，防止复权基准变化悄悄改变因子取值。

后复权系数是**逐证券**的：同一天不同证券乘上的常数各不相同。因此扰动必须
按证券取不同的缩放系数，绝不能全截面统一乘一个常数——统一缩放下横截面
排名与 z-score 恒等不变，会给带价格量纲的因子发一张假的合格证，而那正是
需要被抓出来的情况（见 ``test_cross_sectional_rank_of_price_is_not_scale_invariant``）。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.factor_research.factor_dsl import DailyFactorFrame
from quant.factor_research.factor_factories import FACTOR_FACTORIES
from quant.factor_research.factor_factories.capabilities import (
    daily_capable_factor_names,
    dimensional_factor_names,
    intraday_factor_names,
    non_homogeneous_factor_names,
    price_homogeneity,
)
from quant.factor_research.factors import build_daily_features

#: 参与缩放的价格列。成交量与成交额刻意保持原值：复权只作用于价格，
#: 混用复权价与未复权量的因子会因此暴露出来。
_PRICE_COLUMNS = ("open", "high", "low", "close")

#: 逐证券缩放系数取值表，按几何级数铺开而不是随机抽样：真实后复权系数跨越
#: 一到两个数量级（老牌高分红股可达数十，次新股为 1.0），系数挤在同一量级时
#: 横截面对比度不足，会让本该失败的用例侥幸通过。含小于 1 的档位以覆盖缩股
#: 方向。**顺序刻意打乱**：测试样本的价位随证券序号递增，若系数也单调递增，
#: 缩放只会放大原有的横截面顺序而不打乱它，横截面排名类用例就会失去意义。
_SCALE_LADDER = (12.0, 0.3, 50.0, 0.9, 3.0)

#: 固定随机种子，保证样本逐次可复现。
_SEED = 20260816

#: 样本日收益率的对数标准差。alpha_001 的非齐次翻转门槛正比于它，因此两处共用
#: 同一个常量，改动样本波动率时门槛会自动跟着变。
_DAILY_SIGMA = 0.02


def _daily_frame(days: int = 60, codes: tuple[str, ...] = ("A", "B", "C", "D", "E")) -> pd.DataFrame:
    """生成多证券随机游走日频样本，长度足以填满 20 日滚动窗口。

    参数：
        days: 每只证券生成的连续工作日数，缺省 60，需大于最长因子窗口 20 加
            上其自身的一阶差分，否则部分因子整列为缺失值而失去校验意义。
        codes: 要生成的证券代码及其排列顺序；不同证券起始价位刻意拉开，
            以便横截面上的价格水平差异足够明显。
    """

    rng = np.random.default_rng(_SEED)
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for index, code in enumerate(codes):
        level = 8.0 + index * 7.0
        for date in dates:
            level *= float(np.exp(rng.normal(0.0, _DAILY_SIGMA)))
            open_price = level * float(np.exp(rng.normal(0.0, 0.005)))
            close_price = level * float(np.exp(rng.normal(0.0, 0.005)))
            rows.append(
                {
                    "code": code,
                    "trade_date": date,
                    "open": open_price,
                    "high": max(open_price, close_price) * (1.0 + abs(rng.normal(0.0, 0.004))),
                    "low": min(open_price, close_price) * (1.0 - abs(rng.normal(0.0, 0.004))),
                    "close": close_price,
                    "volume": float(rng.integers(1_000, 50_000)),
                }
            )
    return pd.DataFrame(rows)


def _minute_frame(
    days: int = 6,
    bars_per_day: int = 8,
    codes: tuple[str, ...] = ("A", "B", "C"),
) -> pd.DataFrame:
    """生成多证券分钟行情样本，供依赖日内行情的因子使用。

    参数：
        days: 每只证券生成的连续工作日数，缺省 6。
        bars_per_day: 每个交易日的分钟线根数，缺省 8，需大于 6 以便末 30 分钟
            类因子取到的尾部窗口与全天窗口不同。
        codes: 要生成的证券代码及其排列顺序。
    """

    rng = np.random.default_rng(_SEED + 1)
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for index, code in enumerate(codes):
        level = 8.0 + index * 7.0
        for date in dates:
            for bar in range(bars_per_day):
                open_price = level
                level *= float(np.exp(rng.normal(0.0, 0.003)))
                close_price = level
                rows.append(
                    {
                        "code": code,
                        "trade_time": date + pd.Timedelta(hours=9, minutes=35 + 5 * bar),
                        "open": open_price,
                        "high": max(open_price, close_price) * 1.001,
                        "low": min(open_price, close_price) * 0.999,
                        "close": close_price,
                        "volume": float(rng.integers(100, 5_000)),
                    }
                )
    return pd.DataFrame(rows)


def _code_scales(codes) -> pd.Series:
    """给每只证券分配一个确定的正缩放系数，模拟逐证券不同的后复权基准。

    参数：
        codes: 证券代码可迭代对象；重复项会被折叠成一个系数，顺序决定各证券
            取到 ``_SCALE_LADDER`` 中的哪一档，证券多于档数时循环取用。

    返回：
        以证券代码为索引的浮点系数。刻意不用随机抽样：随机值容易挤在同一量级，
        使横截面对比度依赖种子运气。
    """

    unique = list(dict.fromkeys(str(code) for code in codes))
    values = [_SCALE_LADDER[index % len(_SCALE_LADDER)] for index in range(len(unique))]
    return pd.Series(values, index=unique, dtype=float)


def _rescale_prices(frame: pd.DataFrame, scales: pd.Series) -> pd.DataFrame:
    """按证券给价格列各乘一个常数，等价于逐证券更换复权基准。

    参数：
        frame: 日频或分钟频行情表，至少含 ``code`` 与 ``_PRICE_COLUMNS``。
        scales: 以证券代码为索引的缩放系数。

    返回：
        仅价格列被缩放的新表；成交量等非价格列逐字节保持原值。
    """

    result = frame.copy()
    multiplier = result["code"].astype(str).map(scales).astype(float)
    for column in _PRICE_COLUMNS:
        result[column] = result[column].astype(float) * multiplier
    return result


def _assert_close(actual: np.ndarray, expected: np.ndarray, name: str, degree: int) -> None:
    """断言缩放后的因子值与解析换算值一致，容差随取值量级放宽。

    参数：
        actual: 缩放价格后重新计算出的因子值，已剔除非有限值。
        expected: 由原始因子值乘 ``c**k`` 解析换算出的期望值，与 ``actual`` 等长。
        name: 因子名，仅用于失败信息。
        degree: 该因子声明的齐次度，仅用于失败信息。

    说明：
        ``(a - b) / (c - d)`` 形态的因子在分子接近零时相对误差会被放大，纯 ``rtol``
        余量不足；这里按整列最大绝对值给一个 ``atol``，仍比「齐次度声明错误」导致
        的数量级差异低若干个数量级，不削弱检出能力。
    """

    tolerance = 1e-10 * float(np.max(np.abs(expected))) if expected.size else 0.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-9,
        atol=tolerance,
        err_msg=f"因子 {name} 不满足声明的齐次度 k={degree}",
    )


class DailyFactorScaleInvarianceTest(unittest.TestCase):
    """逐证券缩放价格后，日频因子必须严格满足自己声明的齐次度。"""

    def setUp(self) -> None:
        self.daily = _daily_frame()
        self.scales = _code_scales(self.daily["code"])
        self.scaled = _rescale_prices(self.daily, self.scales)
        self.row_scale = self.daily["code"].astype(str).map(self.scales).astype(float).to_numpy()

    def test_scales_spread_across_orders_of_magnitude(self):
        # 守住扰动强度本身：系数挤在同一量级时，横截面对比度不足，
        # 带价格量纲的因子可能侥幸通过下面的用例。
        self.assertGreater(self.scales.max() / self.scales.min(), 10.0)
        self.assertLess(self.scales.min(), 1.0)
        self.assertGreater(self.scales.max(), 1.0)

    def _values(self, name: str, frame: pd.DataFrame) -> np.ndarray:
        """计算单个日频因子并转成浮点数组。

        参数：
            name: 已注册的因子名。
            frame: 传给因子的日频行情表。
        """

        return pd.Series(FACTOR_FACTORIES[name].compute(None, frame)).astype(float).to_numpy()

    def test_daily_factors_match_declared_homogeneity(self):
        for name in sorted(daily_capable_factor_names()):
            degree = price_homogeneity(FACTOR_FACTORIES[name])
            if degree is None:
                # 非齐次因子没有可断言的解析形式，改由下面的专项用例守护。
                continue
            with self.subTest(factor=name, k=degree):
                base = self._values(name, self.daily)
                scaled = self._values(name, self.scaled)
                expected = base * self.row_scale**degree
                # 只按 expected 取 mask：若缩放后才出现 NaN 或 ±inf，必须让断言
                # 失败，而不是把这些行悄悄剔除。
                mask = np.isfinite(expected)
                self.assertGreater(mask.sum(), 0, f"因子 {name} 没有可比较的有效值")
                _assert_close(scaled[mask], expected[mask], name, degree)

    def test_daily_finite_pattern_is_scale_stable(self):
        # 有限值位置随缩放变化，说明因子里藏着绝对价格阈值（如 close < 5.0）。
        # 这类因子不满足任何齐次度，且数值断言会因为 NaN 被 mask 掉而漏判。
        # 比较 isfinite 而不是 isnan，才能同时覆盖缩放后新冒出的 ±inf。
        for name in sorted(daily_capable_factor_names()):
            with self.subTest(factor=name):
                base = np.isfinite(self._values(name, self.daily))
                scaled = np.isfinite(self._values(name, self.scaled))
                self.assertTrue(
                    bool((base == scaled).all()),
                    f"因子 {name} 缩放价格后有限值位置发生变化，疑似含绝对价格阈值",
                )

    def test_non_homogeneous_declaration_is_justified(self):
        # 锁定 alpha_001 的 price_homogeneity = None 不是保守标注：把价格缩到
        # 与 20 日收益率波动率同量级时，SignedPower 的比较结果确实会翻转。
        # 翻转门槛约为「日收益率波动率 / 价格水平」，这里按本样本的最高价推出
        # 一个足够小的系数，避免写死的魔数在样本价位变化后失效。
        self.assertIsNone(price_homogeneity(FACTOR_FACTORIES["alpha_001"]))
        base = self._values("alpha_001", self.daily)
        threshold = _DAILY_SIGMA / float(self.daily["close"].max()) / 10.0
        tiny = pd.Series(threshold, index=self.daily["code"].astype(str).unique())
        scaled = self._values("alpha_001", _rescale_prices(self.daily, tiny))
        mask = np.isfinite(base) & np.isfinite(scaled)
        self.assertGreater(mask.sum(), 0)
        self.assertTrue(
            bool((base[mask] != scaled[mask]).any()),
            "alpha_001 在极端缩放下仍不变，应重新评估它能否声明为 k=0",
        )


class IntradayFactorScaleInvarianceTest(unittest.TestCase):
    """分钟路径的因子同样必须满足声明的齐次度。"""

    def test_intraday_factors_match_declared_homogeneity(self):
        bars = _minute_frame()
        scales = _code_scales(bars["code"])
        names = sorted(intraday_factor_names())
        self.assertTrue(names, "没有依赖日内行情的因子，用例失去意义")

        base = build_daily_features(bars, names)
        scaled = build_daily_features(_rescale_prices(bars, scales), names)
        # build_daily_features 按 (trade_date, code) 排序输出，两次调用行序一致。
        pd.testing.assert_frame_equal(base[["code", "trade_date"]], scaled[["code", "trade_date"]])
        row_scale = base["code"].astype(str).map(scales).astype(float).to_numpy()

        for name in names:
            degree = price_homogeneity(FACTOR_FACTORIES[name])
            with self.subTest(factor=name, k=degree):
                base_values = base[name].astype(float).to_numpy()
                scaled_values = scaled[name].astype(float).to_numpy()
                # 有限值位置的检查对非齐次因子同样适用，因此放在 degree 分支之前。
                self.assertTrue(
                    bool((np.isfinite(base_values) == np.isfinite(scaled_values)).all()),
                    f"因子 {name} 缩放价格后有限值位置发生变化，疑似含绝对价格阈值",
                )
                if degree is None:
                    continue
                expected = base_values * row_scale**degree
                mask = np.isfinite(expected)
                self.assertGreater(mask.sum(), 0, f"因子 {name} 没有可比较的有效值")
                _assert_close(scaled_values[mask], expected[mask], name, degree)


class PriceHomogeneityDeclarationTest(unittest.TestCase):
    """``price_homogeneity`` 的读取契约与注册表汇总。"""

    def test_undeclared_factory_defaults_to_scale_invariant(self):
        class Undeclared:
            name = "undeclared"

        self.assertEqual(price_homogeneity(Undeclared), 0)

    def test_declared_integer_and_none_are_accepted(self):
        class Level:
            name = "level"
            price_homogeneity = 1

        class Mixed:
            name = "mixed"
            price_homogeneity = None

        self.assertEqual(price_homogeneity(Level), 1)
        self.assertIsNone(price_homogeneity(Mixed))

    def test_invalid_declaration_is_rejected(self):
        class Floating:
            name = "floating"
            price_homogeneity = 1.0

        class Boolean:
            name = "boolean"
            # bool 是 int 的子类，必须单独拦截，避免 True 被误读成 k = 1。
            price_homogeneity = True

        for factory in (Floating, Boolean):
            with self.subTest(factory=factory.name):
                with self.assertRaises(TypeError):
                    price_homogeneity(factory)

    def test_registry_summaries_match_current_declarations(self):
        # 当前没有任何带价格量纲的因子；alpha_001 是唯一混用了价格与收益率量纲
        # 的非齐次因子。新增尺度相关因子时这里会失败，提醒同步评估它能否直接
        # 进入横截面比较。两个集合按定义互斥。
        self.assertEqual(dimensional_factor_names(), frozenset())
        self.assertEqual(non_homogeneous_factor_names(), frozenset({"alpha_001"}))
        self.assertFalse(dimensional_factor_names() & non_homogeneous_factor_names())

    def test_base_class_does_not_declare_price_homogeneity(self):
        # 属性一旦上提到基类，FactorCache 的实现指纹会作废全部因子的历史缓存。
        from quant.factor_research.factor_factories.base import FactorFactory

        self.assertNotIn("price_homogeneity", vars(FactorFactory))


class DslExpressionScaleInvarianceTest(unittest.TestCase):
    """DSL 候选表达式同样分尺度无关与尺度相关两类。"""

    def setUp(self) -> None:
        self.daily = _daily_frame(days=30, codes=("A", "B", "C"))
        self.scales = _code_scales(self.daily["code"])
        self.scaled = _rescale_prices(self.daily, self.scales)

    def test_ratio_expression_is_scale_invariant(self):
        def ratio(frame: pd.DataFrame) -> np.ndarray:
            handle = DailyFactorFrame(frame)
            return (handle.close() / handle.close().mean(5)).compute().astype(float).to_numpy()

        base, scaled = ratio(self.daily), ratio(self.scaled)
        mask = np.isfinite(base) & np.isfinite(scaled)
        self.assertGreater(mask.sum(), 0)
        np.testing.assert_allclose(scaled[mask], base[mask], rtol=1e-9)

    def test_cross_sectional_rank_of_price_is_not_scale_invariant(self):
        # 这是全截面统一缩放抓不到、必须逐证券缩放才能暴露的那一类：
        # 横截面排名对「所有证券乘同一个常数」恒等不变，却会被逐证券的
        # 后复权系数彻底打乱。搜索产生的候选表达式必须按此筛掉。
        def ranked(frame: pd.DataFrame) -> np.ndarray:
            return DailyFactorFrame(frame).close().rank().compute().astype(float).to_numpy()

        base, scaled = ranked(self.daily), ranked(self.scaled)
        mask = np.isfinite(base) & np.isfinite(scaled)
        self.assertGreater(mask.sum(), 0)
        # 先确认用例前提成立：缩放系数必须真的打乱横截面价格顺序，否则本用例
        # 只是在验证一个恒等式。系数与样本价位同向单调时就会出现这种空转。
        self.assertTrue(
            bool((base[mask] != scaled[mask]).any()),
            "逐证券缩放没有改变横截面排名，_SCALE_LADDER 与样本价位可能同向单调",
        )

        # 同一个表达式在全截面统一缩放下完全不变，正是假合格证的来源。
        uniform = pd.Series(3.0, index=self.daily["code"].astype(str).unique())
        uniform_scaled = ranked(_rescale_prices(self.daily, uniform))
        np.testing.assert_allclose(uniform_scaled[mask], base[mask], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
