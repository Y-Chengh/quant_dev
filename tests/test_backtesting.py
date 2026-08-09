from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from factor_research.backtesting import (
    TRADING_DAYS_PER_YEAR,
    TopNBacktestResult,
    run_top_n_intraday_backtest,
)
from factor_research.experiment import ExperimentResult
from factor_research.reporting import (
    render_equity_curve_svg,
    write_evaluation_report,
)
from run_factor_demo import parse_args, resolve_equity_chart_path


class TopNIntradayBacktestTest(unittest.TestCase):
    """验证 Top N 日内回测的选股、成本、指标和报告输出。"""

    def _predictions(self) -> pd.DataFrame:
        """构造两个交易日、每日三个证券的确定性预测样本。

        返回：
            含日期、代码、实际收益、模型分数和方向标签的预测表。
        """

        return pd.DataFrame(
            {
                "target_date": pd.to_datetime(
                    ["2025-01-02"] * 3 + ["2025-01-03"] * 3
                ),
                "code": ["C", "A", "B", "A", "B", "C"],
                "target_return": [-0.05, 0.10, 0.00, -0.02, 0.10, 0.02],
                "up_probability": [0.1, 0.9, 0.8, 0.7, 0.1, 0.8],
                "label": [0, 1, 0, 0, 1, 1],
                "prediction": [0, 1, 1, 1, 0, 1],
            }
        )

    def test_selects_daily_top_n_and_applies_two_sided_costs_exactly(self) -> None:
        """每日应独立选股，并按买卖双边口径精确扣除滑点和手续费。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        result = run_top_n_intraday_backtest(
            self._predictions(),
            "up_probability",
            top_n=2,
            slippage_bps=10.0,
            commission_bps=5.0,
        )

        multiplier = (0.999 * 0.9995) / (1.001 * 1.0005)
        expected_gross = np.array([0.05, 0.0])
        expected_net = (1.0 + expected_gross) * multiplier - 1.0
        np.testing.assert_array_equal(result.daily_returns["selected_count"], [2, 2])
        np.testing.assert_allclose(
            result.daily_returns["gross_return"], expected_gross
        )
        np.testing.assert_allclose(result.daily_returns["net_return"], expected_net)
        np.testing.assert_allclose(
            result.daily_returns["equity"], np.cumprod(1.0 + expected_net)
        )
        self.assertEqual(
            list(result.top_selections.columns),
            [
                "target_date",
                "top_rank",
                "code",
                "predicted_value",
                "actual_return",
                "previous_actual_return",
            ],
        )
        self.assertEqual(
            result.top_selections["code"].tolist(), ["A", "B", "C", "A"]
        )
        np.testing.assert_array_equal(
            result.top_selections["top_rank"], [1, 2, 1, 2]
        )
        np.testing.assert_allclose(
            result.top_selections["predicted_value"], [0.9, 0.8, 0.8, 0.7]
        )
        np.testing.assert_allclose(
            result.top_selections["actual_return"], [0.10, 0.00, 0.02, -0.02]
        )
        np.testing.assert_allclose(
            result.top_selections["previous_actual_return"],
            [np.nan, np.nan, -0.05, 0.10],
            equal_nan=True,
        )
        expected_control_gross = {
            "universe": np.array([1 / 60, 1 / 30]),
            "bottom": np.array([-0.025, 0.04]),
            "mid": expected_gross,
        }
        for prefix, gross_returns in expected_control_gross.items():
            expected_control_net = (1.0 + gross_returns) * multiplier - 1.0
            np.testing.assert_allclose(
                result.daily_returns[f"{prefix}_net_return"],
                expected_control_net,
            )
            np.testing.assert_allclose(
                result.daily_returns[f"{prefix}_equity"],
                np.cumprod(1.0 + expected_control_net),
            )
        expected_sharpe = (
            np.sqrt(TRADING_DAYS_PER_YEAR)
            * expected_net.mean()
            / expected_net.std(ddof=1)
        )
        self.assertAlmostEqual(result.metrics["sharpe_ratio"], expected_sharpe)

    def test_ties_are_resolved_by_code_and_invalid_rows_are_excluded(self) -> None:
        """同分时应按代码稳定选取，非有限分数不得进入当日组合。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02"] * 3),
                "code": ["B", "A", "C"],
                "target_return": [0.20, 0.10, 9.0],
                "predicted_return": [0.5, 0.5, np.nan],
            }
        )
        result = run_top_n_intraday_backtest(
            predictions, "predicted_return", top_n=1
        )

        self.assertAlmostEqual(result.daily_returns.loc[0, "gross_return"], 0.10)
        self.assertEqual(result.daily_returns.loc[0, "selected_count"], 1)
        self.assertTrue(np.isnan(result.metrics["sharpe_ratio"]))

    def test_previous_return_keeps_history_with_invalid_previous_score(self) -> None:
        """前日模型分数无效时，其有限实际收益仍应进入次日 Top N 明细。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "code": ["A", "A"],
                "target_return": [0.03, -0.02],
                "predicted_return": [np.nan, 0.01],
            }
        )

        result = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=1,
            random_simulations=20,
        )

        self.assertEqual(len(result.top_selections), 1)
        self.assertAlmostEqual(
            result.top_selections.loc[0, "previous_actual_return"], 0.03
        )

    def test_duplicate_history_uses_highest_score_and_rejects_tied_conflicts(
        self,
    ) -> None:
        """重复证券日期应稳定采用最高分收益，并拒绝最高分并列的冲突收益。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(
                    ["2025-01-02", "2025-01-02", "2025-01-03"]
                ),
                "code": ["A", "A", "A"],
                "target_return": [0.01, 0.04, -0.02],
                "predicted_return": [0.2, 0.8, 0.9],
            }
        )
        expected_previous = []
        for frame in (predictions, predictions.iloc[[1, 0, 2]].reset_index(drop=True)):
            result = run_top_n_intraday_backtest(
                frame,
                "predicted_return",
                top_n=1,
                random_simulations=20,
            )
            expected_previous.append(
                result.top_selections.loc[
                    result.top_selections["target_date"]
                    == pd.Timestamp("2025-01-03"),
                    "previous_actual_return",
                ].item()
            )
        np.testing.assert_allclose(expected_previous, [0.04, 0.04])

        conflicting = predictions.copy()
        conflicting.loc[:1, "predicted_return"] = 0.8
        with self.assertRaisesRegex(ValueError, "冲突实际收益"):
            run_top_n_intraday_backtest(
                conflicting,
                "predicted_return",
                top_n=1,
                random_simulations=20,
            )

    def test_cross_sectional_controls_match_exact_returns_and_are_reproducible(self) -> None:
        """等权、价差、十分位、Top 对照和随机基准应使用约定口径。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        rows = []
        for target_date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
            for value in range(10):
                rows.append(
                    {
                        "target_date": target_date,
                        "code": f"S{value:02d}",
                        "target_return": value / 1_000,
                        "predicted_return": float(value),
                    }
                )
        predictions = pd.DataFrame(rows)
        result = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=2,
            random_simulations=200,
            random_seed=7,
        )
        repeated = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=2,
            random_simulations=200,
            random_seed=7,
        )

        expected_equal_weight = (1.0 + 0.0045) ** 2 - 1.0
        self.assertAlmostEqual(
            result.benchmark_metrics["equal_weight_total_return"],
            expected_equal_weight,
        )
        self.assertAlmostEqual(
            result.relative_metrics["top_minus_universe_annualized_return"],
            0.004 * TRADING_DAYS_PER_YEAR,
        )
        self.assertAlmostEqual(
            result.relative_metrics["top_minus_bottom_annualized_return"],
            0.008 * TRADING_DAYS_PER_YEAR,
        )
        np.testing.assert_allclose(result.daily_returns["bottom_net_return"], 0.0005)
        np.testing.assert_allclose(result.daily_returns["mid_net_return"], 0.0045)
        np.testing.assert_allclose(
            result.daily_returns["bottom_equity"],
            np.cumprod(np.full(2, 1.0005)),
        )
        np.testing.assert_allclose(
            result.daily_returns["mid_equity"],
            np.cumprod(np.full(2, 1.0045)),
        )
        top_decile = result.decile_returns.set_index("predicted_decile").loc[10]
        self.assertEqual(top_decile["samples"], 2)
        self.assertAlmostEqual(top_decile["average_return"], 0.009)
        self.assertEqual(result.selection_metrics["top_samples"], 4.0)
        self.assertAlmostEqual(result.selection_metrics["top_hit_rate"], 1.0)
        self.assertAlmostEqual(result.selection_metrics["top_average_return"], 0.0085)
        for key in (
            "random_annualized_p05",
            "random_annualized_median",
            "random_annualized_p95",
            "strategy_random_percentile",
        ):
            self.assertEqual(
                result.benchmark_metrics[key], repeated.benchmark_metrics[key]
            )
        different_seed = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=2,
            random_simulations=200,
            random_seed=8,
        )
        self.assertNotEqual(
            result.benchmark_metrics["random_annualized_median"],
            different_seed.benchmark_metrics["random_annualized_median"],
        )

    def test_small_cross_section_and_duplicate_codes_keep_exact_selection(self) -> None:
        """小截面应标明最高十分位，重复代码不得扩张 Top N 样本。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02"] * 3),
                "code": ["A", "A", "B"],
                "target_return": [0.10, -0.90, 0.00],
                "predicted_return": [0.9, 0.8, 0.7],
            }
        )
        result = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=1,
            random_simulations=20,
        )

        self.assertEqual(
            result.decile_returns["predicted_decile"].tolist(), [1, 5, 10]
        )
        self.assertEqual(result.selection_metrics["top_samples"], 1.0)
        self.assertEqual(result.selection_metrics["other_samples"], 2.0)
        self.assertAlmostEqual(result.selection_metrics["top_average_return"], 0.10)

    def test_decile_returns_weight_each_trading_day_equally(self) -> None:
        """十分位收益应先做日内等权，再对不同规模的交易日等权平均。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        rows = []
        for target_date, size in (
            (pd.Timestamp("2025-01-02"), 10),
            (pd.Timestamp("2025-01-03"), 20),
        ):
            for value in range(size):
                rows.append(
                    {
                        "target_date": target_date,
                        "code": f"{target_date.date()}-{value:02d}",
                        "target_return": (
                            1.0
                            if target_date == pd.Timestamp("2025-01-02")
                            and value == size - 1
                            else 0.0
                        ),
                        "predicted_return": float(value),
                    }
                )
        result = run_top_n_intraday_backtest(
            pd.DataFrame(rows),
            "predicted_return",
            top_n=1,
            random_simulations=20,
        )

        top_decile = result.decile_returns.set_index("predicted_decile").loc[10]
        self.assertEqual(top_decile["samples"], 3)
        self.assertEqual(top_decile["trading_days"], 2)
        self.assertAlmostEqual(top_decile["average_return"], 0.5)

    def test_full_universe_selection_matches_benchmarks_and_old_constructor(self) -> None:
        """全选时随机及等权基准应等于策略，旧结果构造接口仍应可用。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = self._predictions().loc[
            lambda frame: frame["target_date"] == pd.Timestamp("2025-01-02")
        ]
        result = run_top_n_intraday_backtest(
            predictions,
            "up_probability",
            top_n=3,
            random_simulations=20,
        )
        self.assertAlmostEqual(
            result.metrics["annualized_return"],
            result.benchmark_metrics["equal_weight_annualized_return"],
        )
        self.assertAlmostEqual(
            result.metrics["annualized_return"],
            result.benchmark_metrics["random_annualized_median"],
        )
        self.assertEqual(result.selection_metrics["other_samples"], 0.0)
        self.assertTrue(np.isnan(result.selection_metrics["other_average_return"]))

        legacy = TopNBacktestResult(
            top_n=1,
            score_column="score",
            slippage_bps=0.0,
            commission_bps=0.0,
            daily_returns=pd.DataFrame(),
            metrics={},
        )
        self.assertEqual(legacy.benchmark_metrics, {})
        self.assertTrue(legacy.decile_returns.empty)
        self.assertTrue(legacy.top_selections.empty)

    def test_zero_final_equity_has_undefined_random_annualized_metrics(self) -> None:
        """所有随机路径净值归零时分位数应为缺失值而不是抛出异常。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02"]),
                "code": ["A"],
                "target_return": [-1.0],
                "predicted_return": [0.0],
            }
        )
        result = run_top_n_intraday_backtest(
            predictions,
            "predicted_return",
            top_n=1,
            random_simulations=20,
        )

        self.assertAlmostEqual(result.metrics["total_return"], -1.0)
        self.assertTrue(np.isnan(result.metrics["annualized_return"]))
        for key in (
            "random_annualized_p05",
            "random_annualized_median",
            "random_annualized_p95",
            "strategy_random_percentile",
        ):
            self.assertTrue(np.isnan(result.benchmark_metrics[key]))

    def test_invalid_realized_return_cannot_change_top_n_selection(self) -> None:
        """已选高分证券的事后收益缺失时必须失败，不得用次高分证券递补。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02"] * 2),
                "code": ["A", "B"],
                "target_return": [np.nan, 0.50],
                "up_probability": [0.9, 0.8],
            }
        )

        with self.assertRaisesRegex(ValueError, "2025-01-02:A"):
            run_top_n_intraday_backtest(
                predictions, "up_probability", top_n=1
            )

    def test_rejects_invalid_parameters_and_missing_columns(self) -> None:
        """非法成本、选股数量和缺失字段应明确报错。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = self._predictions()
        with self.assertRaisesRegex(ValueError, "top_n"):
            run_top_n_intraday_backtest(predictions, "up_probability", top_n=0)
        with self.assertRaisesRegex(ValueError, "slippage_bps"):
            run_top_n_intraday_backtest(
                predictions, "up_probability", slippage_bps=-1.0
            )
        with self.assertRaisesRegex(ValueError, "random_simulations"):
            run_top_n_intraday_backtest(
                predictions, "up_probability", random_simulations=0
            )
        with self.assertRaisesRegex(ValueError, "缺少列"):
            run_top_n_intraday_backtest(
                predictions.drop(columns="target_return"), "up_probability"
            )
        invalid_date = predictions.copy()
        invalid_date.loc[0, "target_date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "target_date"):
            run_top_n_intraday_backtest(invalid_date, "up_probability")

    def test_cli_supports_backtest_defaults_overrides_and_yaml(self) -> None:
        """Top N 和成本参数应接受默认值、命令行覆盖及 YAML 配置。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        defaults = parse_args([])
        self.assertEqual(defaults.backtest_top_n, 10)
        self.assertEqual(defaults.slippage_bps, 0.0)
        self.assertEqual(defaults.commission_bps, 0.0)
        overridden = parse_args(
            [
                "--backtest-top-n",
                "5",
                "--slippage-bps",
                "3.5",
                "--commission-bps",
                "2",
            ]
        )
        self.assertEqual(overridden.backtest_top_n, 5)
        self.assertEqual(overridden.slippage_bps, 3.5)
        self.assertEqual(overridden.commission_bps, 2.0)

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                "backtest_top_n: 7\nslippage_bps: 4\ncommission_bps: 1.5\n",
                encoding="utf-8",
            )
            configured = parse_args(["--config", str(config_path)])
        self.assertEqual(configured.backtest_top_n, 7)
        self.assertEqual(configured.slippage_bps, 4.0)
        self.assertEqual(configured.commission_bps, 1.5)
        with self.assertRaises(SystemExit):
            parse_args(["--commission-bps", "nan"])

    def test_report_contains_sharpe_and_writes_equity_curve(self) -> None:
        """评估报告应展示成本、夏普比率并生成可链接的收益曲线。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        predictions = self._predictions()
        backtest = run_top_n_intraday_backtest(
            predictions, "up_probability", top_n=2, slippage_bps=2, commission_bps=1
        )
        accuracy = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "samples": [3, 3],
                "accuracy": [1.0, 2 / 3],
                "accuracy_change": [np.nan, -1 / 3],
            }
        )
        experiment = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="test",
            feature_columns=["return_1d"],
            metrics={"samples": 6.0, "accuracy": 5 / 6},
            predictions=predictions,
            feature_importance=None,
            daily_accuracy_trend=accuracy,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "evaluation.md"
            accuracy_path = root / "evaluation_accuracy.svg"
            equity_path = resolve_equity_chart_path(accuracy_path)
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "test-run",
                {},
                backtest=backtest,
                equity_chart_path=equity_path,
            )
            report = report_path.read_text(encoding="utf-8")
            equity_svg = equity_path.read_text(encoding="utf-8")

        self.assertIn("## Top N 日内策略回测", report)
        self.assertIn("### 每日 Top N 选股明细", report)
        self.assertIn("| Top N 排名 | 2025-01-02 | 2025-01-03 |", report)
        self.assertIn(
            "`A`<br>预估：0.900000<br>实际：10.00%<br>前日：N/A",
            report,
        )
        self.assertIn(
            "`C`<br>预估：0.800000<br>实际：2.00%<br>前日：-5.00%",
            report,
        )
        self.assertIn("## Top N 与横截面对照收益曲线", report)
        self.assertIn("## Top N 基准与横截面对照", report)
        self.assertIn("## Top N 超额与多空价差", report)
        self.assertIn("## 预测分数十分位收益", report)
        self.assertIn("## Top N 与其余股票命中对照", report)
        self.assertIn("随机 Top N 年化收益率中位数", report)
        self.assertIn("| 夏普比率 |", report)
        self.assertIn("单边滑点：2.0000 bps", report)
        self.assertIn(equity_path.name, report)
        self.assertIn('<polyline class="equity"', equity_svg)
        self.assertIn('<polyline class="universe"', equity_svg)
        self.assertIn('<polyline class="bottom"', equity_svg)
        self.assertIn('<polyline class="mid"', equity_svg)
        self.assertIn(">全市场平均</text>", equity_svg)
        self.assertIn(">Bottom N</text>", equity_svg)
        self.assertIn(">Mid N</text>", equity_svg)
        self.assertIn(">期初</text>", equity_svg)

    def test_equity_renderer_accepts_legacy_top_only_frame(self) -> None:
        """旧调用方只提供 Top N 净值时仍应生成单曲线 SVG。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        legacy_daily = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "equity": [1.01, 0.99],
            }
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "legacy_equity.svg"
            render_equity_curve_svg(legacy_daily, output_path)
            equity_svg = output_path.read_text(encoding="utf-8")

        self.assertIn('<polyline class="equity"', equity_svg)
        self.assertNotIn('<polyline class="universe"', equity_svg)
        self.assertNotIn('<polyline class="bottom"', equity_svg)
        self.assertNotIn('<polyline class="mid"', equity_svg)

    def test_report_accepts_legacy_backtest_without_selection_details(self) -> None:
        """旧回测结果没有 Top N 明细列时，完整报告应兼容并显示暂无明细。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        target_date = pd.Timestamp("2025-01-02")
        predictions = pd.DataFrame(
            {
                "target_date": [target_date],
                "code": ["A"],
                "target_return": [0.01],
                "up_probability": [0.8],
                "label": [1],
                "prediction": [1],
            }
        )
        accuracy = pd.DataFrame(
            {
                "target_date": [target_date],
                "samples": [1],
                "accuracy": [1.0],
                "accuracy_change": [np.nan],
            }
        )
        experiment = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="legacy",
            feature_columns=[],
            metrics={"samples": 1.0},
            predictions=predictions,
            feature_importance=None,
            daily_accuracy_trend=accuracy,
        )
        legacy_backtest = TopNBacktestResult(
            top_n=1,
            score_column="up_probability",
            slippage_bps=0.0,
            commission_bps=0.0,
            daily_returns=pd.DataFrame(
                {"target_date": [target_date], "equity": [1.01]}
            ),
            metrics={},
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "legacy.md"
            accuracy_path = root / "legacy_accuracy.svg"
            equity_path = root / "legacy_equity.svg"
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "legacy-run",
                {},
                backtest=legacy_backtest,
                equity_chart_path=equity_path,
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("暂无 Top N 选股明细。", report)


if __name__ == "__main__":
    unittest.main()
