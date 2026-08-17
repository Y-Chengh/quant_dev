from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant.cli.factor_demo import (
    parse_args,
    resolve_drawdown_chart_path,
    resolve_equity_chart_path,
)
from quant.factor_research.backtesting import (
    TRADING_DAYS_PER_YEAR,
    TopNBacktestResult,
    run_top_n_intraday_backtest,
)
from quant.factor_research.experiment import ExperimentResult
from quant.factor_research.reporting import (
    render_drawdown_curve_svg,
    render_equity_curve_svg,
    write_evaluation_report,
)


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
                "previous_close_to_close_return": [
                    0.01,
                    0.03,
                    0.05,
                    -0.01,
                    0.07,
                    -0.04,
                ],
                "previous_open_to_close_return": [
                    0.02,
                    0.04,
                    0.06,
                    -0.02,
                    0.08,
                    -0.05,
                ],
                "target_close_to_previous_close_return": [
                    0.11,
                    0.13,
                    0.15,
                    -0.11,
                    0.17,
                    -0.14,
                ],
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
                "target_close_to_previous_close_return",
                "previous_close_to_close_return",
                "previous_open_to_close_return",
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
            result.top_selections["target_close_to_previous_close_return"],
            [0.13, 0.15, -0.14, -0.11],
        )
        np.testing.assert_allclose(
            result.top_selections["previous_close_to_close_return"],
            [0.03, 0.05, -0.04, -0.01],
        )
        np.testing.assert_allclose(
            result.top_selections["previous_open_to_close_return"],
            [0.04, 0.06, -0.05, -0.02],
        )
        np.testing.assert_allclose(
            result.top_selections["previous_actual_return"],
            [0.04, 0.06, -0.05, -0.02],
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
        predictions.loc[
            predictions["target_date"] == pd.Timestamp("2025-01-03"),
            "target_date",
        ] = pd.Timestamp("2025-02-03")
        backtest = run_top_n_intraday_backtest(
            predictions, "up_probability", top_n=2, slippage_bps=2, commission_bps=1
        )
        accuracy = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-02-03"]),
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
            drawdown_path = resolve_drawdown_chart_path(accuracy_path)
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "test-run",
                {},
                backtest=backtest,
                equity_chart_path=equity_path,
                drawdown_chart_path=drawdown_path,
            )
            report = report_path.read_text(encoding="utf-8")
            equity_svg = equity_path.read_text(encoding="utf-8")
            html_report = report_path.with_suffix(".html").read_text(
                encoding="utf-8"
            )

        self.assertIn("## Top N 日内策略回测", report)
        self.assertIn("### 每日 Top N 选股明细", report)
        self.assertIn("#### 2025-01", report)
        self.assertIn("#### 2025-02", report)
        self.assertIn("| Top N 排名 | 2025-01-02 |", report)
        self.assertIn("| Top N 排名 | 2025-02-03 |", report)
        self.assertIn(
            "`A`<br>预估：90.00%<br>实际：10.00%<br>t涨幅：13.00%"
            "<br>t-1涨幅：3.00%<br>t-1日内涨幅：4.00%",
            report,
        )
        self.assertIn(
            "`C`<br>预估：80.00%<br>实际：2.00%<br>t涨幅：-14.00%"
            "<br>t-1涨幅：-4.00%<br>t-1日内涨幅：-5.00%",
            report,
        )
        self.assertIn("### Top N 与横截面对照收益曲线", report)
        self.assertIn("### Top N 基准与横截面对照", report)
        self.assertIn("### Top N 超额与多空价差", report)
        self.assertIn("### 预测分数十分位收益", report)
        self.assertIn("### Top N 与其余股票命中对照", report)
        self.assertIn("### 回撤诊断", report)
        self.assertIn("### 历史回撤与修复", report)
        self.assertIn("### 历次回撤区间", report)
        self.assertIn(drawdown_path.name, report)
        self.assertIn("| 全股票等权最大回撤 |", report)
        self.assertLess(
            report.index("## Top N 日内策略回测"),
            report.index("### 每日 Top N 选股明细"),
        )
        # 回撤诊断与三个横截面对照小节都排在体量最大的每日明细之前。
        selection_position = report.index("### 每日 Top N 选股明细")
        for heading in (
            "### 回撤诊断",
            "### 历史回撤与修复",
            "### 历次回撤区间",
            "### Top N 与横截面对照收益曲线",
            "### Top N 基准与横截面对照",
            "### Top N 超额与多空价差",
        ):
            self.assertLess(report.index(heading), selection_position)
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
        self.assertIn("每日 Top N 选股明细", html_report)
        self.assertIn('class="toc-level-4"', html_report)
        self.assertIn(">2025-01</a>", html_report)
        self.assertIn(">2025-02</a>", html_report)
        # 目录里的逐月条目收进默认折叠分组：<details> 不带 open 属性。
        self.assertIn('<details class="toc-group"><summary>', html_report)
        self.assertNotIn("<details class=\"toc-group\" open", html_report)
        group = html_report[
            html_report.index('<details class="toc-group">') :
        ].split("</details>")[0]
        self.assertIn("每日 Top N 选股明细", group)
        self.assertIn('<div class="toc-children">', group)
        self.assertIn('class="toc-level-4" href="#', group)
        self.assertIn(">2025-01</a>", group)
        self.assertIn(">2025-02</a>", group)
        self.assertIn("`A`", report)
        self.assertIn("<code>A</code><br>预估：90.00%", html_report)
        self.assertIn('class="table-scroll"', html_report)

    def _single_security_predictions(self, returns: list[float]) -> pd.DataFrame:
        """构造每日只有一只证券的预测样本，使组合净值等于给定收益序列。

        参数：
            returns: 按交易日排序的组合日收益率序列，单位为一；每个交易日只有
                一只候选证券，因此 Top 1 组合的毛收益与之逐日相等。

        返回：
            含目标日期、证券代码、实际收益、模型分数和方向标签的预测表。
        """

        dates = pd.bdate_range("2025-01-06", periods=len(returns))
        return pd.DataFrame(
            {
                "target_date": dates,
                "code": [f"A{index}" for index in range(len(returns))],
                "target_return": list(returns),
                "up_probability": [0.6] * len(returns),
                "label": [int(value > 0) for value in returns],
                "prediction": [1] * len(returns),
            }
        )

    def test_drawdown_metrics_track_depth_recovery_and_calmar(self) -> None:
        """回撤指标应按期初净值 1.0 精确给出深度、修复时长与 Calmar 比率。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        returns = [0.10, -0.20, 0.05, 0.15, -0.10, 0.20]
        result = run_top_n_intraday_backtest(
            self._single_security_predictions(returns), "up_probability", top_n=1
        )
        equity = result.daily_returns["equity"].to_numpy(dtype=float)
        expected_equity = np.cumprod(1.0 + np.asarray(returns))
        np.testing.assert_allclose(equity, expected_equity)
        # 峰值 1.10 出现在首个交易日，谷底 0.88 对应 -20%，末日创出新高完成修复。
        expected_peaks = np.maximum.accumulate(
            np.concatenate(([1.0], expected_equity))
        )[1:]
        np.testing.assert_allclose(
            result.daily_returns["drawdown"].to_numpy(dtype=float),
            expected_equity / expected_peaks - 1.0,
        )
        np.testing.assert_allclose(
            result.daily_returns["peak_equity"].to_numpy(dtype=float),
            expected_peaks,
        )
        metrics = result.drawdown_metrics
        self.assertAlmostEqual(metrics["max_drawdown"], -0.2)
        self.assertEqual(metrics["max_drawdown_decline_days"], 1.0)
        self.assertEqual(metrics["max_drawdown_recovery_days"], 4.0)
        self.assertEqual(metrics["max_drawdown_total_days"], 5.0)
        self.assertEqual(metrics["longest_drawdown_days"], 5.0)
        self.assertEqual(metrics["drawdown_episodes"], 1.0)
        self.assertEqual(metrics["recovered_drawdown_episodes"], 1.0)
        self.assertAlmostEqual(metrics["current_drawdown"], 0.0)
        self.assertAlmostEqual(metrics["drawdown_days_ratio"], 4 / 6)
        self.assertAlmostEqual(metrics["average_drawdown"], -0.5246 / 6)
        self.assertAlmostEqual(
            metrics["calmar_ratio"],
            result.metrics["annualized_return"] / 0.2,
            places=5,
        )

        episodes = result.drawdown_episodes
        self.assertEqual(len(episodes), 1)
        episode = episodes.iloc[0]
        dates = pd.to_datetime(result.daily_returns["target_date"])
        self.assertEqual(pd.Timestamp(episode["peak_date"]), dates.iloc[0])
        self.assertEqual(pd.Timestamp(episode["trough_date"]), dates.iloc[1])
        self.assertEqual(pd.Timestamp(episode["recovery_date"]), dates.iloc[5])
        self.assertTrue(bool(episode["recovered"]))

    def test_drawdown_from_first_day_reports_unrecovered_tail(self) -> None:
        """首日即回撤且未修复时，峰值应记为期初且修复字段保持缺失。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        result = run_top_n_intraday_backtest(
            self._single_security_predictions([-0.10, 0.05, -0.02]),
            "up_probability",
            top_n=1,
        )
        metrics = result.drawdown_metrics
        self.assertAlmostEqual(metrics["max_drawdown"], -0.10)
        self.assertAlmostEqual(metrics["current_drawdown"], 1.0 * 0.9 * 1.05 * 0.98 - 1.0)
        self.assertEqual(metrics["drawdown_days_ratio"], 1.0)
        self.assertEqual(metrics["recovered_drawdown_episodes"], 0.0)
        self.assertTrue(np.isnan(metrics["max_drawdown_recovery_days"]))
        self.assertTrue(np.isfinite(metrics["calmar_ratio"]))

        episodes = result.drawdown_episodes
        self.assertEqual(len(episodes), 1)
        episode = episodes.iloc[0]
        # 峰值落在期初净值上，因此峰值日期缺失，回撤持续到样本末尾。
        self.assertTrue(pd.isna(episode["peak_date"]))
        self.assertTrue(pd.isna(episode["recovery_date"]))
        self.assertFalse(bool(episode["recovered"]))
        self.assertEqual(int(episode["total_days"]), 3)
        self.assertTrue(np.isnan(float(episode["recovery_days"])))

        accuracy = pd.DataFrame(
            {
                "target_date": result.daily_returns["target_date"],
                "samples": [1, 1, 1],
                "accuracy": [1.0, 1.0, 0.0],
                "accuracy_change": [np.nan, 0.0, -1.0],
            }
        )
        experiment = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="drawdown",
            feature_columns=[],
            metrics={"samples": 3.0},
            predictions=self._single_security_predictions([-0.10, 0.05, -0.02]),
            feature_importance=None,
            daily_accuracy_trend=accuracy,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "evaluation.md"
            accuracy_path = root / "evaluation_accuracy.svg"
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "drawdown-run",
                {},
                backtest=result,
                equity_chart_path=resolve_equity_chart_path(accuracy_path),
                drawdown_chart_path=resolve_drawdown_chart_path(accuracy_path),
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("| 最大回撤 | -10.00% |", report)
        self.assertIn("| 最大回撤谷底至修复交易日 | N/A |", report)
        self.assertIn("| 期初 | ", report)
        self.assertIn("| 未修复 | -10.00% |", report)

    def test_drawdown_chart_marks_trough_and_recovery(self) -> None:
        """回撤图应画出水下区间、回撤面积并标注谷底与修复位置。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        result = run_top_n_intraday_backtest(
            self._single_security_predictions([0.10, -0.20, 0.05, 0.15, -0.10, 0.20]),
            "up_probability",
            top_n=1,
        )
        legacy_daily = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "equity": [1.10, 0.99],
            }
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chart_path = root / "drawdown.svg"
            render_drawdown_curve_svg(result.daily_returns, chart_path)
            drawdown_svg = chart_path.read_text(encoding="utf-8")
            legacy_path = root / "legacy_drawdown.svg"
            render_drawdown_curve_svg(legacy_daily, legacy_path)
            legacy_svg = legacy_path.read_text(encoding="utf-8")

        self.assertIn('<polygon class="underwater"', drawdown_svg)
        self.assertIn('<polygon class="drawdown"', drawdown_svg)
        self.assertIn('<polyline class="universe"', drawdown_svg)
        self.assertIn('<circle class="marker"', drawdown_svg)
        self.assertIn('class="recovery"', drawdown_svg)
        # 图上标注的最大回撤必须与回测算出的 drawdown 列同源，不能各算各的。
        self.assertIn(
            f"最大回撤 {result.daily_returns['drawdown'].min():.2%}", drawdown_svg
        )
        self.assertIn("最大回撤 -20.00%", drawdown_svg)
        self.assertIn("修复，用时 4 个交易日", drawdown_svg)
        self.assertIn(">期初</text>", drawdown_svg)
        # 旧结果只有 Top N 净值时不画横截面对照，未修复的回撤要如实标注。
        self.assertNotIn('<polyline class="universe"', legacy_svg)
        self.assertNotIn('class="recovery"', legacy_svg)
        self.assertIn("截至验证区间末尾尚未修复", legacy_svg)

    def test_drawdown_chart_rejects_inconsistent_drawdown_column(self) -> None:
        """回撤列与净值对不上说明上游口径已分叉，应报错而不是画出错图。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        result = run_top_n_intraday_backtest(
            self._single_security_predictions([0.10, -0.20, 0.05]),
            "up_probability",
            top_n=1,
        )
        corrupted = result.daily_returns.copy()
        corrupted.loc[corrupted.index[-1], "drawdown"] = -0.5
        corrupted_peak = result.daily_returns.copy()
        corrupted_peak["peak_equity"] = 5.0
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "corrupted_drawdown.svg"
            with self.assertRaisesRegex(ValueError, "drawdown 与净值不一致"):
                render_drawdown_curve_svg(corrupted, output_path)
            # 峰值列同样不被无条件信任，否则图上的回撤深度会被外部列决定。
            with self.assertRaisesRegex(ValueError, "peak_equity 与净值不一致"):
                render_drawdown_curve_svg(corrupted_peak, output_path)

    def test_no_drawdown_reports_empty_episode_table(self) -> None:
        """净值一路创新高时回撤区间表应为空表，并给出明确说明而非空白。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        returns = [0.01, 0.02, 0.03]
        result = run_top_n_intraday_backtest(
            self._single_security_predictions(returns), "up_probability", top_n=1
        )
        self.assertTrue(result.drawdown_episodes.empty)
        # 空表也要保持列与类型，避免下游读取时才报错。
        self.assertEqual(
            list(result.drawdown_episodes.columns),
            [
                "peak_date",
                "trough_date",
                "recovery_date",
                "max_drawdown",
                "decline_days",
                "recovery_days",
                "total_days",
                "recovered",
            ],
        )
        self.assertEqual(result.drawdown_episodes["trough_date"].dtype, "datetime64[ns]")
        self.assertEqual(result.drawdown_episodes["recovered"].dtype, bool)
        metrics = result.drawdown_metrics
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertEqual(metrics["drawdown_episodes"], 0.0)
        self.assertEqual(metrics["longest_drawdown_days"], 0.0)
        self.assertEqual(metrics["drawdown_days_ratio"], 0.0)
        self.assertTrue(np.isnan(metrics["calmar_ratio"]))

        accuracy = pd.DataFrame(
            {
                "target_date": result.daily_returns["target_date"],
                "samples": [1, 1, 1],
                "accuracy": [1.0, 1.0, 1.0],
                "accuracy_change": [np.nan, 0.0, 0.0],
            }
        )
        experiment = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="no-drawdown",
            feature_columns=[],
            metrics={"samples": 3.0},
            predictions=self._single_security_predictions(returns),
            feature_importance=None,
            daily_accuracy_trend=accuracy,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "evaluation.md"
            accuracy_path = root / "evaluation_accuracy.svg"
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "no-drawdown-run",
                {},
                backtest=result,
                equity_chart_path=resolve_equity_chart_path(accuracy_path),
                drawdown_chart_path=resolve_drawdown_chart_path(accuracy_path),
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("验证区间内净值未跌破历史峰值，没有回撤区间。", report)
        self.assertIn("| 最大回撤 | 0.00% |", report)
        self.assertIn("| Calmar 比率（年化收益/最大回撤） | N/A |", report)

    def test_drawdown_episode_table_truncates_and_reports_total(self) -> None:
        """回撤区间超过展示上限时应只列最深的若干段并说明总段数。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        # 每个 +2% 创新高、随后 -1% 形成一段可修复回撤，共 12 段。
        returns = [0.02, -0.01] * 12
        result = run_top_n_intraday_backtest(
            self._single_security_predictions(returns), "up_probability", top_n=1
        )
        self.assertEqual(len(result.drawdown_episodes), 12)
        # 序列以 -1% 收尾，因此最后一段回撤截至样本末尾仍未修复。
        self.assertEqual(int(result.drawdown_episodes["recovered"].sum()), 11)
        self.assertFalse(bool(result.drawdown_episodes.iloc[-1]["recovered"]))

        accuracy = pd.DataFrame(
            {
                "target_date": result.daily_returns["target_date"],
                "samples": [1] * len(returns),
                "accuracy": [1.0] * len(returns),
                "accuracy_change": [np.nan] + [0.0] * (len(returns) - 1),
            }
        )
        experiment = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="many-drawdowns",
            feature_columns=[],
            metrics={"samples": float(len(returns))},
            predictions=self._single_security_predictions(returns),
            feature_importance=None,
            daily_accuracy_trend=accuracy,
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "evaluation.md"
            accuracy_path = root / "evaluation_accuracy.svg"
            write_evaluation_report(
                experiment,
                report_path,
                accuracy_path,
                "many-drawdowns-run",
                {},
                backtest=result,
                equity_chart_path=resolve_equity_chart_path(accuracy_path),
                drawdown_chart_path=resolve_drawdown_chart_path(accuracy_path),
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("共 12 段", report)
        section = report[report.index("### 历次回撤区间") :].split("### ")[1]
        rows = [
            line
            for line in section.splitlines()
            if line.startswith("| ") and not line.startswith("| ---")
        ]
        # 表头 1 行 + 最多 10 行明细。
        self.assertEqual(len(rows), 11)

    def test_drawdown_chart_rejects_unsorted_target_dates(self) -> None:
        """日期乱序会算错历史峰值与修复位置，应显式拒绝而非画出错误曲线。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        unsorted_daily = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-03", "2025-01-02"]),
                "equity": [0.99, 1.10],
            }
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unsorted_drawdown.svg"
            with self.assertRaisesRegex(ValueError, "升序"):
                render_drawdown_curve_svg(unsorted_daily, output_path)

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

    def test_equity_curve_svg_annotates_calendar_year_returns(self) -> None:
        """收益曲线应含按自然年重定基的下面板，并标注年份与各组当年收益。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        daily = pd.DataFrame(
            {
                "target_date": pd.to_datetime(
                    ["2024-12-30", "2024-12-31", "2025-01-02"]
                ),
                "equity": [1.10, 1.20, 1.32],
                "universe_equity": [1.05, 1.10, 0.99],
            }
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "yearly_equity.svg"
            render_equity_curve_svg(daily, output_path)
            equity_svg = output_path.read_text(encoding="utf-8")

        # 年份直接标注在下面板曲线区段内。
        self.assertIn(">2024</text>", equity_svg)
        self.assertIn(">2025</text>", equity_svg)
        # 2024 年：Top N 1.20/1.0-1=+20.0%，全市场 1.10/1.0-1=+10.0%；
        # 2025 年：Top N 1.32/1.20-1=+10.0%，全市场 0.99/1.10-1=-10.0%。
        self.assertIn(">Top N +20.0%</text>", equity_svg)
        self.assertIn(">Top N +10.0%</text>", equity_svg)
        self.assertIn(">全市场平均 +10.0%</text>", equity_svg)
        self.assertIn(">全市场平均 -10.0%</text>", equity_svg)
        # 每组曲线 = 全区间 1 条 + 每个自然年 1 条重定基区段。
        self.assertEqual(equity_svg.count('<polyline class="equity"'), 3)
        self.assertEqual(equity_svg.count('<polyline class="universe"'), 3)
        # 跨年处画一条年度分界虚线。
        self.assertEqual(equity_svg.count('class="year-boundary"'), 1)

    def test_equity_curve_svg_rejects_unsorted_target_dates(self) -> None:
        """日期乱序会破坏自然年切分口径，应显式拒绝而非画出错误区段。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        unsorted_daily = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2024-12-30"]),
                "equity": [1.05, 1.10],
            }
        )
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unsorted_equity.svg"
            with self.assertRaisesRegex(ValueError, "升序"):
                render_equity_curve_svg(unsorted_daily, output_path)

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
