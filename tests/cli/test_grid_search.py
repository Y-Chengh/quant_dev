"""网格与遗传搜索完整报告产出测试。"""

from __future__ import annotations

import json
import re
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant.cli.grid_search import (
    build_genetic_search_config,
    build_model_evaluator,
    build_search_space,
    print_genetic_progress,
    resolve_report_output_dir,
)
from quant.factor_research.factor_search import (
    FactorGeneticSearch,
    FactorGridSearch,
    GeneticProgressEvent,
    HoldoutIcEvaluator,
    PipelineGrid,
    SearchContext,
    identity,
    op,
)
from quant.factor_research.search_report import write_grid_search_report
from quant.factor_research.search_report.charts import (
    _write_rolling_ic_stability_chart,
)
from quant.factor_research.search_report.formatting import (
    _markdown_table,
    _powershell_single_quoted,
)


class _FailingVolumeHoldoutEvaluator:
    """模拟一个候选在 holdout 阶段失败、其余候选正常完成的评价器。"""

    def evaluate(self, candidate, values, context):
        """让 volume 候选失败，并对其他候选沿用标准 holdout IC 口径。

        参数：
            candidate: 当前待评价候选，用规范表达式识别 volume 数据源。
            values: 与上下文日频表等长的候选因子值。
            context: 保存目标收益和 holdout 日期掩码的搜索上下文。

        返回：
            非 volume 候选的标准 holdout IC 汇总指标。
        """

        if candidate.canonical == "column(volume)":
            raise RuntimeError("测试用 holdout 失败")
        return HoldoutIcEvaluator().evaluate(candidate, values, context)


class GridSearchReportTests(unittest.TestCase):
    """验证报告附件完整性以及 selection/holdout 披露边界。"""

    def test_default_output_directory_uses_requested_log_hierarchy(self) -> None:
        """默认目录应包含日期层、秒级启动时间和随机运行 ID。"""

        output = resolve_report_output_dir(
            datetime(2026, 8, 8, 21, 45, 30), random_id="a1b2c3d4"
        )
        self.assertEqual(
            output,
            Path("logs/_search/2026-08-08/20260808_214530_a1b2c3d4"),
        )

    def test_powershell_expression_argument_escapes_special_characters(self) -> None:
        """报告命令应禁止变量展开，并正确保留表达式内部的单引号。"""

        self.assertEqual(
            _powershell_single_quoted('column("quote\'s-$value`tick")'),
            "'column(\"quote''s-$value`tick\")'",
        )

    def test_markdown_rank_columns_remove_redundant_decimal_zeroes(self) -> None:
        """名次列应紧凑显示整数和并列平均名次，其他指标仍保留六位小数。"""

        table = _markdown_table(
            pd.DataFrame(
                {
                    "selection_oriented_rank_ic_rank": [6.0, 2.5],
                    "holdout_oriented_ic_rank": [1.0, 3.5],
                    "selection_oriented_rank_ic": [0.12345678, 0.02],
                }
            ),
            [
                "selection_oriented_rank_ic_rank",
                "holdout_oriented_ic_rank",
                "selection_oriented_rank_ic",
            ],
        )

        self.assertIn("| 6 | 1 | 0.123457 |", table)
        self.assertIn("| 2.5 | 3.5 | 0.020000 |", table)
        self.assertNotIn("6.000000", table)

    def test_model_evaluator_uses_registered_passthrough_regressor(self) -> None:
        """smoke 搜索应通过与主实验相同的模型工厂直出候选末列。"""

        evaluator = build_model_evaluator("2025-01-08")

        self.assertEqual(evaluator.model_factory.name, "factor_passthrough")
        self.assertEqual(evaluator.task, "regression")
        self.assertEqual(evaluator.training_mode, "single")

    def test_rolling_ic_chart_contains_both_metrics_and_holdout_boundary(self) -> None:
        """稳定性图应同时展示滚动 IC、Rank IC，并明确标记 holdout 起点。"""

        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        daily_ic = pd.DataFrame(
            {
                "factor_id": ["factor_a"] * len(dates),
                "period": ["selection"] * 60 + ["holdout"] * 20,
                "target_date": dates,
                "ic": [0.2] * len(dates),
                "rank_ic": [0.1] * len(dates),
            }
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rolling.svg"
            _write_rolling_ic_stability_chart(daily_ic, path, "factor_a")
            chart = path.read_text(encoding="utf-8")

        self.assertIn("Rolling IC stability — factor_a", chart)
        self.assertIn("IC · 20 日近期趋势", chart)
        self.assertIn("IC · 60 日长期趋势", chart)
        self.assertIn("Rank IC · 20 日近期趋势", chart)
        self.assertIn("Rank IC · 60 日长期趋势", chart)
        self.assertIn("动态纵轴", chart)
        self.assertIn('class="boundary"', chart)
        self.assertIn('class="short" points=', chart)
        self.assertIn('class="long" points=', chart)

    def test_rolling_ic_chart_waits_for_minimum_finite_history(self) -> None:
        """滚动均值在有限历史不足时不应提前绘制，逐日序列仍应保留。"""

        dates = pd.date_range("2025-01-01", periods=20, freq="D")
        daily_ic = pd.DataFrame(
            {
                "factor_id": ["factor_a"] * len(dates),
                "period": ["selection"] * len(dates),
                "target_date": dates,
                "ic": [pd.NA] + [0.2] * 19,
                "rank_ic": [pd.NA] + [0.1] * 19,
            }
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rolling.svg"
            _write_rolling_ic_stability_chart(daily_ic, path, "factor_a")
            chart = path.read_text(encoding="utf-8")

        self.assertIn('<polyline class="short" points=', chart)
        self.assertNotIn('<polyline class="long" points=', chart)

    def test_rolling_ic_chart_uses_exact_trailing_mean_and_boundary_date(self) -> None:
        """滚动线应精确使用尾随窗口，并把分界线放在首个 holdout 日期。"""

        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        daily_ic = pd.DataFrame(
            {
                "factor_id": ["factor_a"] * len(dates),
                "period": ["selection"] * 3 + ["holdout"] * 2,
                "target_date": dates,
                "ic": [pd.NA, 0.2, 0.4, 0.6, 0.8],
                "rank_ic": [pd.NA, 0.1, 0.3, 0.5, 0.7],
            }
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rolling.svg"
            _write_rolling_ic_stability_chart(
                daily_ic,
                path,
                "factor_a",
                window=3,
                min_periods=2,
            )
            chart = path.read_text(encoding="utf-8")

        self.assertIn(
            '<polyline class="long" points="915.00,299.62 '
            '1042.50,243.21 1170.00,130.38"/>',
            chart,
        )
        self.assertIn(
            '<polyline class="long" points="915.00,614.62 '
            '1042.50,558.21 1170.00,445.38"/>',
            chart,
        )
        self.assertIn(
            '<line x1="1042.50" y1="105" x2="1042.50" y2="325" '
            'class="boundary"/>',
            chart,
        )
        self.assertEqual(chart.count('x1="1042.50"'), 2)

    def test_short_and_long_trends_have_independent_visible_scales(self) -> None:
        """短期波动远大于长期波动时，两种趋势仍应各自占据可见纵向范围。"""

        dates = pd.bdate_range("2025-01-02", periods=160)
        index = np.arange(len(dates), dtype=float)
        daily_ic = pd.DataFrame(
            {
                "factor_id": ["factor_a"] * len(dates),
                "period": ["selection"] * 120 + ["holdout"] * 40,
                "target_date": dates,
                "ic": 0.02 + 0.03 * np.sin(index / 3.0) + 0.004 * np.sin(index / 30.0),
                "rank_ic": 0.03 + 0.04 * np.sin(index / 4.0) + 0.006 * np.sin(index / 35.0),
            }
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rolling.svg"
            _write_rolling_ic_stability_chart(daily_ic, path, "factor_a")
            chart = path.read_text(encoding="utf-8")

        short_points = re.findall(
            r'<polyline class="short" points="([^"]+)"', chart
        )[0]
        long_points = re.findall(
            r'<polyline class="long" points="([^"]+)"', chart
        )[0]
        short_y = [float(point.split(",")[1]) for point in short_points.split()]
        long_y = [float(point.split(",")[1]) for point in long_points.split()]
        self.assertGreater(max(short_y) - min(short_y), 150.0)
        self.assertGreater(max(long_y) - min(long_y), 150.0)

    def test_console_progress_includes_batch_budget_failures_and_eta(self) -> None:
        """smoke 进度输出应显示批次、selection 预算、失败数与 ETA。"""

        output = StringIO()
        event = GeneticProgressEvent(
            stage="selection",
            generation=2,
            max_generations=8,
            completed=16,
            total=96,
            successful=15,
            failed=1,
            selection_evaluations=112,
            max_evaluations=500,
            elapsed_seconds=4.0,
            eta_seconds=20.0,
        )

        with redirect_stdout(output):
            print_genetic_progress(event)

        text = output.getvalue()
        self.assertIn("第 2/8 代", text)
        self.assertIn("16/96", text)
        self.assertIn("失败 1", text)
        self.assertIn("selection 112/500", text)
        self.assertIn("ETA 20.0s", text)

    def test_smoke_genetic_config_covers_continuous_factor_operators(self) -> None:
        """smoke 配置应覆盖连续因子算子、长度惩罚和固定随机种子。"""

        config = build_genetic_search_config()

        expected_operators = {
            "add",
            "subtract",
            "multiply",
            "divide",
            "negative",
            "absolute",
            "log",
            "sign",
            "power",
            "signed_power",
            "delay",
            "delta",
            "returns",
            "ts_sum",
            "ts_mean",
            "ts_min",
            "ts_max",
            "ts_stddev",
            "ts_argmax",
            "ts_argmin",
            "ts_rank",
            "ts_correlation",
            "ts_covariance",
            "cs_rank",
            "cs_demean",
            "cs_zscore",
            "cs_scale",
            "cs_winsorize",
        }
        self.assertEqual(set(config.operator_parameters), expected_operators)
        self.assertEqual(
            config.operator_parameters["ts_stddev"]["ddof"], (0, 1)
        )
        self.assertEqual(
            config.operator_parameters["cs_winsorize"]["lower"], (0.01, 0.05)
        )
        self.assertEqual(config.free_node_count, 3)
        self.assertEqual(config.length_penalty, 0.001)
        self.assertGreater(config.max_nodes, config.free_node_count)
        self.assertEqual(config.random_seed, 20260809)
        legacy_space = build_search_space()
        self.assertEqual(
            legacy_space.estimate_size(), 30 * len(legacy_space.sources)
        )

    @staticmethod
    def _daily_frame() -> pd.DataFrame:
        """构造三个证券、十个交易日且横截面收益有差异的日频样本。

        返回：
            可直接创建 ``SearchContext`` 的已排序 OHLCV 日频表。
        """

        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        rows: list[dict[str, object]] = []
        for code_index, code in enumerate(("AAA", "BBB", "CCC"), start=1):
            for date_index, trade_date in enumerate(dates):
                close = 10.0 + code_index * 2.0 + date_index * (0.1 + code_index * 0.02)
                rows.append(
                    {
                        "code": code,
                        "trade_date": trade_date,
                        "open": close * (0.99 + code_index * 0.001),
                        "high": close * 1.01,
                        "low": close * 0.98,
                        "close": close,
                        "volume": 1_000.0 * code_index + date_index * 10.0,
                    }
                )
        return pd.DataFrame(rows).sort_values(["code", "trade_date"]).reset_index(drop=True)

    def test_report_writes_complete_artifact_set(self) -> None:
        """报告应保存完整榜单、候选树、逐日 IC、最优值、元数据和 SVG。"""

        daily = self._daily_frame()
        context = SearchContext.from_daily(
            daily,
            selection_start="2025-01-02",
            holdout_start="2025-01-08",
            holdout_end="2025-01-10",
        )
        space = PipelineGrid(
            sources=["close", "volume"],
            stages=[[identity(), op("delta", periods=[1])], [identity(), op("cs_rank")]],
        )
        result = FactorGridSearch(
            max_candidates=20,
            max_depth=3,
            max_lookback=5,
            min_coverage=0.5,
        ).run(
            context,
            space,
            holdout_top_k=2,
            model_evaluator=build_model_evaluator("2025-01-08"),
            model_top_k=2,
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "report"
            report_path = write_grid_search_report(
                result,
                context,
                output_dir,
                {"codes": ["AAA", "BBB", "CCC"], "bar_rows": 30, "elapsed_seconds": 1.25},
                holdout_top_k=2,
            )
            expected = {
                "report.md",
                "leaderboard.csv",
                "errors.csv",
                "top_candidates_daily_ic.csv",
                "best_factor_values.csv",
                "candidates.json",
                "run_metadata.json",
                "selection_ranking.svg",
                "rolling_ic_stability.svg",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            self.assertIn("holdout 结果不用于重排候选", report_path.read_text(encoding="utf-8"))
            leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
            self.assertEqual(len(leaderboard), len(result.leaderboard))
            self.assertTrue(
                leaderboard["expression_str"].equals(leaderboard["expression"])
            )
            self.assertIn("selection_oriented_rank_icir", leaderboard)
            top_row = leaderboard.iloc[0]
            self.assertEqual(top_row["direction"], -1.0)
            self.assertEqual(
                top_row["selection_oriented_positive_rank_ic_ratio"], 1.0
            )
            completed = leaderboard.loc[
                leaderboard["holdout_elapsed_seconds"].notna()
            ]
            pd.testing.assert_series_equal(
                completed["model_ic"],
                completed["holdout_ic"],
                check_names=False,
            )
            pd.testing.assert_series_equal(
                completed["model_rank_ic"],
                completed["holdout_rank_ic"],
                check_names=False,
            )
            for metric in (
                "selection_oriented_rank_ic",
                "holdout_oriented_ic",
                "holdout_oriented_rank_ic",
            ):
                rank_column = f"{metric}_rank"
                expected_ranks = completed[metric].rank(
                    method="average", ascending=False
                )
                pd.testing.assert_series_equal(
                    completed[rank_column],
                    expected_ranks,
                    check_names=False,
                )
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn(
                "| factor_id | expression_str | depth |",
                report_text,
            )
            self.assertIn(
                "| factor_id | expression_str | selection_oriented_rank_ic |",
                report_text,
            )
            self.assertIn("selection_oriented_rank_ic_rank", report_text)
            self.assertIn("holdout_oriented_ic_rank", report_text)
            self.assertIn("holdout_oriented_rank_ic_rank", report_text)
            self.assertEqual(
                list(pd.read_csv(output_dir / "errors.csv").columns),
                ["factor_id", "stage", "error", "elapsed_seconds"],
            )
            candidates = json.loads((output_dir / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(candidates), len(result.candidates))
            self.assertTrue(
                all(item["expression_str"] == item["canonical"] for item in candidates)
            )
            self.assertEqual(
                candidates[0]["expression_str"],
                result.candidates[0].expression_str,
            )
            metadata = json.loads(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertFalse(metadata["equivalence_deduplication_enabled"])
            self.assertEqual(metadata["rolling_ic_window"], 60)
            self.assertEqual(metadata["rolling_ic_min_periods"], 20)
            self.assertEqual(metadata["rolling_ic_short_window"], 20)
            self.assertEqual(metadata["rolling_ic_short_min_periods"], 5)
            self.assertEqual(
                metadata["rolling_ic_stability_factor_id"],
                completed.iloc[0]["factor_id"],
            )
            self.assertNotIn("selection 等价去重", report_text)
            self.assertEqual(metadata["best_expression_str"], result.best_candidate.canonical)
            self.assertIn("--factor-expressions", report_path.read_text(encoding="utf-8"))
            self.assertIn(
                "--model factor_passthrough --task regression",
                report_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "--training-mode single --validation-start '2025-01-08'",
                report_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "--codes 'AAA' 'BBB' 'CCC'",
                report_path.read_text(encoding="utf-8"),
            )
            daily_ic = pd.read_csv(output_dir / "top_candidates_daily_ic.csv")
            self.assertLessEqual(daily_ic["factor_id"].nunique(), 2)
            self.assertEqual(set(daily_ic["period"]), {"selection", "holdout"})
            self.assertIn("rolling_ic_stability.svg", report_text)
            self.assertIn("衰减、漂移与符号翻转", report_text)

    def test_genetic_report_writes_evolution_history(self) -> None:
        """遗传搜索报告应额外保存逐代轨迹、复杂度字段和遗传算法元数据。"""

        daily = self._daily_frame()
        context = SearchContext.from_daily(
            daily,
            selection_start="2025-01-02",
            holdout_start="2025-01-08",
            holdout_end="2025-01-10",
        )
        config = replace(
            build_genetic_search_config(),
            sources=("close", "volume"),
            operator_parameters={
                "cs_rank": {},
                "ts_correlation": {"window": (3,)},
            },
            population_size=12,
            max_generations=2,
            max_evaluations=20,
            initial_max_depth=1,
            max_depth=2,
            max_nodes=5,
            max_lookback=3,
            min_coverage=0.2,
            target_coverage=0.5,
        )
        result = FactorGeneticSearch(config).run(
            context,
            holdout_top_k=1,
        )

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "genetic-report"
            report_path = write_grid_search_report(
                result,
                context,
                output_dir,
                {
                    "codes": ["AAA", "BBB", "CCC"],
                    "bar_rows": 30,
                    "elapsed_seconds": 0.5,
                    "search_algorithm": "genetic_programming",
                },
                holdout_top_k=1,
            )

            history = pd.read_csv(output_dir / "evolution_history.csv")
            leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
            metadata = json.loads(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            report_text = report_path.read_text(encoding="utf-8")
            self.assertEqual(len(history), len(result.history))
            self.assertIn("node_count", leaderboard)
            self.assertIn("length_penalty", leaderboard)
            self.assertEqual(metadata["evolution_generations"], len(result.history))
            self.assertTrue(metadata["equivalence_deduplication_enabled"])
            self.assertEqual(
                metadata["equivalent_candidates_removed"],
                len(result.candidates) - len(result.leaderboard),
            )
            self.assertIn("# 因子遗传编程搜索报告", report_text)
            self.assertIn("selection 上按方向调整后的候选值", report_text)
            self.assertIn("evolution_history.csv", report_text)

    def test_passthrough_model_excludes_the_same_missing_holdout_values_as_ic(self) -> None:
        """候选含缺失值时，模型复验与搜索 IC 应排除完全相同的样本。"""

        daily = self._daily_frame()
        daily["candidate"] = daily["close"]
        missing_row = (
            daily["code"].eq("AAA")
            & daily["trade_date"].eq(pd.Timestamp("2025-01-08"))
        )
        daily.loc[missing_row, "candidate"] = float("nan")
        context = SearchContext.from_daily(
            daily,
            selection_start="2025-01-02",
            holdout_start="2025-01-08",
            holdout_end="2025-01-10",
        )
        result = FactorGridSearch(
            max_candidates=2,
            min_coverage=0.5,
        ).run(
            context,
            PipelineGrid(sources=["candidate"], stages=[[identity()]]),
            holdout_top_k=1,
            model_evaluator=build_model_evaluator("2025-01-08"),
            model_top_k=1,
        )

        row = result.leaderboard.iloc[0]
        self.assertEqual(row["model_samples"], row["holdout_valid_values"])
        self.assertAlmostEqual(row["model_ic"], row["holdout_ic"])
        self.assertAlmostEqual(row["model_rank_ic"], row["holdout_rank_ic"])

    def test_report_cannot_expand_or_bypass_completed_holdout_candidates(self) -> None:
        """报告只能披露搜索阶段成功完成 holdout 的冻结候选，不能自行扩展 K。"""

        context = SearchContext.from_daily(
            self._daily_frame(),
            selection_start="2025-01-02",
            holdout_start="2025-01-08",
            holdout_end="2025-01-10",
        )
        space = PipelineGrid(sources=["close", "volume"], stages=[[identity()]])
        result = FactorGridSearch(
            max_candidates=10,
            max_depth=2,
            max_lookback=2,
            min_coverage=0.5,
        ).run(
            context,
            space,
            holdout_evaluator=_FailingVolumeHoldoutEvaluator(),
            holdout_top_k=2,
        )
        successful_ids = set(
            result.leaderboard.loc[
                result.leaderboard["holdout_elapsed_seconds"].notna(), "factor_id"
            ]
        )
        self.assertEqual(len(successful_ids), 1)

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "report"
            write_grid_search_report(
                result,
                context,
                output_dir,
                {"codes": ["AAA", "BBB", "CCC"], "elapsed_seconds": 0.1},
                holdout_top_k=5,
            )
            daily_ic = pd.read_csv(output_dir / "top_candidates_daily_ic.csv")
            self.assertEqual(set(daily_ic["factor_id"]), successful_ids)
            metadata = json.loads(
                (output_dir / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(metadata["successful_holdout_candidates"]), successful_ids)
            errors = pd.read_csv(output_dir / "errors.csv")
            self.assertEqual(errors["stage"].tolist(), ["holdout"])


if __name__ == "__main__":
    unittest.main()
