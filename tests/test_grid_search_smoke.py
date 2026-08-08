"""网格搜索完整报告产出测试。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from factor_research.factor_search import (
    FactorGridSearch,
    HoldoutIcEvaluator,
    PipelineGrid,
    SearchContext,
    identity,
    op,
)
from grid_search_smoke import (
    _powershell_single_quoted,
    resolve_report_output_dir,
    write_grid_search_report,
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
        ).run(context, space, holdout_top_k=2)

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
            self.assertEqual(metadata["best_expression_str"], result.best_candidate.canonical)
            self.assertIn("--factor-expressions", report_path.read_text(encoding="utf-8"))
            daily_ic = pd.read_csv(output_dir / "top_candidates_daily_ic.csv")
            self.assertLessEqual(daily_ic["factor_id"].nunique(), 2)
            self.assertEqual(set(daily_ic["period"]), {"selection", "holdout"})

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
