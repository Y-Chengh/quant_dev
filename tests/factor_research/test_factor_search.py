from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.factor_research.dataset import build_direction_dataset, build_forward_targets
from quant.factor_research.factor_dsl import ExpressionNode
from quant.factor_research.factor_factories import FACTOR_FACTORIES
from quant.factor_research.factor_search import (
    CombinedGrid,
    ExpressionGrid,
    FactorGridSearch,
    ModelCandidateEvaluator,
    PipelineGrid,
    SearchContext,
    identity,
    op,
    prepare_search_context,
)
from quant.factor_research.models.simple_decision_tree import SimpleDecisionTreeModelFactory


def _search_daily(days: int = 12) -> pd.DataFrame:
    """生成横截面收益顺序稳定且包含固定因子的日频搜索样本。

    参数：
        days: 每个证券生成的连续工作日数，缺省为 12。
    """

    dates = pd.bdate_range("2024-01-02", periods=days)
    returns = {"A": -0.02, "B": -0.005, "C": 0.01, "D": 0.025}
    rows = []
    for date in dates:
        for code_index, (code, target_return) in enumerate(returns.items()):
            rows.append(
                {
                    "code": code,
                    "trade_date": date,
                    "open": 100.0,
                    "high": 103.0,
                    "low": 97.0,
                    "close": 100.0 * (1 + target_return),
                    "volume": 1000.0 + code_index * 100,
                    "fixed": float(code_index),
                }
            )
    return pd.DataFrame(rows)


def _minute_bars(days: int = 8) -> pd.DataFrame:
    """生成可供真实聚合和固定因子工厂使用的多证券五分钟样本。

    参数：
        days: 每个证券生成分钟行情的连续工作日数，缺省为 8。
    """

    rows = []
    for date in pd.bdate_range("2024-01-02", periods=days):
        for code_index, code in enumerate(("A", "B", "C")):
            for offset, minute in enumerate((35, 40)):
                opening = 10.0 + code_index + offset * 0.1
                closing = opening * (1 + (code_index - 1) * 0.001)
                rows.append(
                    {
                        "code": code,
                        "trade_time": date + pd.Timedelta(hours=9, minutes=minute),
                        "open": opening,
                        "high": max(opening, closing) + 0.01,
                        "low": min(opening, closing) - 0.01,
                        "close": closing,
                        "volume": 1000.0 + offset,
                    }
                )
    return pd.DataFrame(rows)


def _context() -> SearchContext:
    """构造带固定因子、selection 和 holdout 的标准测试上下文。"""

    daily = _search_daily()
    dates = daily["trade_date"].drop_duplicates().sort_values()
    return SearchContext.from_daily(
        daily,
        fixed_features=["fixed"],
        selection_start=dates.iloc[1],
        holdout_start=dates.iloc[8],
    )


class FactorSearchSpaceTest(unittest.TestCase):
    """验证搜索空间展开、参数默认值、确定性和去重行为。"""

    def test_pipeline_grid_expands_deterministically_and_deduplicates_identity(self):
        """流水线重复展开顺序应稳定，多个 identity 产生的等价式应去重。"""

        space = PipelineGrid(
            sources=["close"],
            stages=[
                [op("delta", periods=[1, 2]), identity()],
                [op("cs_rank"), identity(), identity()],
            ],
        )

        first = space.generate()
        second = space.generate()

        self.assertEqual(len(first), 6)
        self.assertEqual(
            [candidate.canonical for candidate in first],
            [candidate.canonical for candidate in second],
        )
        self.assertEqual(len({candidate.factor_id for candidate in first}), len(first))

    def test_candidate_string_round_trips_for_non_identifier_source(self):
        """含标点的合法搜索源也必须生成可安全解析的候选字符串。"""

        candidate = PipelineGrid(
            sources=["adjusted-close"],
            stages=[[identity()]],
        ).generate()[0]

        self.assertEqual(candidate.expression_str, "column('adjusted-close')")
        self.assertEqual(
            ExpressionNode.from_string(candidate.expression_str),
            candidate.expression,
        )

    def test_pipeline_rejects_binary_operator(self):
        """单输入流水线必须拒绝相关性等需要多个输入的算子。"""

        space = PipelineGrid(
            sources=["close"],
            stages=[[op("ts_correlation", window=[3], min_periods=[3])]],
        )

        with self.assertRaisesRegex(ValueError, "多输入"):
            space.generate()

    def test_expression_grid_builds_multi_input_formulas(self):
        """模板网格应能展开多输入公式并正确累计历史回看长度。"""

        space = ExpressionGrid(
            builder=lambda daily, parameters: daily.close().correlation(
                daily.volume(), int(parameters["window"])
            ),
            parameters={"window": [3, 5]},
        )

        candidates = space.generate()

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.lookback for candidate in candidates], [2, 4])

    def test_rolling_operator_grid_applies_documented_defaults(self):
        """滚动算子网格应补齐完整窗口和标准差自由度的默认参数。"""

        space = PipelineGrid(
            sources=["close"],
            stages=[[op("ts_stddev", window=[3, 5])]],
        )

        candidates = space.generate()

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.lookback for candidate in candidates], [2, 4])
        self.assertIn("ddof=1", candidates[0].canonical)
        self.assertIn("min_periods=3", candidates[0].canonical)


class FactorSearchExecutionTest(unittest.TestCase):
    """验证上下文准备、评价隔离、并行执行、模型复验和结果物化。"""

    def test_context_rejects_invalid_key_types_and_values(self):
        """搜索上下文只校验上游主键契约，不在内部静默转换。"""

        daily = _search_daily(days=3)

        invalid_code = daily.copy()
        invalid_code["code"] = pd.Series(
            invalid_code["code"].tolist(), dtype=object
        )
        invalid_code.loc[0, "code"] = 1
        with self.assertRaisesRegex(TypeError, "code 列必须全部为字符串"):
            SearchContext.from_daily(invalid_code)

        string_dates = daily.assign(
            trade_date=daily["trade_date"].dt.strftime("%Y-%m-%d")
        )
        with self.assertRaisesRegex(TypeError, "trade_date 列必须为 datetime64"):
            SearchContext.from_daily(string_dates)

        intraday_dates = daily.copy()
        intraday_dates.loc[0, "trade_date"] += pd.Timedelta(hours=6)
        with self.assertRaisesRegex(ValueError, "trade_date 必须为归零后的交易日"):
            SearchContext.from_daily(intraday_dates)

    def test_context_rejects_duplicate_daily_keys(self):
        """同一证券同一交易日只能存在一行。"""

        daily = _search_daily(days=3)
        duplicate = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "存在重复的.*code, trade_date"):
            SearchContext.from_daily(duplicate)

    def test_forward_targets_match_existing_dataset_target_semantics(self):
        """搜索目标必须与现有方向数据集的下一交易日目标口径一致。"""

        daily = _search_daily()
        targets = build_forward_targets(daily)
        dataset = build_direction_dataset(daily, feature_columns=["fixed"])

        pd.testing.assert_frame_equal(
            targets,
            dataset[
                [
                    "feature_date",
                    "target_date",
                    "target_end_date",
                    "code",
                    "target_return",
                    "label",
                ]
            ],
        )

    def test_context_keeps_selection_and_holdout_isolated(self):
        """selection 和 holdout 应按目标日期严格分离且互不重叠。"""

        context = _context()
        selection_dates = context.targets.loc[context.selection_mask, "target_date"]
        holdout_dates = context.targets.loc[context.holdout_mask, "target_date"]

        self.assertLess(selection_dates.max(), holdout_dates.min())
        self.assertEqual(holdout_dates.min(), context.holdout_start)
        self.assertFalse((context.selection_mask & context.holdout_mask).any())

    def test_fixed_factor_factory_is_called_once_during_context_preparation(self):
        """固定因子工厂应只在上下文准备时调用一次，worker 不得重复计算。"""

        factory = FACTOR_FACTORIES["return_1d"]
        original = factory.compute
        dates = pd.bdate_range("2024-01-02", periods=8)

        with patch.object(factory, "compute", wraps=original) as compute:
            context = prepare_search_context(
                _minute_bars(),
                fixed_features=["return_1d"],
                selection_start=dates[1],
                holdout_start=dates[5],
            )
            # 搜索阶段使用进程 worker；如果 worker 错误地重新访问工厂，调用次数会
            # 超过一次或 mock 无法序列化。这里同时验证完整边界。
            FactorGridSearch(backend="process", n_jobs=2, batch_size=1).run(
                context,
                PipelineGrid(["close"], [[op("cs_rank")]]),
            )

        self.assertEqual(compute.call_count, 1)
        self.assertIn("return_1d", context.daily)

    def test_sequential_search_scores_and_materializes_best_candidate(self):
        """串行搜索应排序有效候选并把最优表达式物化为普通日频列。"""

        context = _context()
        space = PipelineGrid(
            sources=["close", "volume"],
            stages=[[op("cs_rank")]],
        )
        search = FactorGridSearch(min_coverage=1.0)

        result = search.run(context, space)
        materialized = result.materialize(context, oriented=True)

        self.assertTrue(result.errors.empty)
        self.assertEqual(len(result.leaderboard), 2)
        self.assertTrue(result.leaderboard.iloc[0]["eligible"])
        self.assertAlmostEqual(
            result.leaderboard.iloc[0]["selection_oriented_rank_ic"], 1.0
        )
        self.assertIn(result.best_candidate.factor_id, materialized)
        self.assertEqual(len(materialized), len(context.daily))
        # holdout 默认只对 selection 第一名披露，不能借助完整 holdout 榜单挑选。
        self.assertEqual(result.leaderboard["holdout_rank_ic"].notna().sum(), 1)

    def test_factor_direction_is_locked_on_selection_not_holdout(self):
        """因子方向只能由 selection 确定，holdout 反转时不得重新选方向。"""

        dates = pd.bdate_range("2024-01-02", periods=8)
        rows = []
        for day_index, date in enumerate(dates):
            for code_index, code in enumerate(("A", "B", "C", "D")):
                # signal 始终递增；目标日进入 holdout 后，实际收益顺序反转。
                signal = float(code_index)
                target_signal = signal if day_index < 5 else -signal
                rows.append(
                    {
                        "code": code,
                        "trade_date": date,
                        "open": 100.0,
                        "close": 100.0 * (1 + target_signal * 0.01),
                        "signal": signal,
                    }
                )
        context = SearchContext.from_daily(
            pd.DataFrame(rows),
            selection_start=dates[1],
            holdout_start=dates[5],
        )

        result = FactorGridSearch().run(
            context,
            PipelineGrid(["signal"], [[op("cs_rank")]]),
        )
        row = result.leaderboard.iloc[0]

        self.assertEqual(row["direction"], 1.0)
        self.assertAlmostEqual(row["selection_oriented_rank_ic"], 1.0)
        self.assertAlmostEqual(row["holdout_oriented_rank_ic"], -1.0)

    def test_runtime_error_is_isolated_to_invalid_candidate(self):
        """单个候选运行错误应写入错误表，而不终止其他合法候选。"""

        context = _context()
        space = CombinedGrid(
            [
                PipelineGrid(["close"], [[op("cs_rank")]]),
                PipelineGrid(["missing_column"], [[op("cs_rank")]]),
            ]
        )

        result = FactorGridSearch().run(context, space)

        self.assertEqual(len(result.leaderboard), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("缺少表达式列", result.errors.iloc[0]["error"])

    def test_candidate_limit_and_depth_are_checked_before_execution(self):
        """候选数量、深度及禁止 holdout 排序应在实际计算前完成校验。"""

        context = _context()
        space = PipelineGrid(
            ["close", "volume"],
            [[op("delta", periods=[1, 2])], [op("cs_rank")]],
        )

        with self.assertRaisesRegex(ValueError, "超过上限"):
            FactorGridSearch(max_candidates=2).run(context, space)
        with self.assertRaisesRegex(ValueError, "最大深度"):
            FactorGridSearch(max_depth=1).run(context, space)
        with self.assertRaisesRegex(ValueError, "禁止使用 holdout"):
            FactorGridSearch(objective="holdout_oriented_rank_ic")

    def test_process_and_sequential_backends_produce_identical_metrics(self):
        """进程与串行后端应返回一致的候选身份、顺序和 selection 指标。"""

        context = _context()
        space = PipelineGrid(
            ["close", "volume"],
            [[op("delta", periods=[1, 2]), identity()], [op("cs_rank")]],
        )
        sequential = FactorGridSearch(backend="sequential", batch_size=2).run(
            context, space
        )
        parallel = FactorGridSearch(
            backend="process", n_jobs=2, batch_size=2
        ).run(context, space)

        columns = [
            "factor_id",
            "selection_ic",
            "selection_rank_ic",
            "holdout_ic",
            "holdout_rank_ic",
            "eligible",
        ]
        left = sequential.leaderboard[columns].sort_values("factor_id").reset_index(drop=True)
        right = parallel.leaderboard[columns].sort_values("factor_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(len(sequential.errors), len(parallel.errors))

    def test_top_k_model_evaluation_reuses_existing_experiment(self):
        """Top K 模型复验应复用既有实验流程并把指标并入排行榜。"""

        context = _context()
        space = PipelineGrid(["close"], [[op("cs_rank")]])
        evaluator = ModelCandidateEvaluator(
            SimpleDecisionTreeModelFactory(max_depth=1, min_samples_leaf=1),
            training_mode="single",
        )

        result = FactorGridSearch().run(
            context,
            space,
            model_evaluator=evaluator,
            model_top_k=1,
        )

        self.assertIn("model_auc", result.leaderboard)
        self.assertTrue(np.isfinite(result.leaderboard.iloc[0]["model_auc"]))
        self.assertTrue(result.errors.empty)

    def test_model_evaluation_respects_context_holdout_end(self):
        """模型复验样本必须截断在上下文声明的 holdout 结束日期。"""

        daily = _search_daily()
        dates = daily["trade_date"].drop_duplicates().sort_values()
        context = SearchContext.from_daily(
            daily,
            fixed_features=["fixed"],
            selection_start=dates.iloc[1],
            holdout_start=dates.iloc[8],
            holdout_end=dates.iloc[9],
        )
        evaluator = ModelCandidateEvaluator(
            SimpleDecisionTreeModelFactory(max_depth=1, min_samples_leaf=1),
            training_mode="single",
        )

        result = FactorGridSearch().run(
            context,
            PipelineGrid(["close"], [[op("cs_rank")]]),
            model_evaluator=evaluator,
            model_top_k=1,
        )

        # holdout 只含 2 个目标日期，每日 4 只证券；结束日期之后不能进入模型指标。
        self.assertEqual(result.leaderboard.iloc[0]["model_samples"], 8.0)


if __name__ == "__main__":
    unittest.main()
