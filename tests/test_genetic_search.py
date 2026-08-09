from __future__ import annotations

from dataclasses import replace
import unittest

import pandas as pd

from factor_research.factor_dsl import ExpressionNode, operation_node
from factor_research.factor_search import (
    FactorCandidate,
    FactorGeneticSearch,
    GeneticProgressEvent,
    GeneticSearchConfig,
    IcEvaluator,
    PipelineGrid,
    ProcessBackend,
    SearchContext,
    op,
)


def _genetic_context() -> SearchContext:
    """构造同时包含 selection 和 holdout 的多证券遗传搜索样本。"""

    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-02", periods=12)
    for day_index, date in enumerate(dates):
        for code_index, code in enumerate(("A", "B", "C", "D")):
            daily_return = (code_index - 1.5) * 0.01 + day_index * 0.0001
            rows.append(
                {
                    "code": code,
                    "trade_date": date,
                    "open": 100.0,
                    "close": 100.0 * (1.0 + daily_return),
                    "signal": float(code_index) + day_index * 0.01,
                    "volume": 1000.0 + code_index * 100.0 + day_index,
                }
            )
    return SearchContext.from_daily(
        pd.DataFrame(rows),
        selection_start=dates[1],
        holdout_start=dates[9],
    )


def _small_config() -> GeneticSearchConfig:
    """返回适合单元测试且同时包含一元和二元算子的遗传配置。"""

    return GeneticSearchConfig(
        sources=("signal", "volume"),
        operator_parameters={
            "cs_rank": {},
            "delta": {"periods": (1, 2)},
            "ts_correlation": {"window": (3,)},
        },
        population_size=24,
        max_generations=3,
        max_evaluations=60,
        initial_max_depth=2,
        max_depth=3,
        max_nodes=9,
        max_lookback=8,
        min_coverage=0.2,
        target_coverage=0.5,
        patience=3,
        random_seed=17,
    )


class _CountingEvaluator:
    """记录实际评价次数并返回与候选复杂度无关的稳定 selection 指标。"""

    def __init__(self, *, coverage: float = 1.0) -> None:
        """初始化尚未发生候选评价的计数器。

        参数：
            coverage: 每个候选返回的 selection 有效值覆盖率。
        """

        self.calls = 0
        self.coverage = coverage

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """累计调用次数并返回遗传适应度所需的完整指标。

        参数：
            candidate: 当前被评价的规范因子候选。
            values: 与日频表逐行对齐的候选数值；本测试不使用具体数值。
            context: 提供日期切分的只读搜索上下文；本测试不读取标签。
        """

        self.calls += 1
        return {
            "selection_oriented_rank_ic": 0.05,
            "selection_rank_ic": 0.05,
            "selection_rank_ic_std": 0.0,
            "selection_rank_ic_dates": 5.0,
            "selection_coverage": self.coverage,
            "direction": 1.0,
        }


class _FailingEvaluator:
    """模拟每个候选都在 worker 内评价失败的测试评价器。"""

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """抛出确定性异常以验证候选级错误隔离。

        参数：
            candidate: 当前被评价的规范因子候选。
            values: 与日频表逐行对齐的候选数值。
            context: 提供日期切分的只读搜索上下文。
        """

        raise RuntimeError("expected genetic evaluation failure")


class _RecordingEvaluator:
    """记录 selection 与 holdout 评价顺序的确定性测试评价器。"""

    def __init__(self, period: str, events: list[str]) -> None:
        """保存当前评价区间名称和共享事件列表。

        参数：
            period: 当前评价器代表的区间，只允许 selection 或 holdout。
            events: 按调用先后追加区间名称的共享列表。
        """

        self.period = period
        self.events = events

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """记录调用并返回当前区间需要的稳定指标。

        参数：
            candidate: 当前被评价的规范因子候选。
            values: 与日频表逐行对齐的候选数值。
            context: 提供 selection 和 holdout 掩码的只读搜索上下文。
        """

        self.events.append(self.period)
        if self.period == "holdout":
            return {"holdout_ic": 0.02, "holdout_rank_ic": 0.03}
        if self.period == "model":
            return {"model_ic": 0.02, "model_rank_ic": 0.03}
        return {
            "selection_oriented_rank_ic": 0.05,
            "selection_rank_ic": 0.05,
            "selection_rank_ic_std": 0.0,
            "selection_rank_ic_dates": 5.0,
            "selection_coverage": 1.0,
            "direction": 1.0,
        }


class GeneticExpressionTest(unittest.TestCase):
    """验证表达式长度口径和遗传配置边界。"""

    def test_node_count_includes_terminals_and_operators(self) -> None:
        """表达式长度必须累计所有算子节点和数据源节点。"""

        close = ExpressionNode.column("close")
        volume = ExpressionNode.column("volume")
        delta = operation_node("delta", (close,), {"periods": 5})
        correlation = operation_node(
            "ts_correlation", (delta, volume), {"window": 10}
        )

        self.assertEqual(close.node_count, 1)
        self.assertEqual(delta.node_count, 2)
        self.assertEqual(correlation.node_count, 4)

    def test_configuration_rejects_non_selection_objective(self) -> None:
        """遗传适应度不得使用 holdout 指标。"""

        with self.assertRaisesRegex(ValueError, "holdout"):
            GeneticSearchConfig(objective="holdout_rank_ic")

    def test_integer_configuration_rejects_booleans_and_floats(self) -> None:
        """整数配置必须在入口拒绝 Python 布尔值和浮点值。"""

        invalid_configs = (
            {"initial_max_depth": 1.5},
            {"max_lookback": True},
            {"free_node_count": 2.5},
            {"free_depth": False},
            {"random_seed": 1.5},
        )
        for parameters in invalid_configs:
            with self.subTest(parameters=parameters):
                with self.assertRaisesRegex(ValueError, "整数"):
                    GeneticSearchConfig(**parameters)

        with self.assertRaisesRegex(ValueError, "batch_size"):
            FactorGeneticSearch(batch_size=1.5)
        with self.assertRaisesRegex(ValueError, "n_jobs"):
            FactorGeneticSearch(n_jobs=True)

    def test_holdout_top_k_rejects_non_integer(self) -> None:
        """holdout 候选数量必须在进入搜索前拒绝非整数。"""

        with self.assertRaisesRegex(ValueError, "holdout_top_k"):
            FactorGeneticSearch(_small_config()).run(
                _genetic_context(), holdout_top_k=1.5
            )
        with self.assertRaisesRegex(ValueError, "model_top_k"):
            FactorGeneticSearch(_small_config()).run(
                _genetic_context(), model_top_k=True
            )
        with self.assertRaisesRegex(TypeError, "progress_callback"):
            FactorGeneticSearch(_small_config()).run(
                _genetic_context(), progress_callback=True
            )


class GeneticSearchExecutionTest(unittest.TestCase):
    """验证多输入进化、长度惩罚、缓存和多进程确定性。"""

    def test_search_generates_unary_and_automatically_nested_binary_inputs(self) -> None:
        """搜索应自动生成一元根节点及输入同样由搜索产生的相关性表达式。"""

        config = _small_config()
        result = FactorGeneticSearch(config).run(
            _genetic_context(), holdout_top_k=0
        )
        expressions = [candidate.expression for candidate in result.candidates]

        self.assertTrue(
            all(
                node.node_count <= config.max_nodes
                and node.depth <= config.max_depth
                and node.lookback <= config.max_lookback
                for node in expressions
            )
        )
        self.assertTrue(any(node.operator == "cs_rank" for node in expressions))
        correlations = [node for node in expressions if node.operator == "ts_correlation"]
        self.assertTrue(correlations)
        for node in correlations:
            child_keys = [child.canonical for child in node.inputs]
            self.assertEqual(child_keys, sorted(child_keys))
            self.assertEqual(len(child_keys), len(set(child_keys)))
        self.assertTrue(
            any(any(child.inputs for child in node.inputs) for node in correlations)
        )

    def test_length_penalty_and_cross_generation_cache_are_applied(self) -> None:
        """同一表达式只能评价一次，且超出免费节点数后按节点线性扣分。"""

        evaluator = _CountingEvaluator()
        config = _small_config()
        result = FactorGeneticSearch(config).run(
            _genetic_context(), evaluator=evaluator, holdout_top_k=0
        )

        self.assertEqual(evaluator.calls, len(result.candidates))
        expected = (
            result.leaderboard["node_count"] - config.free_node_count
        ).clip(lower=0) * config.length_penalty
        pd.testing.assert_series_equal(
            result.leaderboard["length_penalty"],
            expected,
            check_names=False,
        )
        self.assertTrue(
            result.history["total_evaluations"].is_monotonic_increasing
        )

    def test_process_and_sequential_searches_are_identical(self) -> None:
        """相同种子下串行和持久化多进程搜索应产生完全相同的进化轨迹。"""

        context = _genetic_context()
        config = _small_config()
        sequential = FactorGeneticSearch(
            config, backend="sequential", batch_size=4
        ).run(context, holdout_top_k=0)
        process = FactorGeneticSearch(
            config, backend="process", n_jobs=2, batch_size=4
        ).run(context, holdout_top_k=0)

        columns = [
            "factor_id",
            "node_count",
            "depth",
            "lookback",
            "first_generation",
            "genetic_operation",
            "parent_ids",
            "selection_rank_ic",
            "length_penalty",
            "fitness",
            "eligible",
        ]
        pd.testing.assert_frame_equal(
            sequential.leaderboard[columns],
            process.leaderboard[columns],
        )
        pd.testing.assert_frame_equal(sequential.history, process.history)

    def test_progress_callback_reports_completed_batches_across_all_stages(self) -> None:
        """进度回调应按批次覆盖 selection、holdout 和模型复验阶段。"""

        events: list[GeneticProgressEvent] = []
        evaluation_events: list[str] = []
        result = FactorGeneticSearch(
            _small_config(), backend="sequential", batch_size=2
        ).run(
            _genetic_context(),
            evaluator=_RecordingEvaluator("selection", evaluation_events),
            holdout_evaluator=_RecordingEvaluator("holdout", evaluation_events),
            holdout_top_k=2,
            model_evaluator=_RecordingEvaluator("model", evaluation_events),
            model_top_k=2,
            progress_callback=events.append,
        )

        self.assertTrue(events)
        self.assertEqual(
            {event.stage for event in events}, {"selection", "holdout", "model"}
        )
        self.assertTrue(
            any(
                event.stage == "selection" and event.completed < event.total
                for event in events
            )
        )
        self.assertTrue(
            all(
                event.successful + event.failed == event.completed
                and 0 < event.completed <= event.total
                for event in events
            )
        )
        final_by_stage_generation: dict[tuple[str, int | None], GeneticProgressEvent] = {}
        for event in events:
            final_by_stage_generation[(event.stage, event.generation)] = event
        self.assertTrue(
            all(
                event.completed == event.total
                for event in final_by_stage_generation.values()
            )
        )
        selection_events = [event for event in events if event.stage == "selection"]
        self.assertEqual(
            selection_events[-1].selection_evaluations, len(result.candidates)
        )

    def test_process_progress_callback_runs_in_main_process_per_batch(self) -> None:
        """多进程搜索应在主进程按完成批次触发细粒度进度事件。"""

        events: list[GeneticProgressEvent] = []
        result = FactorGeneticSearch(
            replace(_small_config(), max_generations=1),
            backend="process",
            n_jobs=2,
            batch_size=2,
        ).run(
            _genetic_context(),
            evaluator=_CountingEvaluator(),
            holdout_top_k=0,
            progress_callback=events.append,
        )

        self.assertGreater(len(events), 1)
        self.assertTrue(all(event.stage == "selection" for event in events))
        self.assertEqual(events[-1].completed, events[-1].total)
        self.assertEqual(events[-1].selection_evaluations, len(result.candidates))

    def test_persistent_pool_keeps_configured_capacity_after_small_first_batch(self) -> None:
        """小首批不能缩小 worker 上限，且后续调用必须复用同一个进程池。"""

        context = _genetic_context()
        candidates = PipelineGrid(
            ["signal", "volume"], [[op("cs_rank")]]
        ).generate()
        session = ProcessBackend(n_jobs=2).open_session(
            context, IcEvaluator(), batch_size=16
        )

        with session:
            session.run(candidates[:1])
            self.assertIsNotNone(session._executor)
            self.assertEqual(session._executor._max_workers, 2)
            executor_identity = id(session._executor)
            session.run(candidates[1:])
            self.assertEqual(id(session._executor), executor_identity)

    def test_parameter_mutation_without_parameters_is_not_mislabeled(self) -> None:
        """没有可变参数时不得隐式执行并误记为参数变异。"""

        config = GeneticSearchConfig(
            sources=("signal", "volume"),
            operator_parameters={"cs_rank": {}},
            population_size=8,
            max_generations=2,
            max_evaluations=16,
            initial_max_depth=1,
            max_depth=3,
            max_nodes=4,
            max_lookback=0,
            min_coverage=0.2,
            target_coverage=0.5,
            crossover_probability=0.0,
            subtree_mutation_probability=0.0,
            parameter_mutation_probability=1.0,
            reproduction_probability=0.0,
            patience=2,
            random_seed=9,
        )

        result = FactorGeneticSearch(config).run(
            _genetic_context(), evaluator=_CountingEvaluator(), holdout_top_k=0
        )

        self.assertNotIn(
            "parameter_mutation",
            set(result.leaderboard["genetic_operation"]),
        )

    def test_evaluation_budget_is_a_hard_limit(self) -> None:
        """初始种群超过预算时只能评价稳定顺序中的前 max_evaluations 个候选。"""

        config = _small_config()
        limited = replace(config, max_evaluations=5)
        result = FactorGeneticSearch(limited).run(
            _genetic_context(), evaluator=_CountingEvaluator(), holdout_top_k=0
        )

        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(int(result.history["total_evaluations"].max()), 5)

    def test_no_eligible_candidate_and_worker_failures_are_reported(self) -> None:
        """覆盖率不合格与候选异常都应安全终止搜索并给出可审计结果。"""

        context = _genetic_context()
        ineligible = FactorGeneticSearch(_small_config()).run(
            context,
            evaluator=_CountingEvaluator(coverage=0.0),
            holdout_top_k=0,
        )
        self.assertFalse(ineligible.leaderboard["eligible"].any())
        with self.assertRaisesRegex(ValueError, "没有满足"):
            _ = ineligible.best_candidate

        failed = FactorGeneticSearch(_small_config()).run(
            context,
            evaluator=_FailingEvaluator(),
            holdout_top_k=0,
        )
        self.assertTrue(failed.leaderboard.empty)
        self.assertEqual(len(failed.errors), len(failed.candidates))
        self.assertTrue(failed.errors["error"].str.contains("expected").all())

    def test_holdout_runs_only_after_all_selection_evaluations(self) -> None:
        """预先入选候选的 holdout 调用必须全部发生在进化评价结束之后。"""

        events: list[str] = []
        result = FactorGeneticSearch(_small_config()).run(
            _genetic_context(),
            evaluator=_RecordingEvaluator("selection", events),
            holdout_evaluator=_RecordingEvaluator("holdout", events),
            holdout_top_k=2,
            model_evaluator=_RecordingEvaluator("model", events),
            model_top_k=2,
        )

        first_holdout = events.index("holdout")
        first_model = events.index("model")
        self.assertTrue(all(event == "selection" for event in events[:first_holdout]))
        self.assertTrue(
            all(event == "holdout" for event in events[first_holdout:first_model])
        )
        self.assertTrue(all(event == "model" for event in events[first_model:]))
        self.assertEqual(events.count("holdout"), 2)
        self.assertEqual(events.count("model"), 2)
        self.assertEqual(result.leaderboard["holdout_rank_ic"].notna().sum(), 2)
        self.assertEqual(result.leaderboard["model_rank_ic"].notna().sum(), 2)


if __name__ == "__main__":
    unittest.main()
