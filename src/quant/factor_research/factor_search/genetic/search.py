"""遗传搜索主流程：适应度评价、去重、选择与逐代进化。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..backends import (
    CandidateTaskResult,
    ExecutionBackend,
    ExecutionSession,
    ProcessBackend,
    SequentialBackend,
)
from ..context import SearchContext
from ..evaluators import CandidateEvaluator, HoldoutIcEvaluator, IcEvaluator
from ..space import FactorCandidate
from .config import GeneticSearchConfig
from .events import (
    GeneticProgressCallback,
    GeneticSearchResult,
    _Provenance,
)
from .generator import _ExpressionGenerator
from .session import (
    _BackendSessionAdapter,
    _ProgressReporter,
    _run_session_with_progress,
)
from .trees import _tree_operators


@dataclass(frozen=True)
class FactorGeneticSearch:
    """以主进程确定性进化、执行后端并行评价候选的遗传因子搜索器。"""

    config: GeneticSearchConfig = field(default_factory=GeneticSearchConfig)
    backend: str | ExecutionBackend = "sequential"
    n_jobs: int = -1
    batch_size: int = 16

    def __post_init__(self) -> None:
        """校验执行后端名称、worker 数和候选批次大小。"""

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size 必须是正整数")
        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValueError("n_jobs 必须是 -1 或正整数")
        if isinstance(self.backend, str) and self.backend not in {"sequential", "process"}:
            raise ValueError("backend 必须是 'sequential'、'process' 或 ExecutionBackend")

    def _backend(self) -> SequentialBackend | ProcessBackend | ExecutionBackend:
        """把后端名称解析为可创建持久化执行会话的后端对象。"""

        if not isinstance(self.backend, str):
            return self.backend
        if self.backend == "sequential":
            return SequentialBackend()
        return ProcessBackend(self.n_jobs)

    def _open_session(
        self,
        backend: SequentialBackend | ProcessBackend | ExecutionBackend,
        context: SearchContext,
        evaluator: CandidateEvaluator,
    ) -> ExecutionSession:
        """优先创建持久化会话，并兼容仅实现旧批量接口的自定义后端。

        参数：
            backend: 已解析的内置或调用方注入执行后端。
            context: 全部遗传代次共享的只读搜索上下文。
            evaluator: 把候选值转换为 selection 或 holdout 指标的评价器。
        """

        open_session = getattr(backend, "open_session", None)
        if callable(open_session):
            return open_session(context, evaluator, self.batch_size)
        return _BackendSessionAdapter(backend, context, evaluator, self.batch_size)

    def _fitness_row(
        self,
        candidate: FactorCandidate,
        result: CandidateTaskResult,
        provenance: _Provenance,
    ) -> dict[str, object]:
        """把原始 selection 指标、复杂度惩罚和来源合并为排行榜行。

        参数：
            candidate: 当前已完成规范化的因子表达式。
            result: worker 返回的原始 selection 指标或隔离错误。
            provenance: 候选首次出现时的代次、生成方式和父代。
        """

        identity: dict[str, object] = {
            "factor_id": candidate.factor_id,
            "expression": candidate.canonical,
            "expression_str": candidate.expression_str,
            "node_count": candidate.node_count,
            "depth": candidate.depth,
            "lookback": candidate.lookback,
            "first_generation": provenance.generation,
            "genetic_operation": provenance.operation,
            "parent_ids": ",".join(provenance.parent_ids),
            "elapsed_seconds": result.elapsed_seconds,
        }
        if result.error is not None:
            return {
                **identity,
                "eligible": False,
                "fitness": float("-inf"),
                "error": result.error,
            }
        metrics: dict[str, object] = dict(result.metrics)
        objective = float(metrics.get(self.config.objective, float("nan")))
        rank_std = float(metrics.get("selection_rank_ic_std", float("nan")))
        rank_dates = float(metrics.get("selection_rank_ic_dates", 0.0))
        coverage = float(metrics.get("selection_coverage", float("nan")))
        standard_error = 0.0 if self.config.stability_weight == 0 else (
            rank_std / math.sqrt(rank_dates)
            if math.isfinite(rank_std) and rank_dates > 0
            else float("inf")
        )
        stability_penalty = self.config.stability_weight * standard_error
        length_penalty = self.config.length_penalty * max(
            0, candidate.node_count - self.config.free_node_count
        )
        depth_penalty = self.config.depth_penalty * max(
            0, candidate.depth - self.config.free_depth
        )
        coverage_penalty = (
            self.config.coverage_penalty
            * max(0.0, self.config.target_coverage - coverage)
            if math.isfinite(coverage)
            else float("inf")
        )
        eligible = (
            math.isfinite(objective)
            and math.isfinite(coverage)
            and coverage >= self.config.min_coverage
            and math.isfinite(standard_error)
        )
        fitness = (
            objective
            - stability_penalty
            - length_penalty
            - depth_penalty
            - coverage_penalty
            if eligible
            else float("-inf")
        )
        return {
            **identity,
            **metrics,
            "rank_ic_standard_error": standard_error,
            "stability_penalty": stability_penalty,
            "length_penalty": length_penalty,
            "depth_penalty": depth_penalty,
            "coverage_penalty": coverage_penalty,
            "fitness": fitness,
            "eligible": eligible,
        }

    @staticmethod
    def _sort_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        """按适应度、长度、深度和稳定 ID 对排行榜执行确定性排序。

        参数：
            rows: 已计算适应度且包含候选身份的排行榜记录。
        """

        return sorted(
            rows,
            key=lambda row: (
                not bool(row["eligible"]),
                -float(row["fitness"]),
                int(row["node_count"]),
                int(row["depth"]),
                str(row["factor_id"]),
            ),
        )

    @staticmethod
    def _deduplicate_equivalent_rows(
        rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        """按 selection 方向值或逐日秩指纹保留最简单候选。

        参数：
            rows: 已计算适应度的当前代或全量候选排行榜行；没有等价指纹的
                自定义评价结果保持原样。

        返回：
            每个等价类只保留节点数最少、再按深度和适应度稳定决胜的排行榜行，
            最终仍按正式适应度顺序排列。
        """

        simplest_first = sorted(
            rows,
            key=lambda row: (
                int(row["node_count"]),
                int(row["depth"]),
                not bool(row["eligible"]),
                -float(row["fitness"]),
                str(row["factor_id"]),
            ),
        )
        retained: list[dict[str, object]] = []
        seen_values: set[str] = set()
        seen_ranks: set[str] = set()
        for row in simplest_first:
            value_key = str(
                row.get("selection_oriented_value_fingerprint", "")
            )
            rank_key = str(
                row.get("selection_oriented_rank_fingerprint", "")
            )
            if (value_key and value_key in seen_values) or (
                rank_key and rank_key in seen_ranks
            ):
                continue
            retained.append(row)
            if value_key:
                seen_values.add(value_key)
            if rank_key:
                seen_ranks.add(rank_key)
        return FactorGeneticSearch._sort_rows(retained)

    def _tournament(
        self,
        ranked: Sequence[FactorCandidate],
        rng: np.random.Generator,
    ) -> FactorCandidate:
        """从随机抽取的合格候选中返回排名最靠前的父代。

        参数：
            ranked: 已按适应度稳定排序的合格候选序列。
            rng: 只由主进程使用的确定性随机数生成器。
        """

        size = min(self.config.tournament_size, len(ranked))
        indices = rng.choice(len(ranked), size=size, replace=False)
        return ranked[min(int(index) for index in indices)]

    def _inject_missing_coverage(
        self,
        selected: dict[str, FactorCandidate],
        provenance: dict[str, _Provenance],
        generator: _ExpressionGenerator,
        rng: np.random.Generator,
        generation: int,
    ) -> None:
        """向下一代优先注入缺失数据源和缺失算子的随机候选。

        数据源和算子分别按主进程随机顺序补全，并在每次注入后重新累计覆盖，
        因而一个随机表达式可以同时补足多个缺失项。若种群容量或表达式硬限制
        不足，则保持已获得的最大覆盖，不突破既有规模与复杂度约束。

        参数：
            selected: 已包含本轮精英、并将被原地补充的下一代候选映射。
            provenance: 与 ``selected`` 同步记录首次生成方式的来源映射。
            generator: 负责生成并校验覆盖候选的表达式生成器。
            rng: 控制缺失项顺序和表达式结构的主进程随机数生成器。
            generation: 正在构造的下一代零基代次编号。

        返回：
            无；通过原地修改 ``selected`` 和 ``provenance`` 返回补全结果。
        """

        covered_sources = set().union(
            *(candidate.expression.columns for candidate in selected.values())
        )
        covered_operators = set().union(
            *(_tree_operators(candidate.expression) for candidate in selected.values())
        )

        source_order = list(self.config.sources)
        rng.shuffle(source_order)
        for source in source_order:
            if len(selected) >= self.config.population_size:
                return
            if source in covered_sources:
                continue
            node = generator.random_tree_covering_source(source, rng)
            candidate = FactorCandidate(node)
            if candidate.factor_id in selected:
                continue
            selected[candidate.factor_id] = candidate
            provenance[candidate.factor_id] = _Provenance(
                generation, "coverage_source"
            )
            covered_sources.update(node.columns)
            covered_operators.update(_tree_operators(node))

        operator_order = list(self.config.operator_parameters)
        rng.shuffle(operator_order)
        for operator in operator_order:
            if len(selected) >= self.config.population_size:
                return
            if operator in covered_operators:
                continue
            node = generator.random_tree_with_root(operator, rng)
            if node is None:
                continue
            candidate = FactorCandidate(node)
            if candidate.factor_id in selected:
                continue
            selected[candidate.factor_id] = candidate
            provenance[candidate.factor_id] = _Provenance(
                generation, "coverage_operator"
            )
            covered_sources.update(node.columns)
            covered_operators.update(_tree_operators(node))

    def _next_population(
        self,
        ranked: Sequence[FactorCandidate],
        generator: _ExpressionGenerator,
        rng: np.random.Generator,
        generation: int,
    ) -> tuple[list[FactorCandidate], dict[str, _Provenance]]:
        """执行精英保留、锦标赛选择、交叉、变异和随机移民。

        参数：
            ranked: 当前代按适应度排序的合格候选。
            generator: 负责构造和校验表达式树的生成器。
            rng: 控制全部遗传选择且只在主进程使用的随机数生成器。
            generation: 即将生成的新种群代次编号。
        """

        elite_count = max(1, math.ceil(len(ranked) * self.config.elite_ratio))
        selected: dict[str, FactorCandidate] = {
            candidate.factor_id: candidate for candidate in ranked[:elite_count]
        }
        provenance: dict[str, _Provenance] = {
            candidate.factor_id: _Provenance(
                generation, "elite", (candidate.factor_id,)
            )
            for candidate in ranked[:elite_count]
        }
        self._inject_missing_coverage(
            selected,
            provenance,
            generator,
            rng,
            generation,
        )
        attempts = 0
        limit = self.config.population_size * 100
        boundaries = np.cumsum(
            [
                self.config.crossover_probability,
                self.config.subtree_mutation_probability,
                self.config.parameter_mutation_probability,
                self.config.reproduction_probability,
            ]
        )
        while len(selected) < self.config.population_size and attempts < limit:
            parent = self._tournament(ranked, rng)
            draw = rng.random()
            operation = "reproduction"
            parent_ids = (parent.factor_id,)
            node = parent.expression
            if draw < boundaries[0]:
                other = self._tournament(ranked, rng)
                node = generator.crossover(parent.expression, other.expression, rng)
                operation = "crossover"
                parent_ids = (parent.factor_id, other.factor_id)
            elif draw < boundaries[1]:
                node = generator.subtree_mutation(parent.expression, rng)
                operation = "subtree_mutation"
            elif draw < boundaries[2]:
                node = generator.parameter_mutation(parent.expression, rng)
                operation = "parameter_mutation"
            candidate = FactorCandidate(node)
            if generator.valid(node) and candidate.factor_id not in selected:
                selected[candidate.factor_id] = candidate
                provenance[candidate.factor_id] = _Provenance(
                    generation, operation, parent_ids
                )
            attempts += 1

        # 搜索空间较小时，随机移民比重复个体更能维持有效种群多样性。
        immigrant_attempts = 0
        while (
            len(selected) < self.config.population_size
            and immigrant_attempts < limit
        ):
            depth = int(rng.integers(self.config.max_depth + 1))
            node = generator.random_tree(rng, depth, force_operator=depth > 0)
            candidate = FactorCandidate(node)
            if generator.valid(node) and candidate.factor_id not in selected:
                selected[candidate.factor_id] = candidate
                provenance[candidate.factor_id] = _Provenance(generation, "immigrant")
            immigrant_attempts += 1
        return list(selected.values()), provenance

    def run(
        self,
        context: SearchContext,
        *,
        evaluator: CandidateEvaluator | None = None,
        holdout_evaluator: CandidateEvaluator | None = None,
        holdout_top_k: int = 1,
        model_evaluator: CandidateEvaluator | None = None,
        model_top_k: int = 0,
        progress_callback: GeneticProgressCallback | None = None,
    ) -> GeneticSearchResult:
        """并行进化，并在结束后评价 selection 入选候选的 holdout 和模型指标。

        参数：
            context: 一次性准备的日频行情、目标和严格隔离的日期掩码。
            evaluator: 每代候选的 selection 评价器；缺省使用横截面 IC。
            holdout_evaluator: 最终候选的 holdout 评价器；缺省使用横截面 IC。
            holdout_top_k: 搜索结束后披露 holdout 指标的 selection 前 K 名数量。
            model_evaluator: 可选的现有模型实验评价器；为空时跳过模型复验。
            model_top_k: 搜索结束后进行模型复验的 selection 前 K 名数量。
            progress_callback: 可选主进程进度回调，每完成一个候选批次触发一次。
        """

        for name, value in {
            "holdout_top_k": holdout_top_k,
            "model_top_k": model_top_k,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback 必须可调用或为 None")
        missing_sources = set(self.config.sources).difference(context.daily.columns)
        if missing_sources:
            raise ValueError(f"遗传搜索日频数据缺少源列: {sorted(missing_sources)}")
        generator = _ExpressionGenerator(self.config)
        rng = np.random.default_rng(self.config.random_seed)
        initial_nodes = generator.initial_population(rng)
        population = [FactorCandidate(node) for node in initial_nodes]
        provenance: dict[str, _Provenance] = {
            candidate.factor_id: _Provenance(0, "initial")
            for candidate in population
        }
        candidates: dict[str, FactorCandidate] = {}
        cache: dict[str, CandidateTaskResult] = {}
        errors: dict[tuple[str, str], dict[str, object]] = {}
        history: list[dict[str, object]] = []
        best_fitness = float("-inf")
        stale_generations = 0
        backend = self._backend()
        selection_evaluator = evaluator or IcEvaluator()
        if isinstance(selection_evaluator, IcEvaluator):
            selection_evaluator = replace(
                selection_evaluator,
                include_equivalence_fingerprints=True,
            )
        with self._open_session(
            backend, context, selection_evaluator
        ) as session:
            for generation in range(self.config.max_generations):
                remaining = self.config.max_evaluations - len(cache)
                new_candidates = [
                    candidate
                    for candidate in population
                    if candidate.factor_id not in cache
                ][: max(0, remaining)]
                if new_candidates:
                    for candidate in new_candidates:
                        candidates[candidate.factor_id] = candidate
                    reporter = (
                        _ProgressReporter(
                            progress_callback,
                            stage="selection",
                            generation=generation + 1,
                            config=self.config,
                            selection_base=len(cache),
                        )
                        if progress_callback is not None
                        else None
                    )
                    for result in _run_session_with_progress(
                        session, new_candidates, reporter
                    ):
                        cache[result.factor_id] = result
                population = [
                    candidate
                    for candidate in population
                    if candidate.factor_id in cache
                ]
                rows = [
                    self._fitness_row(
                        candidate,
                        cache[candidate.factor_id],
                        provenance[candidate.factor_id],
                    )
                    for candidate in population
                ]
                ranked_rows = self._deduplicate_equivalent_rows(rows)
                for row in ranked_rows:
                    if "error" in row:
                        factor_id = str(row["factor_id"])
                        errors[(factor_id, "screening")] = {
                            **row,
                            "stage": "screening",
                        }
                eligible_ids = [
                    str(row["factor_id"])
                    for row in ranked_rows
                    if bool(row["eligible"])
                ]
                generation_best = (
                    float(ranked_rows[0]["fitness"])
                    if ranked_rows and bool(ranked_rows[0]["eligible"])
                    else float("-inf")
                )
                history.append(
                    {
                        "generation": generation,
                        "population": len(population),
                        "new_evaluations": len(new_candidates),
                        "total_evaluations": len(cache),
                        "eligible": len(eligible_ids),
                        "best_fitness": generation_best,
                        "best_factor_id": eligible_ids[0] if eligible_ids else None,
                    }
                )
                if generation_best > best_fitness + self.config.min_improvement:
                    best_fitness = generation_best
                    stale_generations = 0
                else:
                    stale_generations += 1
                if (
                    not eligible_ids
                    or len(cache) >= self.config.max_evaluations
                    or stale_generations >= self.config.patience
                    or generation + 1 >= self.config.max_generations
                ):
                    break
                ranked_candidates = [candidates.get(factor_id) for factor_id in eligible_ids]
                # 当前代新产生且合格的候选均已进入 candidates；精英旧候选同样存在。
                parents = [candidate for candidate in ranked_candidates if candidate is not None]
                population, next_provenance = self._next_population(
                    parents, generator, rng, generation + 1
                )
                for factor_id, item in next_provenance.items():
                    provenance.setdefault(factor_id, item)

        all_rows = [
            self._fitness_row(candidate, cache[factor_id], provenance[factor_id])
            for factor_id, candidate in candidates.items()
            if factor_id in cache and cache[factor_id].error is None
        ]
        final_rows = self._deduplicate_equivalent_rows(all_rows)
        leaderboard = pd.DataFrame(final_rows)
        error_frame = pd.DataFrame(list(errors.values()))

        if (
            holdout_top_k > 0
            and context.holdout_mask.any()
            and not leaderboard.empty
        ):
            holdout_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(holdout_top_k)
            holdout_candidates = [
                candidates[str(factor_id)] for factor_id in holdout_ids
            ]
            with self._open_session(
                backend, context, holdout_evaluator or HoldoutIcEvaluator()
            ) as holdout_session:
                holdout_reporter = (
                    _ProgressReporter(
                        progress_callback,
                        stage="holdout",
                        generation=None,
                        config=self.config,
                        selection_base=len(cache),
                    )
                    if progress_callback is not None
                    else None
                )
                holdout_results = _run_session_with_progress(
                    holdout_session, holdout_candidates, holdout_reporter
                )
            holdout_rows: list[dict[str, object]] = []
            for result in holdout_results:
                if result.error is None:
                    holdout_rows.append(
                        {
                            "factor_id": result.factor_id,
                            "holdout_elapsed_seconds": result.elapsed_seconds,
                            **result.metrics,
                        }
                    )
                else:
                    errors[(result.factor_id, "holdout")] = {
                        "factor_id": result.factor_id,
                        "stage": "holdout",
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
            if holdout_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(holdout_rows), on="factor_id", how="left"
                )
                if "direction" in leaderboard and "holdout_rank_ic" in leaderboard:
                    leaderboard["holdout_oriented_rank_ic"] = (
                        leaderboard["holdout_rank_ic"] * leaderboard["direction"]
                    )
                if "direction" in leaderboard and "holdout_ic" in leaderboard:
                    leaderboard["holdout_oriented_ic"] = (
                        leaderboard["holdout_ic"] * leaderboard["direction"]
                    )
            error_frame = pd.DataFrame(list(errors.values()))

        if model_evaluator is not None and model_top_k > 0 and not leaderboard.empty:
            model_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(model_top_k)
            model_candidates = [
                candidates[str(factor_id)] for factor_id in model_ids
            ]
            with self._open_session(
                backend, context, model_evaluator
            ) as model_session:
                model_reporter = (
                    _ProgressReporter(
                        progress_callback,
                        stage="model",
                        generation=None,
                        config=self.config,
                        selection_base=len(cache),
                    )
                    if progress_callback is not None
                    else None
                )
                model_results = _run_session_with_progress(
                    model_session, model_candidates, model_reporter
                )
            model_rows: list[dict[str, object]] = []
            for result in model_results:
                if result.error is None:
                    model_rows.append(
                        {
                            "factor_id": result.factor_id,
                            "model_elapsed_seconds": result.elapsed_seconds,
                            **result.metrics,
                        }
                    )
                else:
                    errors[(result.factor_id, "model")] = {
                        "factor_id": result.factor_id,
                        "stage": "model",
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
            if model_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(model_rows), on="factor_id", how="left"
                )
            error_frame = pd.DataFrame(list(errors.values()))

        return GeneticSearchResult(
            candidates=tuple(candidates.values()),
            leaderboard=leaderboard.reset_index(drop=True),
            errors=error_frame.reset_index(drop=True),
            objective="fitness",
            history=pd.DataFrame(history),
        )
