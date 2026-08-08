"""因子网格搜索编排器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .backends import (
    CandidateTaskResult,
    ExecutionBackend,
    ProcessBackend,
    SequentialBackend,
)
from .context import SearchContext
from .evaluators import CandidateEvaluator, HoldoutIcEvaluator, IcEvaluator
from .result import FactorSearchResult
from .space import FactorCandidate, SearchSpace


@dataclass(frozen=True)
class FactorGridSearch:
    """控制候选规模、并行后端、覆盖率过滤和两阶段评价。"""

    backend: str | ExecutionBackend = "sequential"
    n_jobs: int = -1
    batch_size: int = 32
    max_candidates: int = 10_000
    max_depth: int | None = None
    max_lookback: int | None = None
    min_coverage: float = 0.5
    objective: str = "selection_oriented_rank_ic"

    def __post_init__(self) -> None:
        """校验执行规模、覆盖率和只允许 selection 指标排序的安全约束。"""

        if self.batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates 必须是正整数")
        if not 0 <= self.min_coverage <= 1:
            raise ValueError("min_coverage 必须在 0 到 1 之间")
        if not self.objective.startswith("selection_"):
            raise ValueError("objective 必须是 selection 指标，禁止使用 holdout 选择因子")

    def _backend(self) -> ExecutionBackend:
        """把后端名称解析为执行对象，或直接返回调用方注入的自定义后端。"""

        if not isinstance(self.backend, str):
            return self.backend
        if self.backend == "sequential":
            return SequentialBackend()
        if self.backend == "process":
            return ProcessBackend(self.n_jobs)
        raise ValueError("backend 必须是 'sequential'、'process' 或 ExecutionBackend")

    def _validate_candidates(
        self, candidates: Sequence[FactorCandidate]
    ) -> None:
        """在执行前校验候选非空、实际数量、表达式深度和历史长度。

        参数：
            candidates: 已生成并完成去重的全部候选因子。
        """

        if not candidates:
            raise ValueError("搜索空间没有生成任何候选")
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"候选数量 {len(candidates)} 超过上限 {self.max_candidates}"
            )
        too_deep = [
            candidate for candidate in candidates
            if self.max_depth is not None and candidate.depth > self.max_depth
        ]
        if too_deep:
            raise ValueError(
                f"存在 {len(too_deep)} 个候选超过最大深度 {self.max_depth}"
            )
        too_long = [
            candidate for candidate in candidates
            if self.max_lookback is not None and candidate.lookback > self.max_lookback
        ]
        if too_long:
            raise ValueError(
                f"存在 {len(too_long)} 个候选超过最大回看长度 {self.max_lookback}"
            )

    def _build_leaderboard(
        self,
        candidates: Sequence[FactorCandidate],
        results: Sequence[CandidateTaskResult],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """合并候选身份与初筛结果，隔离错误并按 selection 目标稳定排序。

        参数：
            candidates: 用于补充表达式、深度和回看长度的候选集。
            results: 与候选 ID 对应的初筛指标或隔离错误。
        """

        candidates_by_id = {candidate.factor_id: candidate for candidate in candidates}
        rows: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        for result in results:
            candidate = candidates_by_id[result.factor_id]
            identity = {
                "factor_id": candidate.factor_id,
                "expression": candidate.canonical,
                "expression_str": candidate.expression_str,
                "depth": candidate.depth,
                "lookback": candidate.lookback,
                "elapsed_seconds": result.elapsed_seconds,
            }
            if result.error is not None:
                errors.append({**identity, "stage": "screening", "error": result.error})
            else:
                rows.append({**identity, **result.metrics})
        leaderboard = pd.DataFrame(rows)
        error_frame = pd.DataFrame(errors)
        if leaderboard.empty:
            return leaderboard, error_frame
        if self.objective not in leaderboard.columns:
            raise ValueError(f"评价结果不包含排序目标 {self.objective!r}")
        objective = pd.to_numeric(leaderboard[self.objective], errors="coerce")
        coverage_source = (
            leaderboard["selection_coverage"]
            if "selection_coverage" in leaderboard
            else pd.Series(np.nan, index=leaderboard.index)
        )
        coverage = pd.to_numeric(coverage_source, errors="coerce")
        leaderboard["eligible"] = (
            np.isfinite(objective.to_numpy(dtype=float))
            & np.isfinite(coverage.to_numpy(dtype=float))
            & (coverage >= self.min_coverage)
        )
        return (
            leaderboard.sort_values(
                ["eligible", self.objective, "factor_id"],
                ascending=[False, False, True],
                kind="stable",
            ).reset_index(drop=True),
            error_frame,
        )

    def run(
        self,
        context: SearchContext,
        space: SearchSpace,
        *,
        evaluator: CandidateEvaluator | None = None,
        holdout_evaluator: CandidateEvaluator | None = None,
        holdout_top_k: int = 1,
        model_evaluator: CandidateEvaluator | None = None,
        model_top_k: int = 0,
    ) -> FactorSearchResult:
        """执行全量 selection 评价、Top K holdout 报告和可选模型复验。

        初筛使用调用方注入的 ``evaluator``，未注入时才默认计算横截面 IC；候选
        始终只按 selection 指标排序。holdout 仅评价 selection 预先入选的 Top K，
        不参与候选方向或名次确定；模型评价同样只作用于预先入选的 Top K。

        参数：
            context: 一次性准备的日频数据、目标和日期切分。
            space: 按确定顺序估算并生成候选的搜索空间。
            evaluator: 全量 selection 候选的初筛评价器；缺省使用 IC。
            holdout_evaluator: 预先入选候选的 holdout 评价器；缺省使用 IC。
            holdout_top_k: 要报告 holdout 指标的 selection 前 K 名数量，缺省为 1。
            model_evaluator: 可选的现有模型实验评价器；为空时跳过模型复验。
            model_top_k: 要进行模型复验的 selection 前 K 名数量，缺省为 0，即不复验。
        """

        if holdout_top_k < 0:
            raise ValueError("holdout_top_k 不能为负数")
        if model_top_k < 0:
            raise ValueError("model_top_k 不能为负数")
        estimated = space.estimate_size()
        if estimated > self.max_candidates:
            raise ValueError(
                f"候选数量上界 {estimated} 超过上限 {self.max_candidates}，"
                "已在构造全部表达式前终止"
            )
        candidates = space.generate()
        self._validate_candidates(candidates)
        backend = self._backend()
        screening_results = backend.run(
            candidates,
            context,
            evaluator or IcEvaluator(),
            self.batch_size,
        )
        leaderboard, errors = self._build_leaderboard(candidates, screening_results)

        if (
            holdout_top_k > 0
            and context.holdout_mask.any()
            and not leaderboard.empty
        ):
            holdout_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(holdout_top_k)
            holdout_set = set(holdout_ids.astype(str))
            holdout_candidates = [
                candidate
                for candidate in candidates
                if candidate.factor_id in holdout_set
            ]
            holdout_results = backend.run(
                holdout_candidates,
                context,
                holdout_evaluator or HoldoutIcEvaluator(),
                self.batch_size,
            )
            holdout_rows: list[dict[str, object]] = []
            holdout_errors: list[dict[str, object]] = []
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
                    holdout_errors.append(
                        {
                            "factor_id": result.factor_id,
                            "stage": "holdout",
                            "error": result.error,
                            "elapsed_seconds": result.elapsed_seconds,
                        }
                    )
            if holdout_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(holdout_rows), on="factor_id", how="left"
                )
                if "direction" in leaderboard and "holdout_ic" in leaderboard:
                    leaderboard["holdout_oriented_ic"] = (
                        leaderboard["holdout_ic"] * leaderboard["direction"]
                    )
                if (
                    "direction" in leaderboard
                    and "holdout_rank_ic" in leaderboard
                ):
                    leaderboard["holdout_oriented_rank_ic"] = (
                        leaderboard["holdout_rank_ic"] * leaderboard["direction"]
                    )
            if holdout_errors:
                errors = pd.concat(
                    [errors, pd.DataFrame(holdout_errors)],
                    ignore_index=True,
                    sort=False,
                )

        if model_evaluator is not None and model_top_k > 0 and not leaderboard.empty:
            selected_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(model_top_k)
            selected_set = set(selected_ids.astype(str))
            selected = [
                candidate for candidate in candidates if candidate.factor_id in selected_set
            ]
            model_results = backend.run(
                selected, context, model_evaluator, self.batch_size
            )
            model_rows: list[dict[str, object]] = []
            model_errors: list[dict[str, object]] = []
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
                    model_errors.append(
                        {
                            "factor_id": result.factor_id,
                            "stage": "model",
                            "error": result.error,
                            "elapsed_seconds": result.elapsed_seconds,
                        }
                    )
            if model_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(model_rows), on="factor_id", how="left"
                )
            if model_errors:
                errors = pd.concat(
                    [errors, pd.DataFrame(model_errors)], ignore_index=True, sort=False
                )

        return FactorSearchResult(
            candidates=tuple(candidates),
            leaderboard=leaderboard,
            errors=errors,
            objective=self.objective,
        )
