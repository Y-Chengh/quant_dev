"""候选执行后端：确定性单进程与 Windows 兼容的多进程批处理。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Protocol, Sequence

from factor_research.factor_dsl import DailyFactorFrame

from .context import SearchContext
from .evaluators import CandidateEvaluator
from .space import FactorCandidate


@dataclass(frozen=True)
class CandidateTaskResult:
    """一个候选在 worker 内的指标或隔离后的错误。"""

    factor_id: str
    metrics: dict[str, float]
    elapsed_seconds: float
    error: str | None = None


class ExecutionBackend(Protocol):
    """约束候选执行后端必须提供保持输入顺序的批量运行接口。"""

    def run(
        self,
        candidates: Sequence[FactorCandidate],
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> list[CandidateTaskResult]:
        """计算并评价候选，返回顺序必须与 candidates 一致。"""


def _evaluate_batch(
    candidates: Sequence[FactorCandidate],
    frame: DailyFactorFrame,
    context: SearchContext,
    evaluator: CandidateEvaluator,
) -> list[CandidateTaskResult]:
    """在共享 Frame 上依次计算一批候选，并隔离每个候选的异常。"""

    results: list[CandidateTaskResult] = []
    for candidate in candidates:
        started = perf_counter()
        try:
            values = frame.evaluate(candidate.expression, name=candidate.factor_id)
            metrics = evaluator.evaluate(candidate, values, context)
            results.append(
                CandidateTaskResult(
                    factor_id=candidate.factor_id,
                    metrics=metrics,
                    elapsed_seconds=perf_counter() - started,
                )
            )
        except Exception as exc:  # 单个非法候选不能终止整个大网格。
            results.append(
                CandidateTaskResult(
                    factor_id=candidate.factor_id,
                    metrics={},
                    elapsed_seconds=perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    # 公共子表达式在同一批次内只计算一次；批次结束后释放，限制大型搜索的峰值内存。
    frame.clear_cache()
    return results


@dataclass(frozen=True)
class SequentialBackend:
    """用于单元测试和调试的确定性后端。"""

    def run(
        self,
        candidates: Sequence[FactorCandidate],
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> list[CandidateTaskResult]:
        """在当前进程中按批次计算候选，供调试和确定性基准使用。"""

        frame = DailyFactorFrame(context.daily)
        results: list[CandidateTaskResult] = []
        for start in range(0, len(candidates), batch_size):
            results.extend(
                _evaluate_batch(
                    candidates[start : start + batch_size], frame, context, evaluator
                )
            )
        return results


_WORKER_FRAME: DailyFactorFrame | None = None
_WORKER_CONTEXT: SearchContext | None = None
_WORKER_EVALUATOR: CandidateEvaluator | None = None


def _initialize_worker(context: SearchContext, evaluator: CandidateEvaluator) -> None:
    """每个进程只反序列化一次日频上下文，而不是为每个候选重复传输。"""

    global _WORKER_FRAME, _WORKER_CONTEXT, _WORKER_EVALUATOR
    _WORKER_CONTEXT = context
    _WORKER_EVALUATOR = evaluator
    _WORKER_FRAME = DailyFactorFrame(context.daily)


def _worker_batch(candidates: Sequence[FactorCandidate]) -> list[CandidateTaskResult]:
    """使用进程初始化器保存的只读上下文评价一个候选批次。"""

    if _WORKER_FRAME is None or _WORKER_CONTEXT is None or _WORKER_EVALUATOR is None:
        raise RuntimeError("搜索 worker 尚未初始化")
    return _evaluate_batch(
        candidates, _WORKER_FRAME, _WORKER_CONTEXT, _WORKER_EVALUATOR
    )


@dataclass(frozen=True)
class ProcessBackend:
    """按候选批次并行的进程后端，适用于 pandas 计算和 Windows spawn。"""

    n_jobs: int = -1

    def _workers(self, candidate_count: int) -> int:
        """根据用户配置、CPU 数和批次数计算实际 worker 数量。"""

        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs 必须是 -1 或正整数")
        requested = (os.cpu_count() or 1) if self.n_jobs == -1 else self.n_jobs
        return max(1, min(int(requested), candidate_count))

    def run(
        self,
        candidates: Sequence[FactorCandidate],
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> list[CandidateTaskResult]:
        """把候选批次分派给进程池，并恢复为与输入候选一致的顺序。"""

        if not candidates:
            return []
        batches = [
            list(candidates[start : start + batch_size])
            for start in range(0, len(candidates), batch_size)
        ]
        workers = self._workers(len(batches))
        by_id: dict[str, CandidateTaskResult] = {}
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(context, evaluator),
        ) as executor:
            futures = [executor.submit(_worker_batch, batch) for batch in batches]
            for future in as_completed(futures):
                for result in future.result():
                    by_id[result.factor_id] = result
        return [by_id[candidate.factor_id] for candidate in candidates]
