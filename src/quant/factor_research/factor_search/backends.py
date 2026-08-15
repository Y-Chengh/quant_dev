"""候选执行后端：确定性单进程与 Windows 兼容的多进程批处理。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from time import perf_counter
from types import TracebackType
from typing import Callable, Protocol, Sequence

from quant.factor_research.factor_dsl import DailyFactorFrame

from .context import SearchContext
from .evaluators import CandidateEvaluator
from .space import FactorCandidate


BatchProgressCallback = Callable[[int, int, int], None]
"""批次进度回调：依次接收累计完成数、总数和累计失败数。"""


@dataclass(frozen=True)
class CandidateTaskResult:
    """一个候选在 worker 内的指标或隔离后的错误。"""

    factor_id: str
    metrics: dict[str, float | str]
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
        """计算并评价候选，返回顺序必须与 candidates 一致。

        参数：
            candidates: 按期望输出顺序排列的候选因子。
            context: 全部候选共享的只读数据和日期切分。
            evaluator: 把候选值转换为排行指标的评价器。
            batch_size: 每个计算批次包含的最大候选数。
        """


class ExecutionSession(Protocol):
    """约束可跨多个候选批次复用资源的执行会话。"""

    def __enter__(self) -> "ExecutionSession":
        """进入执行会话并返回自身。"""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开执行会话并释放资源。

        参数：
            exc_type: 会话内异常的类型；正常退出时为空。
            exc_value: 会话内异常对象；正常退出时为空。
            traceback: 会话内异常的回溯；正常退出时为空。
        """

    def run(
        self, candidates: Sequence[FactorCandidate]
    ) -> list[CandidateTaskResult]:
        """评价一批候选并按输入顺序返回结果。

        参数：
            candidates: 当前要评价且已经确定顺序的候选因子。
        """


def _evaluate_batch(
    candidates: Sequence[FactorCandidate],
    frame: DailyFactorFrame,
    context: SearchContext,
    evaluator: CandidateEvaluator,
) -> list[CandidateTaskResult]:
    """在共享 Frame 上依次计算一批候选，并隔离每个候选的异常。

    参数：
        candidates: 当前批次要执行的候选因子。
        frame: 批次内复用节点缓存的日频执行上下文。
        context: 目标、固定特征及日期掩码的只读容器。
        evaluator: 计算每个候选指标的评价器。
    """

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
        """在当前进程中按批次计算候选，供调试和确定性基准使用。

        参数：
            candidates: 按生成顺序排列的候选因子。
            context: 候选共享的搜索数据上下文。
            evaluator: 为候选计算数值指标的评价器。
            batch_size: 一次复用节点缓存的候选数量上限。
        """

        frame = DailyFactorFrame(context.daily)
        results: list[CandidateTaskResult] = []
        for start in range(0, len(candidates), batch_size):
            results.extend(
                _evaluate_batch(
                    candidates[start : start + batch_size], frame, context, evaluator
                )
            )
        return results

    def open_session(
        self,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> "SequentialExecutionSession":
        """创建可跨遗传代次复用日频执行上下文的串行会话。

        参数：
            context: 全部候选共享的只读搜索上下文。
            evaluator: 把候选值转换为筛选指标的评价器。
            batch_size: 每次共享节点缓存的最大候选数量。
        """

        return SequentialExecutionSession(context, evaluator, batch_size)


class SequentialExecutionSession:
    """在当前进程中复用搜索上下文的持久化执行会话。"""

    def __init__(
        self,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> None:
        """初始化串行执行会话。

        参数：
            context: 全部候选共享的只读搜索上下文。
            evaluator: 把候选值转换为筛选指标的评价器。
            batch_size: 每次共享节点缓存的最大候选数量。
        """

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size 必须是正整数")
        self._context = context
        self._evaluator = evaluator
        self._batch_size = batch_size
        self._frame = DailyFactorFrame(context.daily)

    def __enter__(self) -> "SequentialExecutionSession":
        """进入串行会话并返回自身。"""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开串行会话并清空表达式节点缓存。

        参数：
            exc_type: 会话内异常的类型；正常退出时为空。
            exc_value: 会话内异常对象；正常退出时为空。
            traceback: 会话内异常的回溯；正常退出时为空。
        """

        self._frame.clear_cache()

    def run(
        self, candidates: Sequence[FactorCandidate]
    ) -> list[CandidateTaskResult]:
        """按批次评价候选并保持输入顺序。

        参数：
            candidates: 当前要评价且已经确定顺序的候选因子。
        """

        return self.run_with_progress(candidates, None)

    def run_with_progress(
        self,
        candidates: Sequence[FactorCandidate],
        progress_callback: BatchProgressCallback | None,
    ) -> list[CandidateTaskResult]:
        """按批次评价候选，并在每批结束后报告累计进度。

        参数：
            candidates: 当前要评价且已经确定顺序的候选因子。
            progress_callback: 可选批次回调，接收完成数、总数和失败数。
        """

        results: list[CandidateTaskResult] = []
        failed = 0
        for start in range(0, len(candidates), self._batch_size):
            batch_results = _evaluate_batch(
                candidates[start : start + self._batch_size],
                self._frame,
                self._context,
                self._evaluator,
            )
            results.extend(batch_results)
            failed += sum(result.error is not None for result in batch_results)
            if progress_callback is not None:
                progress_callback(len(results), len(candidates), failed)
        return results


_WORKER_FRAME: DailyFactorFrame | None = None
_WORKER_CONTEXT: SearchContext | None = None
_WORKER_EVALUATOR: CandidateEvaluator | None = None


def _initialize_worker(context: SearchContext, evaluator: CandidateEvaluator) -> None:
    """每个进程只反序列化一次日频上下文，而不是为每个候选重复传输。

    参数：
        context: 当前 worker 后续批次共享的搜索上下文。
        evaluator: 当前 worker 后续批次共享的评价器。
    """

    global _WORKER_FRAME, _WORKER_CONTEXT, _WORKER_EVALUATOR
    _WORKER_CONTEXT = context
    _WORKER_EVALUATOR = evaluator
    _WORKER_FRAME = DailyFactorFrame(context.daily)


def _worker_batch(candidates: Sequence[FactorCandidate]) -> list[CandidateTaskResult]:
    """使用进程初始化器保存的只读上下文评价一个候选批次。

    参数：
        candidates: 分配给当前 worker 的候选因子批次。
    """

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
        """根据用户配置、CPU 数和批次数计算实际 worker 数量。

        参数：
            candidate_count: 待分配的候选批次数，同时是 worker 数上限。
        """

        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
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
        """把候选批次分派给进程池，并恢复为与输入候选一致的顺序。

        参数：
            candidates: 按期望输出顺序排列的候选因子。
            context: 传入每个 worker 一次的搜索数据上下文。
            evaluator: 传入每个 worker 一次的候选评价器。
            batch_size: 每个进程任务包含的最大候选数。
        """

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

    def open_session(
        self,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> "ProcessExecutionSession":
        """创建只初始化一次 worker 上下文的持久化多进程会话。

        参数：
            context: 每个 worker 初始化时接收一次的只读搜索上下文。
            evaluator: 每个 worker 初始化时接收一次的候选评价器。
            batch_size: 每个进程任务包含的最大候选数量。
        """

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size 必须是正整数")
        workers = self._workers(max(1, self.n_jobs if self.n_jobs > 0 else (os.cpu_count() or 1)))
        return ProcessExecutionSession(
            context=context,
            evaluator=evaluator,
            batch_size=batch_size,
            workers=workers,
        )


class ProcessExecutionSession:
    """在多次评价调用之间保持 worker 存活的进程执行会话。"""

    def __init__(
        self,
        *,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
        workers: int,
    ) -> None:
        """初始化持久化进程池及其只读 worker 上下文。

        参数：
            context: 每个 worker 只反序列化一次的搜索上下文。
            evaluator: 每个 worker 只反序列化一次的候选评价器。
            batch_size: 每个进程任务包含的最大候选数量。
            workers: 进程池允许同时运行的最大 worker 数量。
        """

        self._batch_size = batch_size
        self._workers = workers
        self._context = context
        self._evaluator = evaluator
        self._executor: ProcessPoolExecutor | None = None

    def __enter__(self) -> "ProcessExecutionSession":
        """进入多进程会话并返回自身。"""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开会话并等待已提交任务结束后关闭进程池。

        参数：
            exc_type: 会话内异常的类型；正常退出时为空。
            exc_value: 会话内异常对象；正常退出时为空。
            traceback: 会话内异常的回溯；正常退出时为空。
        """

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def run(
        self, candidates: Sequence[FactorCandidate]
    ) -> list[CandidateTaskResult]:
        """并行评价候选批次并恢复为调用方给定的顺序。

        参数：
            candidates: 当前要评价且已经确定顺序的候选因子。
        """

        return self.run_with_progress(candidates, None)

    def run_with_progress(
        self,
        candidates: Sequence[FactorCandidate],
        progress_callback: BatchProgressCallback | None,
    ) -> list[CandidateTaskResult]:
        """并行评价候选，并在主进程收到每个批次后报告累计进度。

        参数：
            candidates: 当前要评价且已经确定顺序的候选因子。
            progress_callback: 可选批次回调，接收完成数、总数和失败数。
        """

        if not candidates:
            return []
        batches = [
            list(candidates[start : start + self._batch_size])
            for start in range(0, len(candidates), self._batch_size)
        ]
        if self._executor is None:
            # max_workers 保留调用方声明的并行能力。ProcessPoolExecutor 会按提交
            # 任务逐步启动 worker，因此首批较小也不会把后续代次永久限制为串行。
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers,
                initializer=_initialize_worker,
                initargs=(self._context, self._evaluator),
            )
        futures = [self._executor.submit(_worker_batch, batch) for batch in batches]
        by_id: dict[str, CandidateTaskResult] = {}
        completed = 0
        failed = 0
        for future in as_completed(futures):
            batch_results = future.result()
            for result in batch_results:
                by_id[result.factor_id] = result
            completed += len(batch_results)
            failed += sum(result.error is not None for result in batch_results)
            if progress_callback is not None:
                progress_callback(completed, len(candidates), failed)
        return [by_id[candidate.factor_id] for candidate in candidates]
