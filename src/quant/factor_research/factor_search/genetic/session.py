"""把执行后端包装为可上报阶段进度与 ETA 的搜索会话。"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from types import TracebackType

from ..backends import (
    BatchProgressCallback,
    CandidateTaskResult,
    ExecutionBackend,
    ExecutionSession,
)
from ..context import SearchContext
from ..evaluators import CandidateEvaluator
from ..space import FactorCandidate
from .config import GeneticSearchConfig
from .events import GeneticProgressCallback, GeneticProgressEvent


class _BackendSessionAdapter:
    """把只实现一次性 run 的旧执行后端适配为遗传搜索会话。"""

    def __init__(
        self,
        backend: ExecutionBackend,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> None:
        """保存每代调用旧执行后端所需的固定参数。

        参数：
            backend: 仅提供批量 ``run`` 接口的自定义执行后端。
            context: 全部代次共享的只读搜索上下文。
            evaluator: 把候选值转换为 selection 指标的评价器。
            batch_size: 每次调用后端时使用的候选批次大小。
        """

        self._backend = backend
        self._context = context
        self._evaluator = evaluator
        self._batch_size = batch_size

    def __enter__(self) -> _BackendSessionAdapter:
        """进入无额外资源的兼容会话并返回自身。"""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出兼容会话；旧后端没有需要持续持有的资源。

        参数：
            exc_type: 会话内异常的类型；正常退出时为空。
            exc_value: 会话内异常对象；正常退出时为空。
            traceback: 会话内异常的回溯；正常退出时为空。
        """

    def run(
        self, candidates: Sequence[FactorCandidate]
    ) -> list[CandidateTaskResult]:
        """通过旧后端的一次性接口评价当前代新增候选。

        参数：
            candidates: 当前代尚未出现在跨代缓存中的候选。
        """

        return self._backend.run(
            candidates,
            self._context,
            self._evaluator,
            self._batch_size,
        )

    def run_with_progress(
        self,
        candidates: Sequence[FactorCandidate],
        progress_callback: BatchProgressCallback | None,
    ) -> list[CandidateTaskResult]:
        """通过旧后端评价候选，并在整批完成后至少报告一次进度。

        参数：
            candidates: 当前代尚未出现在跨代缓存中的候选。
            progress_callback: 可选批次回调；旧后端只能在全部完成后调用一次。
        """

        results = self.run(candidates)
        if progress_callback is not None and candidates:
            progress_callback(
                len(results),
                len(candidates),
                sum(result.error is not None for result in results),
            )
        return results


class _ProgressReporter:
    """把执行会话的批次计数转换为带阶段和 ETA 的公开进度事件。"""

    def __init__(
        self,
        callback: GeneticProgressCallback,
        *,
        stage: str,
        generation: int | None,
        config: GeneticSearchConfig,
        selection_base: int,
    ) -> None:
        """保存当前评价阶段生成进度事件所需的固定上下文。

        参数：
            callback: 接收公开进度事件的调用方回调。
            stage: 当前阶段名称，只使用 selection、holdout 或 model。
            generation: 当前 selection 的一基代次；后置阶段为空。
            config: 提供总代数与 selection 候选预算的遗传配置。
            selection_base: 本批开始前已经完成的 selection 候选数量。
        """

        self._callback = callback
        self._stage = stage
        self._generation = generation
        self._config = config
        self._selection_base = selection_base
        self._started = perf_counter()

    def __call__(self, completed: int, total: int, failed: int) -> None:
        """根据当前批次累计数计算速率和阶段剩余时间并触发回调。

        参数：
            completed: 当前阶段累计完成的候选数量。
            total: 当前阶段计划评价的候选总数。
            failed: 当前阶段累计失败的候选数量。
        """

        elapsed = perf_counter() - self._started
        eta = (
            elapsed * max(0, total - completed) / completed
            if completed > 0
            else float("nan")
        )
        selection_evaluations = self._selection_base
        if self._stage == "selection":
            selection_evaluations += completed
        self._callback(
            GeneticProgressEvent(
                stage=self._stage,
                generation=self._generation,
                max_generations=self._config.max_generations,
                completed=completed,
                total=total,
                successful=completed - failed,
                failed=failed,
                selection_evaluations=selection_evaluations,
                max_evaluations=self._config.max_evaluations,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )
        )


def _run_session_with_progress(
    session: ExecutionSession,
    candidates: Sequence[FactorCandidate],
    progress_callback: BatchProgressCallback | None,
) -> list[CandidateTaskResult]:
    """优先使用细粒度会话接口，并兼容只实现旧 ``run`` 的自定义会话。

    参数：
        session: 已进入且负责实际候选评价的执行会话。
        candidates: 当前阶段按稳定顺序排列的候选。
        progress_callback: 可选批次回调；旧会话在全部完成后补报一次。

    返回：
        与候选输入顺序一致的评价结果。
    """

    run_with_progress = getattr(session, "run_with_progress", None)
    if callable(run_with_progress):
        return run_with_progress(candidates, progress_callback)
    results = session.run(candidates)
    if progress_callback is not None and candidates:
        progress_callback(
            len(results),
            len(candidates),
            sum(result.error is not None for result in results),
        )
    return results
