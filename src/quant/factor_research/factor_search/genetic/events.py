"""遗传搜索的结果、进度事件与候选出身记录。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ..result import FactorSearchResult


@dataclass(frozen=True)
class GeneticSearchResult(FactorSearchResult):
    """在通用因子搜索结果之外保存逐代进化摘要。"""

    history: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class GeneticProgressEvent:
    """描述一个候选批次完成后的遗传搜索进度快照。"""

    stage: str
    generation: int | None
    max_generations: int
    completed: int
    total: int
    successful: int
    failed: int
    selection_evaluations: int
    max_evaluations: int
    elapsed_seconds: float
    eta_seconds: float


GeneticProgressCallback = Callable[[GeneticProgressEvent], None]
"""遗传搜索进度回调，在主进程完成一个候选批次后触发。"""


@dataclass(frozen=True)
class _Provenance:
    """记录候选首次出现时的代次、生成方式和父代。"""

    generation: int
    operation: str
    parent_ids: tuple[str, ...] = ()
