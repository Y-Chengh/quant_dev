"""高内聚、可并行且不污染现有因子注册表的网格与遗传搜索工具。"""

from .backends import ProcessBackend, SequentialBackend
from .context import SearchContext
from .evaluators import HoldoutIcEvaluator, IcEvaluator
from .genetic import FactorGeneticSearch, GeneticSearchConfig, GeneticSearchResult
from .integration import prepare_search_context
from .model_evaluator import ModelCandidateEvaluator
from .result import FactorSearchResult
from .runner import FactorGridSearch
from .space import (
    CombinedGrid,
    ExpressionGrid,
    FactorCandidate,
    OperatorGrid,
    PipelineGrid,
    identity,
    op,
)

__all__ = [
    "CombinedGrid",
    "ExpressionGrid",
    "FactorCandidate",
    "FactorGridSearch",
    "FactorGeneticSearch",
    "FactorSearchResult",
    "GeneticSearchConfig",
    "GeneticSearchResult",
    "IcEvaluator",
    "HoldoutIcEvaluator",
    "ModelCandidateEvaluator",
    "OperatorGrid",
    "PipelineGrid",
    "ProcessBackend",
    "SearchContext",
    "SequentialBackend",
    "identity",
    "op",
    "prepare_search_context",
]
