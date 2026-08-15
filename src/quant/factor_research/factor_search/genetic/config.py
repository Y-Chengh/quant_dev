"""遗传搜索的默认算子参数与搜索配置校验。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from quant.factor_research.factor_dsl import get_operator


def _default_operator_parameters() -> dict[str, dict[str, tuple[object, ...]]]:
    """返回第一版遗传搜索使用的因果算子和离散参数白名单。"""

    windows: tuple[object, ...] = (5, 10, 20, 60)
    periods: tuple[object, ...] = (1, 2, 5, 10, 20)
    return {
        "add": {},
        "subtract": {},
        "multiply": {},
        "divide": {},
        "absolute": {},
        "delta": {"periods": periods},
        "returns": {"periods": periods},
        "ts_mean": {"window": windows},
        "ts_stddev": {"window": windows},
        "ts_rank": {"window": windows},
        "ts_correlation": {"window": windows},
        "cs_rank": {},
        "cs_zscore": {},
    }


@dataclass(frozen=True)
class GeneticSearchConfig:
    """声明遗传表达式空间、种群规模、复杂度限制和适应度惩罚。"""

    sources: tuple[str, ...] = ("close", "volume", "return_1d")
    operator_parameters: Mapping[str, Mapping[str, Sequence[object]]] = field(
        default_factory=_default_operator_parameters
    )
    commutative_operators: frozenset[str] = frozenset(
        {"add", "multiply", "ts_correlation"}
    )
    disallow_same_input_operators: frozenset[str] = frozenset(
        {"ts_correlation"}
    )
    population_size: int = 300
    max_generations: int = 20
    max_evaluations: int = 5_000
    initial_max_depth: int = 3
    max_depth: int = 5
    max_nodes: int = 15
    max_lookback: int = 120
    min_coverage: float = 0.70
    target_coverage: float = 0.90
    objective: str = "selection_oriented_rank_ic"
    stability_weight: float = 0.5
    free_node_count: int = 3
    length_penalty: float = 0.0005
    free_depth: int = 2
    depth_penalty: float = 0.0005
    coverage_penalty: float = 0.02
    elite_ratio: float = 0.10
    tournament_size: int = 5
    crossover_probability: float = 0.35
    subtree_mutation_probability: float = 0.30
    parameter_mutation_probability: float = 0.20
    reproduction_probability: float = 0.15
    terminal_probability: float = 0.25
    patience: int = 5
    min_improvement: float = 0.0001
    random_seed: int = 20260809

    def __post_init__(self) -> None:
        """校验搜索规模、概率、复杂度和算子白名单。"""

        if not self.sources or any(
            not isinstance(source, str) or not source for source in self.sources
        ):
            raise ValueError("sources 必须包含至少一个非空列名")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("sources 不能包含重复列名")
        positive_integers = {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "max_evaluations": self.max_evaluations,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "patience": self.patience,
            "tournament_size": self.tournament_size,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        nonnegative_integers = {
            "initial_max_depth": self.initial_max_depth,
            "max_lookback": self.max_lookback,
            "free_node_count": self.free_node_count,
            "free_depth": self.free_depth,
            "random_seed": self.random_seed,
        }
        for name, value in nonnegative_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if not 0 <= self.initial_max_depth <= self.max_depth:
            raise ValueError("initial_max_depth 必须在 0 到 max_depth 之间")
        for name, value in {
            "min_coverage": self.min_coverage,
            "target_coverage": self.target_coverage,
            "elite_ratio": self.elite_ratio,
            "terminal_probability": self.terminal_probability,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if self.target_coverage < self.min_coverage:
            raise ValueError("target_coverage 不能小于 min_coverage")
        probabilities = (
            self.crossover_probability,
            self.subtree_mutation_probability,
            self.parameter_mutation_probability,
            self.reproduction_probability,
        )
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("交叉、两类变异和复制概率必须非负且合计为 1")
        for name, value in {
            "stability_weight": self.stability_weight,
            "length_penalty": self.length_penalty,
            "depth_penalty": self.depth_penalty,
            "coverage_penalty": self.coverage_penalty,
            "min_improvement": self.min_improvement,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if not self.objective.startswith("selection_"):
            raise ValueError("objective 必须是 selection 指标，禁止使用 holdout 进化")
        if not self.operator_parameters:
            raise ValueError("operator_parameters 不能为空")
        for operator, parameters in self.operator_parameters.items():
            definition = get_operator(operator)
            if not definition.causal:
                raise ValueError(f"遗传搜索拒绝非因果算子 {operator!r}")
            for parameter, values in parameters.items():
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes))
                    or not values
                ):
                    raise ValueError(
                        f"算子 {operator!r} 的参数 {parameter!r} 必须是非空取值序列"
                    )
