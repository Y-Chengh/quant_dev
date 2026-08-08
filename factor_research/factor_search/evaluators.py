"""候选评价协议和不依赖具体模型的横截面 IC 评价。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from factor_research.metrics import (
    cross_sectional_ic_metrics,
    daily_cross_sectional_ic,
)
from .context import SearchContext
from .space import FactorCandidate


class CandidateEvaluator(Protocol):
    """评价策略只读取候选值和上下文，不负责生成或执行表达式。"""

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """返回可直接合并进排行榜的纯数值指标。"""


def _finite_std(values: pd.Series) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")


def _period_ic_metrics(
    context: SearchContext,
    aligned_scores: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    min_daily_samples: int,
) -> dict[str, float]:
    rows = int(mask.sum())
    if rows == 0:
        return {
            f"{prefix}_samples": 0.0,
            f"{prefix}_valid_values": 0.0,
            f"{prefix}_coverage": float("nan"),
            f"{prefix}_ic": float("nan"),
            f"{prefix}_rank_ic": float("nan"),
            f"{prefix}_ic_std": float("nan"),
            f"{prefix}_rank_ic_std": float("nan"),
            f"{prefix}_icir": float("nan"),
            f"{prefix}_rank_icir": float("nan"),
            f"{prefix}_ic_dates": 0.0,
            f"{prefix}_rank_ic_dates": 0.0,
            f"{prefix}_positive_rank_ic_ratio": float("nan"),
            f"{prefix}_mean_cross_section": float("nan"),
        }

    frame = context.targets.loc[
        mask, ["target_date", "target_return"]
    ].copy()
    frame["score"] = aligned_scores[mask]
    valid_pairs = np.isfinite(frame["score"].to_numpy(dtype=float)) & np.isfinite(
        frame["target_return"].to_numpy(dtype=float)
    )
    valid_values = int(valid_pairs.sum())
    daily = daily_cross_sectional_ic(frame, "score")
    too_small = daily["samples"] < min_daily_samples
    daily.loc[too_small, ["ic", "rank_ic"]] = np.nan
    summary = cross_sectional_ic_metrics(daily)
    ic_std = _finite_std(daily["ic"])
    rank_ic_std = _finite_std(daily["rank_ic"])
    valid_rank = daily["rank_ic"].dropna()
    return {
        f"{prefix}_samples": float(rows),
        f"{prefix}_valid_values": float(valid_values),
        f"{prefix}_coverage": float(valid_values / rows),
        f"{prefix}_ic": summary["ic"],
        f"{prefix}_rank_ic": summary["rank_ic"],
        f"{prefix}_ic_std": ic_std,
        f"{prefix}_rank_ic_std": rank_ic_std,
        f"{prefix}_icir": (
            float(summary["ic"] / ic_std)
            if np.isfinite(ic_std) and ic_std > 0
            else float("nan")
        ),
        f"{prefix}_rank_icir": (
            float(summary["rank_ic"] / rank_ic_std)
            if np.isfinite(rank_ic_std) and rank_ic_std > 0
            else float("nan")
        ),
        f"{prefix}_ic_dates": summary["ic_dates"],
        f"{prefix}_rank_ic_dates": summary["rank_ic_dates"],
        f"{prefix}_positive_rank_ic_ratio": (
            float((valid_rank > 0).mean()) if not valid_rank.empty else float("nan")
        ),
        f"{prefix}_mean_cross_section": (
            float(daily["samples"].mean()) if not daily.empty else float("nan")
        ),
    }


@dataclass(frozen=True)
class IcEvaluator:
    """只使用 selection 区间计算候选筛选所需的横截面 IC。"""

    min_daily_samples: int = 2

    def __post_init__(self) -> None:
        if self.min_daily_samples < 2:
            raise ValueError("min_daily_samples 必须大于等于 2")

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        aligned = context.align_factor_values(values)
        metrics = _period_ic_metrics(
            context,
            aligned,
            context.selection_mask,
            "selection",
            self.min_daily_samples,
        )
        selection_rank_ic = metrics["selection_rank_ic"]
        direction = -1.0 if np.isfinite(selection_rank_ic) and selection_rank_ic < 0 else 1.0
        metrics["direction"] = direction
        metrics["selection_oriented_ic"] = metrics["selection_ic"] * direction
        metrics["selection_oriented_rank_ic"] = (
            metrics["selection_rank_ic"] * direction
        )
        return metrics


@dataclass(frozen=True)
class HoldoutIcEvaluator:
    """只计算最终入选候选的 holdout 指标，不参与候选排序。"""

    min_daily_samples: int = 2

    def __post_init__(self) -> None:
        if self.min_daily_samples < 2:
            raise ValueError("min_daily_samples 必须大于等于 2")

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        aligned = context.align_factor_values(values)
        return _period_ic_metrics(
            context,
            aligned,
            context.holdout_mask,
            "holdout",
            self.min_daily_samples,
        )
