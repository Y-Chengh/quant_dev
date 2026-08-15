"""候选评价协议和不依赖具体模型的横截面 IC 评价。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from quant.factor_research.metrics import (
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
    ) -> dict[str, float | str]:
        """返回可直接合并进排行榜的数值指标和可选稳定指纹。

        参数：
            candidate: 当前指标所属的候选因子及其表达式。
            values: 与上下文日频表逐行对齐的候选值。
            context: 目标值、日期掩码和行位置映射。
        """


def _fingerprint_float_array(values: np.ndarray) -> str:
    """为一维浮点数组生成跨进程稳定的等价指纹。

    参数：
        values: 已按固定样本顺序排列的候选值或横截面秩；非有限值统一按缺失
            处理，正负零统一为正零。

    返回：
        包含数组长度与规范化浮点字节的 128 位 BLAKE2b 十六进制摘要。
    """

    canonical = np.asarray(values, dtype="<f8").copy()
    if canonical.ndim != 1:
        raise ValueError("等价指纹只接受一维候选数组")
    canonical[~np.isfinite(canonical)] = np.nan
    canonical[canonical == 0.0] = 0.0
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray([len(canonical)], dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _selection_equivalence_fingerprints(
    context: SearchContext,
    aligned_scores: np.ndarray,
    direction: float,
) -> dict[str, str]:
    """计算 selection 方向值和逐日横截面秩的稳定等价指纹。

    参数：
        context: 提供 selection 掩码、目标日期及目标收益有效性的搜索上下文。
        aligned_scores: 已对齐到目标样本顺序的原始候选值。
        direction: 仅依据 selection Rank IC 锁定的候选方向，只允许正负一。

    返回：
        方向调整后有效值指纹及按目标交易日独立计算的平均秩指纹；缺失位置也
        参与摘要，因此覆盖范围不同的候选不会被误判为等价。
    """

    if direction not in (-1.0, 1.0):
        raise ValueError("selection 等价指纹的方向必须是 -1 或 1")
    selection = np.asarray(context.selection_mask, dtype=bool)
    oriented = np.asarray(aligned_scores[selection] * direction, dtype=float)
    target_returns = context.targets.loc[
        selection, "target_return"
    ].to_numpy(dtype=float)
    comparable = oriented.copy()
    comparable[~np.isfinite(target_returns)] = np.nan
    comparable[~np.isfinite(comparable)] = np.nan

    target_dates = context.targets.loc[
        selection, "target_date"
    ].reset_index(drop=True)
    daily_ranks = (
        pd.Series(comparable)
        .groupby(target_dates, sort=False)
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    return {
        "selection_oriented_value_fingerprint": _fingerprint_float_array(
            comparable
        ),
        "selection_oriented_rank_fingerprint": _fingerprint_float_array(
            daily_ranks
        ),
    }


def _finite_std(values: pd.Series) -> float:
    """计算有限值的样本标准差，不足两个观测时返回缺失值。

    参数：
        values: 要过滤非有限值并计算样本标准差的序列。
    """

    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")


def _period_ic_metrics(
    context: SearchContext,
    aligned_scores: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    min_daily_samples: int,
) -> dict[str, float]:
    """在指定日期掩码上汇总逐日横截面 IC、覆盖率和稳定性指标。

    参数：
        context: 提供目标收益和目标日期的搜索上下文。
        aligned_scores: 已对齐到目标样本顺序的候选得分。
        mask: 指定本次统计使用哪些目标样本的布尔掩码。
        prefix: 写入指标字典时区分时间区间的键前缀。
        min_daily_samples: 单日横截面 IC 有效所需的最小样本数。
    """

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
    """只使用 selection 计算横截面 IC，并可附加等价去重指纹。"""

    min_daily_samples: int = 2
    include_equivalence_fingerprints: bool = False

    def __post_init__(self) -> None:
        """校验每日计算横截面相关系数所需的最小样本数。"""

        if self.min_daily_samples < 2:
            raise ValueError("min_daily_samples 必须大于等于 2")
        if not isinstance(self.include_equivalence_fingerprints, bool):
            raise TypeError("include_equivalence_fingerprints 必须是布尔值")

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float | str]:
        """评价 selection 指标，并只用 selection Rank IC 锁定因子方向。

        参数：
            candidate: 当前评价的候选因子。
            values: 与日频上下文等长的未对齐候选值。
            context: 包含 selection 掩码和目标的搜索上下文。
        """

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
        if self.include_equivalence_fingerprints:
            metrics.update(
                _selection_equivalence_fingerprints(
                    context, aligned, direction
                )
            )
        return metrics


@dataclass(frozen=True)
class HoldoutIcEvaluator:
    """只计算最终入选候选的 holdout 指标，不参与候选排序。"""

    min_daily_samples: int = 2

    def __post_init__(self) -> None:
        """校验 holdout 每日横截面计算的最小样本数。"""

        if self.min_daily_samples < 2:
            raise ValueError("min_daily_samples 必须大于等于 2")

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """仅在预先划定的 holdout 区间计算候选 IC 汇总指标。

        参数：
            candidate: 最终入选并要进行样本外评价的候选因子。
            values: 与日频上下文等长的未对齐候选值。
            context: 包含 holdout 掩码和目标的搜索上下文。
        """

        aligned = context.align_factor_values(values)
        return _period_ic_metrics(
            context,
            aligned,
            context.holdout_mask,
            "holdout",
            self.min_daily_samples,
        )
