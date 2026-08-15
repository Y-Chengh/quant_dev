"""搜索区间摘要与候选逐日 IC 明细的汇总。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..factor_search import FactorSearchResult, SearchContext
from ..metrics import daily_cross_sectional_ic


def _period_summary(context: SearchContext, mask: np.ndarray) -> dict[str, object]:
    """汇总一个目标日期区间的样本数、证券数和日期边界。

    参数：
        context: 已准备好的只读搜索上下文，目标表按证券和目标日期对齐。
        mask: 与 ``context.targets`` 等长的布尔掩码，表示 selection 或 holdout。

    返回：
        可直接写入报告的区间统计字典。
    """

    period = context.targets.loc[mask]
    if period.empty:
        return {"samples": 0, "dates": 0, "codes": 0, "start": None, "end": None}
    return {
        "samples": len(period),
        "dates": period["target_date"].nunique(),
        "codes": period["code"].nunique(),
        "start": period["target_date"].min(),
        "end": period["target_date"].max(),
    }


def _build_top_daily_ic(
    result: FactorSearchResult,
    context: SearchContext,
    factor_ids: Sequence[str],
) -> pd.DataFrame:
    """重建搜索阶段已完成 holdout 评价候选的逐日横截面 IC 明细。

    因子方向沿用 selection 已锁定的 ``direction``；holdout 仅作为报告区间，
    不参与候选排序或方向选择。

    参数：
        result: 已完成 selection 排名和 Top K holdout 评价的搜索结果。
        context: 保存日频数据、目标和两个互斥日期掩码的搜索上下文。
        factor_ids: 搜索阶段已经成功完成 holdout 评价的冻结候选 ID，顺序与
            selection 排名一致；报告不得自行加入其他候选。

    返回：
        按候选排名、区间和目标日期排列的 IC/Rank IC 明细表。
    """

    frozen_ids = set(factor_ids)
    eligible = result.leaderboard.loc[
        result.leaderboard["eligible"]
        & result.leaderboard["factor_id"].isin(frozen_ids)
    ]
    rows: list[pd.DataFrame] = []
    for rank, (_, leaderboard_row) in enumerate(eligible.iterrows(), start=1):
        factor_id = str(leaderboard_row["factor_id"])
        daily = result.materialize(context, factor_id=factor_id, oriented=True)
        aligned = context.align_factor_values(daily[factor_id])
        candidate = result.get_candidate(factor_id)
        for period_name, mask in (
            ("selection", context.selection_mask),
            ("holdout", context.holdout_mask),
        ):
            if not mask.any():
                continue
            predictions = context.targets.loc[
                mask, ["target_date", "target_return"]
            ].copy()
            predictions["score"] = aligned[mask]
            detail = daily_cross_sectional_ic(predictions, "score")
            detail.insert(0, "canonical", candidate.canonical)
            detail.insert(0, "period", period_name)
            detail.insert(0, "factor_id", factor_id)
            detail.insert(0, "selection_rank", rank)
            rows.append(detail)
    columns = [
        "selection_rank",
        "factor_id",
        "period",
        "target_date",
        "samples",
        "ic",
        "rank_ic",
        "canonical",
    ]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)
