"""按历史成交可执行性标记可进入模型拟合的监督学习样本。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .backtesting.execution_filters import apply_execution_filters
from .dataset import (
    TRAINING_SAMPLE_ELIGIBLE_COLUMN,
    TRAINING_SAMPLE_FILTER_REASON_COLUMN,
)


def mark_training_sample_eligibility(
    dataset: pd.DataFrame,
    execution_context: pd.DataFrame | None,
    filter_names: Iterable[str],
    *,
    entry_timing: str,
    exit_timing: str,
    training_cutoff: str | pd.Timestamp | None = None,
    missing_policy: str = "allow",
    options: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按目标持有期的买卖可执行性标记训练样本。

    目标日仍保留在数据集中供验证预测；只有模型拟合历史窗口会读取资格列。买入
    使用 ``target_date``，卖出使用 ``target_end_date``，因此滚动训练只能在收益及
    两端成交状态均已发生后使用该标记，不会引用预测日及之后的信息。

    参数：
        dataset: 含 ``target_date``、``target_end_date`` 和 ``code`` 的监督学习样本。
        execution_context: 以 ``execution_date``、``code`` 唯一定位的原始成交上下文；
            启用任一过滤器时必须提供。
        filter_names: 应用于训练样本买卖两端的成交过滤器名称序列。
        entry_timing: 买入时点，取 ``open`` 或 ``close``。
        exit_timing: 卖出时点，取 ``open`` 或 ``close``。
        training_cutoff: 本次运行可能进入拟合的样本结束日上界，不含该日；``None``
            表示检查全部样本。晚于该边界的末端验证样本不会触发缺失上下文策略。
        missing_policy: 无法判定时取 ``error``、``exclude`` 或 ``allow``；缺省只剔除
            已确认不可成交的样本，保留无法判定行。
        options: 涨跌停比例容差、价格 tick 和 ST 口径等成交过滤器共享参数。

    返回：
        标有训练资格及买卖侧过滤原因的数据集副本，以及买卖两侧逐过滤器统计表。
    """

    required = {"target_date", "target_end_date", "code"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"训练样本成交过滤缺少列: {missing}")
    if dataset[["target_date", "target_end_date"]].isna().any().any():
        raise ValueError("训练样本成交过滤不接受缺失的目标起止日期")
    filter_names = tuple(filter_names)

    if training_cutoff is None:
        candidate_mask = np.ones(len(dataset), dtype=bool)
    else:
        cutoff = pd.Timestamp(training_cutoff)
        if pd.isna(cutoff):
            raise ValueError("training_cutoff 必须是有效日期")
        candidate_mask = (
            pd.to_datetime(dataset["target_end_date"]).lt(cutoff).to_numpy(dtype=bool)
        )
    sample_ids = np.flatnonzero(candidate_mask).astype(np.int64, copy=False)
    marked = dataset.copy()
    marked[TRAINING_SAMPLE_ELIGIBLE_COLUMN] = True
    marked[TRAINING_SAMPLE_FILTER_REASON_COLUMN] = pd.Series(
        "", index=marked.index, dtype="string"
    )
    if not len(sample_ids):
        return marked, pd.DataFrame(
            columns=[
                "filter",
                "side",
                "candidate_count",
                "blocked_count",
                "unknown_count",
            ]
        )

    audits: dict[str, pd.DataFrame] = {}
    statistics: list[pd.DataFrame] = []
    for side, date_column, timing in (
        ("buy", "target_date", entry_timing),
        ("sell", "target_end_date", exit_timing),
    ):
        candidates = pd.DataFrame(
            {
                "_training_sample_id": sample_ids,
                "execution_date": pd.to_datetime(
                    dataset.iloc[sample_ids][date_column]
                ).to_numpy(),
                "code": dataset.iloc[sample_ids]["code"].astype(str).to_numpy(),
            }
        )
        _, stats, audit = apply_execution_filters(
            candidates,
            execution_context,
            filter_names,
            side=side,
            timing=timing,
            missing_policy=missing_policy,
            options=options,
            return_audit=True,
        )
        audits[side] = audit.set_index("_training_sample_id").reindex(sample_ids)
        if not stats.empty:
            statistics.append(stats)

    buy_eligible = audits["buy"]["execution_filter_eligible"].to_numpy(dtype=bool)
    sell_eligible = audits["sell"]["execution_filter_eligible"].to_numpy(dtype=bool)
    eligible = buy_eligible & sell_eligible
    reasons = pd.Series("", index=pd.RangeIndex(len(dataset)), dtype="string")
    for side in ("buy", "sell"):
        audit = audits[side]
        blocked_rows = audit.loc[~audit["execution_filter_eligible"].astype(bool)]
        for sample_id, row in blocked_rows.iterrows():
            side_reason = f"{side}:{row['execution_filter_reason']}"
            current = reasons.iloc[int(sample_id)]
            reasons.iloc[int(sample_id)] = (
                f"{current};{side_reason}" if current else side_reason
            )

    marked.iloc[
        sample_ids, marked.columns.get_loc(TRAINING_SAMPLE_ELIGIBLE_COLUMN)
    ] = eligible
    marked[TRAINING_SAMPLE_FILTER_REASON_COLUMN] = reasons.to_numpy()
    stats = (
        pd.concat(statistics, ignore_index=True)
        if statistics
        else pd.DataFrame(
            columns=[
                "filter",
                "side",
                "candidate_count",
                "blocked_count",
                "unknown_count",
            ]
        )
    )
    return marked, stats
