"""因子搜索的一次性数据准备结果。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.factor_research.dataset import build_forward_targets


@dataclass(frozen=True)
class SearchContext:
    """可被所有候选只读复用的日频数据、目标和日期切分。

    ``target_positions`` 把按目标日期排序的目标行映射回 ``daily`` 的原始行位置，
    因而每个候选只需一次 NumPy 索引即可与标签严格对齐。
    """

    daily: pd.DataFrame
    targets: pd.DataFrame
    target_positions: np.ndarray
    fixed_features: tuple[str, ...]
    selection_mask: np.ndarray
    holdout_mask: np.ndarray
    holdout_start: pd.Timestamp | None
    holdout_end: pd.Timestamp | None

    @classmethod
    def from_daily(
        cls,
        daily: pd.DataFrame,
        *,
        fixed_features: Sequence[str] = (),
        selection_start: str | pd.Timestamp | None = None,
        holdout_start: str | pd.Timestamp | None = None,
        holdout_end: str | pd.Timestamp | None = None,
    ) -> SearchContext:
        """校验日频表的数据契约，一次性构建目标、行映射和日期区间掩码。

        参数：
            daily: 键列已规范化且每个证券交易日唯一的日频表。
            fixed_features: 日频表中要与搜索候选联合建模的固定因子列；缺省为空。
            selection_start: 筛选区间首个目标日期；为空时从最早目标开始。
            holdout_start: 样本外区间首个目标日期，也是筛选区间的排他上界；为空时不划分 holdout。
            holdout_end: 样本外区间最后一个目标日期，包含该日；为空时不限制结束日。
        """

        required = {"code", "trade_date", "open", "close"}
        missing = required.difference(daily.columns)
        if missing:
            raise ValueError(f"搜索日频数据缺少列: {sorted(missing)}")
        fixed = tuple(fixed_features)
        if len(fixed) != len(set(fixed)):
            raise ValueError("固定因子列表不能包含重复项")
        missing_features = set(fixed).difference(daily.columns)
        if missing_features:
            raise ValueError(f"搜索日频数据缺少固定因子: {sorted(missing_features)}")

        if not daily["code"].map(lambda value: isinstance(value, str)).all():
            raise TypeError("搜索日频数据 code 列必须全部为字符串")
        if not pd.api.types.is_datetime64_any_dtype(daily["trade_date"].dtype):
            raise TypeError("搜索日频数据 trade_date 列必须为 datetime64 类型")
        if not daily["trade_date"].eq(daily["trade_date"].dt.normalize()).all():
            raise ValueError("搜索日频数据 trade_date 必须为归零后的交易日")

        prepared = daily.copy()
        # (code, trade_date) 是后续 MultiIndex 映射的唯一业务主键。
        if prepared.duplicated(["code", "trade_date"]).any():
            raise ValueError("搜索日频数据存在重复的 (code, trade_date)")
        targets = build_forward_targets(prepared)

        daily_keys = pd.MultiIndex.from_frame(prepared[["code", "trade_date"]])
        target_keys = pd.MultiIndex.from_frame(targets[["code", "feature_date"]])
        target_positions = daily_keys.get_indexer(target_keys)
        if (target_positions < 0).any():
            raise RuntimeError("目标行无法映射回日频特征行")

        target_dates = targets["target_date"]
        selection = np.ones(len(targets), dtype=bool)
        if selection_start is not None:
            selection &= (target_dates >= pd.Timestamp(selection_start)).to_numpy(
                dtype=bool
            )
        cutoff = pd.Timestamp(holdout_start) if holdout_start is not None else None
        if cutoff is not None:
            selection &= (target_dates < cutoff).to_numpy(dtype=bool)
            # pandas 可能返回只读 NumPy 视图；后续还要叠加 holdout_end，因此必须
            # 显式复制为当前上下文独占的可写布尔数组。
            holdout = (target_dates >= cutoff).to_numpy(
                dtype=bool, copy=True
            )
        else:
            holdout = np.zeros(len(targets), dtype=bool)
        end = pd.Timestamp(holdout_end) if holdout_end is not None else None
        if end is not None:
            if cutoff is None:
                raise ValueError("配置 holdout_end 时必须同时配置 holdout_start")
            if end < cutoff:
                raise ValueError("holdout_end 不能早于 holdout_start")
            holdout &= (target_dates <= end).to_numpy(dtype=bool)
        selection = np.asarray(selection, dtype=bool)
        if not selection.any():
            raise ValueError("selection 区间没有可评价样本")
        if cutoff is not None and not holdout.any():
            raise ValueError("holdout 区间没有可评价样本")

        return cls(
            daily=prepared,
            targets=targets,
            target_positions=target_positions,
            fixed_features=fixed,
            selection_mask=selection,
            holdout_mask=holdout,
            holdout_start=cutoff,
            holdout_end=end,
        )

    def align_factor_values(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """把与 daily 等长的候选值对齐到目标样本顺序。

        参数：
            values: 与 ``daily`` 每行一一对应的候选因子值。
        """

        array = np.asarray(values, dtype=float)
        if array.shape != (len(self.daily),):
            raise ValueError(
                f"候选因子行数错误: expected={len(self.daily)} actual={array.shape}"
            )
        return array[self.target_positions]
