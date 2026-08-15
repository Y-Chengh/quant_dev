"""因子搜索结果及最优表达式物化辅助接口。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.factor_research.factor_dsl import DailyFactorFrame

from .context import SearchContext
from .space import FactorCandidate


@dataclass(frozen=True)
class FactorSearchResult:
    """保存排行榜、失败候选及可复现的表达式对象。"""

    candidates: tuple[FactorCandidate, ...]
    leaderboard: pd.DataFrame
    errors: pd.DataFrame
    objective: str

    @property
    def best_candidate(self) -> FactorCandidate:
        """返回排行榜中第一个满足覆盖率和目标有限性要求的候选。"""

        if self.leaderboard.empty or "eligible" not in self.leaderboard:
            raise ValueError("搜索没有成功产生候选排行榜")
        eligible = self.leaderboard.loc[self.leaderboard["eligible"]]
        if eligible.empty:
            raise ValueError("没有满足覆盖率和目标值要求的候选")
        return self.get_candidate(str(eligible.iloc[0]["factor_id"]))

    def get_candidate(self, factor_id: str) -> FactorCandidate:
        """按稳定因子 ID 查找原始候选表达式，未知 ID 时明确报错。

        参数：
            factor_id: 要查找的稳定候选因子 ID。
        """

        for candidate in self.candidates:
            if candidate.factor_id == factor_id:
                return candidate
        raise ValueError(f"搜索结果中不存在候选 {factor_id!r}")

    def materialize(
        self,
        context: SearchContext,
        factor_id: str | None = None,
        *,
        oriented: bool = False,
    ) -> pd.DataFrame:
        """把选定表达式计算为普通日频列，供现有数据集和实验流程直接使用。

        参数：
            context: 提供物化所需日频数据的搜索上下文。
            factor_id: 要物化的候选 ID；为空时使用最优合格候选。
            oriented: 是否乘以 selection Rank IC 确定的方向，缺省为 ``False``。
        """

        candidate = self.best_candidate if factor_id is None else self.get_candidate(factor_id)
        if candidate.factor_id in context.daily.columns:
            raise ValueError(f"候选列 {candidate.factor_id!r} 已存在，拒绝静默覆盖")
        values = DailyFactorFrame(context.daily).evaluate(
            candidate.expression, name=candidate.factor_id
        )
        if oriented:
            row = self.leaderboard.loc[
                self.leaderboard["factor_id"].eq(candidate.factor_id)
            ].iloc[0]
            values = values * float(row.get("direction", 1.0))
        daily = context.daily.copy()
        daily[candidate.factor_id] = values.to_numpy(dtype=float)
        return daily
