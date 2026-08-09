from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TopNBacktestResult:
    """保存 Top N 日内等权策略的参数、日收益和汇总指标。"""

    top_n: int
    score_column: str
    slippage_bps: float
    commission_bps: float
    daily_returns: pd.DataFrame
    metrics: dict[str, float]


def run_top_n_intraday_backtest(
    predictions: pd.DataFrame,
    score_column: str,
    top_n: int = 10,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
) -> TopNBacktestResult:
    """回测每日买入预测分数最高证券并在收盘卖出的等权策略。

    每个目标交易日在全部有效预测中按分数降序、证券代码升序稳定选取最多
    ``top_n`` 只。买入成交价相对开盘价上浮滑点，卖出成交价相对收盘价下调
    滑点，手续费在买卖两边各收取一次。组合不跨日持仓，收益曲线从 1.0 开始
    按每日净收益复利。夏普比率采用零无风险利率、样本标准差和 252 日年化。

    参数：
        predictions: 验证集逐证券预测，须含目标日期、证券代码、开盘至收盘
            实际收益及指定模型分数。
        score_column: 用于每日横截面选股的模型分数字段；分类通常为
            ``up_probability``，回归通常为 ``predicted_return``。
        top_n: 每日最多买入的证券数量；当日有效证券不足时全部买入。
        slippage_bps: 单边滑点，单位为基点；买卖两边分别应用一次。
        commission_bps: 单边手续费率，单位为基点；买卖两边分别收取一次。

    返回：
        回测参数、按日组合收益与年化夏普等汇总指标。
    """

    if top_n < 1:
        raise ValueError("top_n 必须是正整数")
    if not 0.0 <= slippage_bps < 10_000.0:
        raise ValueError("slippage_bps 必须在 [0, 10000) 范围内")
    if not 0.0 <= commission_bps < 10_000.0:
        raise ValueError("commission_bps 必须在 [0, 10000) 范围内")
    required = {"target_date", "code", "target_return", score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Top N 回测缺少列: {sorted(missing)}")

    frame = predictions.loc[
        :, ["target_date", "code", "target_return", score_column]
    ].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame["target_date"].isna().any():
        raise ValueError("Top N 回测的 target_date 包含缺失或无效日期")
    score = pd.to_numeric(frame[score_column], errors="coerce")
    valid_score = np.isfinite(score.to_numpy(dtype=float))
    frame = frame.loc[valid_score].copy()
    if frame.empty:
        raise ValueError("Top N 回测没有模型分数为有限值的样本")
    frame[score_column] = score.loc[valid_score].to_numpy(dtype=float)
    frame["code"] = frame["code"].astype(str)
    frame = frame.sort_values(
        ["target_date", score_column, "code"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    selected = (
        frame.groupby("target_date", sort=True, group_keys=False).head(top_n).copy()
    )
    selected["target_return"] = pd.to_numeric(
        selected["target_return"], errors="coerce"
    )
    finite_return = np.isfinite(selected["target_return"].to_numpy(dtype=float))
    if not finite_return.all():
        invalid = selected.loc[~finite_return, ["target_date", "code"]]
        details = ", ".join(
            f"{pd.Timestamp(row.target_date).date()}:{row.code}"
            for row in invalid.itertuples(index=False)
        )
        raise ValueError(f"Top N 已选证券的实际收益不是有限值: {details}")

    slippage = slippage_bps / 10_000.0
    commission = commission_bps / 10_000.0
    execution_multiplier = (
        (1.0 - slippage) * (1.0 - commission)
        / ((1.0 + slippage) * (1.0 + commission))
    )
    selected["net_return"] = (
        (1.0 + selected["target_return"]) * execution_multiplier - 1.0
    )
    daily = (
        selected.groupby("target_date", as_index=False, sort=True)
        .agg(
            selected_count=("code", "size"),
            gross_return=("target_return", "mean"),
            net_return=("net_return", "mean"),
        )
        .sort_values("target_date", kind="mergesort")
        .reset_index(drop=True)
    )
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()

    returns = daily["net_return"].to_numpy(dtype=float)
    periods = len(returns)
    total_return = float(daily["equity"].iloc[-1] - 1.0)
    final_equity = float(daily["equity"].iloc[-1])
    annualized_return = (
        float(final_equity ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)
        if final_equity > 0.0
        else float("nan")
    )
    daily_std = float(np.std(returns, ddof=1)) if periods >= 2 else float("nan")
    sharpe_ratio = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(returns) / daily_std)
        if np.isfinite(daily_std) and daily_std > 0.0
        else float("nan")
    )
    metrics = {
        "trading_days": float(periods),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": (
            float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(daily_std)
            else float("nan")
        ),
        "sharpe_ratio": sharpe_ratio,
    }
    return TopNBacktestResult(
        top_n=top_n,
        score_column=score_column,
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        daily_returns=daily,
        metrics=metrics,
    )
