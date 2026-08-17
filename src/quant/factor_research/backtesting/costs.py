"""买卖两端成交乘数与不同滑点下的净值敏感性对比。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .drawdown import _drawdown_curve
from .metrics import _compound_metrics


def _execution_multiplier(slippage_bps: float, commission_bps: float) -> float:
    """把单边滑点与单边手续费换算为买卖两端的成交价乘数。

    买入价相对开盘价上浮滑点并付一次手续费，卖出价相对收盘价下调滑点并再付
    一次手续费，因此持有一日的净收益为 ``(1 + 毛收益) * 乘数 - 1``。

    参数：
        slippage_bps: 单边滑点，单位为基点。
        commission_bps: 单边手续费率，单位为基点。

    返回：
        买卖两端成本合并后的成交价乘数，取值不大于 1。
    """

    slippage = slippage_bps / 10_000.0
    commission = commission_bps / 10_000.0
    return (
        (1.0 - slippage) * (1.0 - commission)
        / ((1.0 + slippage) * (1.0 + commission))
    )


def _normalize_slippage_candidates(
    slippage_bps_candidates: Sequence[float],
    slippage_bps: float,
) -> list[float]:
    """校验滑点候选取值并按出现顺序去重。

    去重使用精确相等比较，因此 ``2.5`` 与 ``2.5000000001`` 会被视为两档不同滑点；
    这样既不会掩盖用户显式写出的细微差别，也避免引入随量级变化的容差。代价是
    这类差别小于展示精度的取值会在图例与报告表格里显示为同一个数，需要调用方
    自行避免。

    参数：
        slippage_bps_candidates: 命令行或调用方给出的单边滑点候选，单位为基点；
            必须是数值序列，字符串与布尔值会被拒绝，以免 ``"12"`` 被逐字符拆成
            两档滑点、``True`` 被当成 1 基点。
        slippage_bps: 基准单边滑点，单位为基点；与之相同的候选会被剔除。

    返回：
        去重后的候选滑点列表，顺序与输入一致；全部与基准重复时返回空列表。
    """

    if isinstance(slippage_bps_candidates, (str, bytes)):
        raise ValueError("slippage_bps_candidates 必须是数值序列，不能是字符串")
    candidates: list[float] = []
    for candidate in slippage_bps_candidates:
        if isinstance(candidate, (str, bytes, bool)):
            raise ValueError(
                f"slippage_bps_candidates 的取值 {candidate!r} 不是数值"
            )
        try:
            value = float(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"slippage_bps_candidates 的取值 {candidate!r} 不是数值"
            ) from exc
        if not np.isfinite(value) or not 0.0 <= value < 10_000.0:
            raise ValueError(
                f"slippage_bps_candidates 的取值 {value!r} 必须在 [0, 10000) 范围内"
            )
        if value == float(slippage_bps) or value in candidates:
            continue
        candidates.append(value)
    return candidates


def _slippage_sensitivity(
    daily: pd.DataFrame,
    slippage_bps: float,
    commission_bps: float,
    candidates: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在同一批每日选股上重算不同滑点的 Top N 净值曲线与汇总指标。

    每日选股只由模型分数决定，与成本无关，因此对比曲线复用基准回测已经选出的
    组合与毛收益，只替换滑点；手续费保持与基准一致，这样曲线差异可以完全归因
    于滑点。候选净值按「先等权平均毛收益、再乘成交乘数」计算，它与基准路径的
    「逐证券扣成本再等权平均」在**等权**下代数恒等，引入非等权权重后该等价关系
    会失效。基准滑点始终作为第一条曲线输出，其逐日净收益直接取自基准回测结果，
    保证与汇总指标严格一致。

    参数：
        daily: 基准回测的按日结果，至少含 ``target_date``、``gross_return``
            与 ``net_return``。
        slippage_bps: 基准单边滑点，单位为基点。
        commission_bps: 单边手续费率，单位为基点；对比曲线沿用该值。
        candidates: 已由 ``_normalize_slippage_candidates`` 校验并去重的候选
            滑点列表，单位为基点。

    返回：
        长表形式的逐日对比曲线（滑点、目标日期、净收益、累计净值、是否基准）
        和每档滑点的汇总指标；候选为空时返回两个空表。
    """

    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    gross = daily["gross_return"].to_numpy(dtype=float)
    baseline_net = daily["net_return"].to_numpy(dtype=float)
    levels: list[tuple[float, np.ndarray, bool]] = [
        (float(slippage_bps), baseline_net, True)
    ]
    for candidate in candidates:
        multiplier = _execution_multiplier(candidate, commission_bps)
        levels.append((float(candidate), (1.0 + gross) * multiplier - 1.0, False))

    # 先算好基准年化，避免相对基准的差值依赖基准恰好排在第一行。
    baseline_annualized = _compound_metrics(baseline_net)["annualized_return"]
    curve_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | bool]] = []
    for level, net_returns, is_baseline in levels:
        equity = np.cumprod(1.0 + net_returns)
        curve_frames.append(
            pd.DataFrame(
                {
                    "slippage_bps": level,
                    "target_date": daily["target_date"].to_numpy(),
                    "net_return": net_returns,
                    "equity": equity,
                    "is_baseline": is_baseline,
                }
            )
        )
        metrics = _compound_metrics(net_returns)
        _, drawdowns = _drawdown_curve(equity)
        metric_rows.append(
            {
                "slippage_bps": level,
                "is_baseline": is_baseline,
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": float(drawdowns.min()),
                "annualized_return_minus_baseline": (
                    metrics["annualized_return"] - baseline_annualized
                ),
            }
        )
    return (
        pd.concat(curve_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
    )
