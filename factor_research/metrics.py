from __future__ import annotations

import numpy as np
import pandas as pd


def information_coefficient(
    probability: np.ndarray,
    target_return: np.ndarray,
) -> float:
    """计算预测上涨概率与目标收益率之间的 Pearson 相关系数。

    仅使用两组数据中都为有限值的样本；有效样本少于两个，或任一序列没有
    波动时，IC 不可定义并返回 NaN。
    """
    score = np.asarray(probability, dtype=float)
    returns = np.asarray(target_return, dtype=float)
    if score.shape != returns.shape:
        raise ValueError("probability and target_return must have the same shape")

    valid = np.isfinite(score) & np.isfinite(returns)
    if valid.sum() < 2:
        return float("nan")

    centered_score = score[valid] - score[valid].mean()
    centered_returns = returns[valid] - returns[valid].mean()
    denominator = np.sqrt(
        np.dot(centered_score, centered_score)
        * np.dot(centered_returns, centered_returns)
    )
    if denominator == 0:
        return float("nan")
    return float(np.dot(centered_score, centered_returns) / denominator)


def rank_information_coefficient(
    score: np.ndarray,
    target_return: np.ndarray,
) -> float:
    """计算平均秩处理并列值后的 Spearman Rank IC。"""
    score = np.asarray(score, dtype=float)
    returns = np.asarray(target_return, dtype=float)
    if score.shape != returns.shape:
        raise ValueError("score and target_return must have the same shape")
    valid = np.isfinite(score) & np.isfinite(returns)
    if valid.sum() < 2:
        return float("nan")
    score_rank = pd.Series(score[valid]).rank(method="average").to_numpy()
    return_rank = pd.Series(returns[valid]).rank(method="average").to_numpy()
    return information_coefficient(score_rank, return_rank)


def daily_cross_sectional_ic(
    predictions: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """按目标交易日计算预测分数与实际收益率的横截面 IC 和 Rank IC。"""
    required = {"target_date", "target_return", score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Missing columns for cross-sectional IC: {sorted(missing)}"
        )

    rows: list[dict[str, object]] = []
    for target_date, group in predictions.groupby("target_date", sort=True):
        score = group[score_column].to_numpy(dtype=float)
        returns = group["target_return"].to_numpy(dtype=float)
        valid = np.isfinite(score) & np.isfinite(returns)
        rows.append(
            {
                "target_date": target_date,
                "samples": int(valid.sum()),
                "ic": information_coefficient(score, returns),
                "rank_ic": rank_information_coefficient(score, returns),
            }
        )
    return pd.DataFrame(rows, columns=["target_date", "samples", "ic", "rank_ic"])


def cross_sectional_ic_metrics(daily_ic: pd.DataFrame) -> dict[str, float]:
    """汇总每日横截面 IC、ICIR 和 IC 胜率。

    ICIR 使用有效交易日日 IC 均值除以样本标准差，不做年化；少于两个有效日或
    标准差为零时返回 NaN。IC 胜率为有效交易日中 IC 大于数值零容差的比例，
    避免理论零相关因浮点舍入被误计为胜出。

    参数：
        daily_ic: 含 ``ic`` 和 ``rank_ic`` 列的逐日横截面指标表；非有限值不参与
            对应指标计算。

    返回：
        IC、Rank IC、ICIR、IC 胜率及对应有效交易日数的汇总字典。
    """
    required = {"ic", "rank_ic"}
    missing = required.difference(daily_ic.columns)
    if missing:
        raise ValueError(f"Missing daily IC columns: {sorted(missing)}")
    ic = daily_ic["ic"].to_numpy(dtype=float)
    rank_ic = daily_ic["rank_ic"].to_numpy(dtype=float)
    valid_ic = np.isfinite(ic)
    valid_rank_ic = np.isfinite(rank_ic)
    finite_ic = ic[valid_ic]
    ic_mean = float(np.mean(finite_ic)) if finite_ic.size else float("nan")
    ic_std = (
        float(np.std(finite_ic, ddof=1))
        if finite_ic.size >= 2
        else float("nan")
    )
    return {
        "ic": ic_mean,
        "rank_ic": (
            float(np.mean(rank_ic[valid_rank_ic]))
            if valid_rank_ic.any()
            else float("nan")
        ),
        "icir": (
            float(ic_mean / ic_std)
            if np.isfinite(ic_std) and ic_std > 0.0
            else float("nan")
        ),
        "ic_win_rate": (
            float(np.mean(finite_ic > 1e-12))
            if finite_ic.size
            else float("nan")
        ),
        "ic_dates": float(valid_ic.sum()),
        "rank_ic_dates": float(valid_rank_ic.sum()),
    }


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_return: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    ic = (
        information_coefficient(probability, target_return)
        if target_return is not None
        else None
    )
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    prediction = (probability >= 0.5).astype(int)
    accuracy = float(np.mean(prediction == y))
    recalls = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    # 对相同概率使用平均秩，避免树模型大量并列预测时AUC产生偏差。
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2
        start = end
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = sorted_ranks
    positive, negative = y == 1, y == 0
    auc = (
        float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))
        if positive.any() and negative.any()
        else float("nan")
    )
    metrics = {
        "samples": float(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "auc": auc,
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
    }
    if ic is not None:
        metrics["ic"] = ic
    return metrics


def regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """计算连续涨跌幅预测的误差、方向准确率和 Pearson IC。"""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and prediction must have the same shape")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        raise ValueError("regression metrics require at least one finite pair")
    actual = actual[valid]
    predicted = predicted[valid]
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum(error**2)) / denominator
        if denominator > 0
        else float("nan")
    )
    return {
        "samples": float(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "direction_accuracy": float(np.mean((predicted > 0) == (actual > 0))),
        "ic": information_coefficient(predicted, actual),
    }


def daily_accuracy_trend(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize out-of-sample prediction accuracy for each target date."""
    required_columns = {"target_date", "label"}
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for daily accuracy trend: {sorted(missing_columns)}")

    frame = predictions.loc[:, ["target_date", "label"]].copy()
    if "prediction" in predictions:
        predicted_direction = predictions["prediction"].to_numpy(dtype=int)
    elif "up_probability" in predictions:
        predicted_direction = (
            predictions["up_probability"].to_numpy(dtype=float) >= 0.5
        ).astype(int)
    else:
        raise ValueError(
            "Missing prediction or up_probability for daily accuracy trend"
        )
    frame["correct"] = predicted_direction == frame["label"].to_numpy(dtype=int)
    trend = (
        frame.groupby("target_date", as_index=False, sort=True)
        .agg(samples=("correct", "size"), accuracy=("correct", "mean"))
    )
    trend["accuracy"] = trend["accuracy"].astype(float)
    trend["accuracy_change"] = trend["accuracy"].diff()
    return trend
