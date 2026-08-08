from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging

import pandas as pd

from .factors import DEFAULT_FEATURES
from .timing import log_elapsed


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


@log_elapsed(logger, "方向预测数据集构建")
def build_direction_dataset(
    daily_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    args: argparse.Namespace | None = None,
) -> pd.DataFrame:
    """用 D 日收盘后的特征预测下一有效交易日从开盘到收盘的方向。"""
    feature_columns = feature_columns or DEFAULT_FEATURES
    missing = set(feature_columns).difference(daily_features.columns)
    if missing:
        raise ValueError(f"缺少因子列: {sorted(missing)}")
    data = daily_features.sort_values(["code", "trade_date"]).copy()
    grouped = data.groupby("code", sort=False)
    data["target_date"] = grouped["trade_date"].shift(-1)
    next_open = grouped["open"].shift(-1)
    next_close = grouped["close"].shift(-1)
    data["target_return"] = next_close / next_open - 1
    data["label"] = (data["target_return"] > 0).astype("Int8")
    columns = ["trade_date", "target_date", "code", *feature_columns, "target_return", "label"]
    return (
        data.loc[data["target_date"].notna(), columns]
        .rename(columns={"trade_date": "feature_date"})
        .sort_values(["target_date", "code"])
        .reset_index(drop=True)
    )


def split_by_date(dataset: pd.DataFrame, validation_start: str | pd.Timestamp) -> DatasetSplit:
    """按配置日期切分；validation_start 当天属于验证区间。"""
    cutoff = pd.Timestamp(validation_start)
    train = dataset.loc[dataset["target_date"] < cutoff].copy()
    validation = dataset.loc[dataset["target_date"] >= cutoff].copy()
    if train.empty:
        raise ValueError(f"验证开始日期 {cutoff.date()} 之前没有训练样本")
    if validation.empty:
        raise ValueError(f"验证开始日期 {cutoff.date()} 之后没有验证样本")
    return DatasetSplit(train=train, validation=validation)
