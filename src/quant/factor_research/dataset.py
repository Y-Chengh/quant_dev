from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging

import pandas as pd

from .factors import DEFAULT_FEATURES
from .timing import log_elapsed


logger = logging.getLogger(__name__)

PREVIOUS_RETURN_COLUMNS = (
    "previous_close_to_close_return",
    "previous_open_to_close_return",
)
REPORT_RETURN_COLUMNS = (
    *PREVIOUS_RETURN_COLUMNS,
    "target_close_to_previous_close_return",
)
DATASET_RESERVED_COLUMNS = frozenset(
    {
        "trade_date",
        "feature_date",
        "target_date",
        "code",
        "target_return",
        "label",
        *REPORT_RETURN_COLUMNS,
    }
)


@dataclass(frozen=True)
class DatasetSplit:
    """保存按目标日期切分且互不重叠的训练集与验证集。"""

    train: pd.DataFrame
    validation: pd.DataFrame


def _with_forward_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """在按证券排序的副本上附加下一有效交易日目标列。

    参数：
        daily: 每行为一个证券交易日的日频行情表。
    """

    required = {"code", "trade_date", "open", "close"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"构建预测目标缺少列: {sorted(missing)}")
    data = daily.sort_values(["code", "trade_date"]).copy()
    grouped = data.groupby("code", sort=False)
    # 两个报告口径都以特征日（即目标日的前一有效交易日）为“前日”。收盘对收盘
    # 先在证券内后移一期；首个观测没有前前日收盘，因此保持缺失，不跨证券填充。
    previous_close = grouped["close"].shift(1)
    data["previous_close_to_close_return"] = data["close"] / previous_close - 1
    data["previous_open_to_close_return"] = data["close"] / data["open"] - 1
    data["target_date"] = grouped["trade_date"].shift(-1)
    next_open = grouped["open"].shift(-1)
    next_close = grouped["close"].shift(-1)
    data["target_return"] = next_close / next_open - 1
    # t 日收盘相对 t-1 日收盘属于目标日收盘后才可见的报告结果，不作为模型特征。
    data["target_close_to_previous_close_return"] = next_close / data["close"] - 1
    data["label"] = (data["target_return"] > 0).astype("Int8")
    return data


def build_forward_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """构建下一有效交易日开盘至收盘收益及方向标签。

    返回结果不包含任何特征列，供因子搜索一次性复用。特征日期为 D，目标日期
    是同一证券 D 之后实际存在行情的下一天；最后一个交易日因为没有目标而丢弃。

    参数：
        daily: 含证券、交易日及开收盘价的日频行情表。
    """

    data = _with_forward_targets(daily)
    columns = ["trade_date", "target_date", "code", "target_return", "label"]
    return (
        data.loc[data["target_date"].notna(), columns]
        .rename(columns={"trade_date": "feature_date"})
        .sort_values(["target_date", "code"])
        .reset_index(drop=True)
    )


@log_elapsed(logger, "方向预测数据集构建")
def build_direction_dataset(
    daily_features: pd.DataFrame,
    feature_columns: list[str] | None = None,
    args: argparse.Namespace | None = None,
) -> pd.DataFrame:
    """用 D 日收盘后的特征预测下一有效交易日从开盘到收盘的方向。

    输出额外保留 D 日收盘相对 D-1 日收盘、D 日收盘相对 D 日开盘，以及目标日
    收盘相对 D 日收盘的收益率，供预测报告展示；这些列不属于
    ``feature_columns``，不会进入模型输入。

    参数：
        daily_features: 已合并日频行情和因子列的特征表。
        feature_columns: 要输出的因子列名；缺省时使用默认因子集。
        args: 为兼容调用链保留的命令行命名空间，不参与目标计算。

    返回：
        按目标日期和证券排序的监督学习样本，包含模型特征、目标及前日行情上下文。
    """
    feature_columns = feature_columns or DEFAULT_FEATURES
    reserved = set(feature_columns).intersection(DATASET_RESERVED_COLUMNS)
    if reserved:
        raise ValueError(f"因子列使用了数据集保留名: {sorted(reserved)}")
    missing = set(feature_columns).difference(daily_features.columns)
    if missing:
        raise ValueError(f"缺少因子列: {sorted(missing)}")
    data = _with_forward_targets(daily_features)
    columns = [
        "trade_date",
        "target_date",
        "code",
        *feature_columns,
        *REPORT_RETURN_COLUMNS,
        "target_return",
        "label",
    ]
    return (
        data.loc[data["target_date"].notna(), columns]
        .rename(columns={"trade_date": "feature_date"})
        .sort_values(["target_date", "code"])
        .reset_index(drop=True)
    )


def split_by_date(dataset: pd.DataFrame, validation_start: str | pd.Timestamp) -> DatasetSplit:
    """按配置日期切分；validation_start 当天属于验证区间。

    参数：
        dataset: 含 ``target_date`` 的已建模样本表。
        validation_start: 验证集首个目标日期，该日期包含在验证集内。
    """
    cutoff = pd.Timestamp(validation_start)
    train = dataset.loc[dataset["target_date"] < cutoff].copy()
    validation = dataset.loc[dataset["target_date"] >= cutoff].copy()
    if train.empty:
        raise ValueError(f"验证开始日期 {cutoff.date()} 之前没有训练样本")
    if validation.empty:
        raise ValueError(f"验证开始日期 {cutoff.date()} 之后没有验证样本")
    return DatasetSplit(train=train, validation=validation)
