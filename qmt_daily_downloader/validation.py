# -*- coding: utf-8 -*-
"""对已抽取数据执行不会静默修正金融含义的质量检查。"""

import pandas as pd


def validate_kline(frame):
    """检查日线主键、价格关系和成交量金额。

    参数：
        frame: 标准化后的日线 ``DataFrame``，应包含代码、交易日和 OHLCV 字段。

    返回：
        问题字典列表；函数只报告问题，不会替换或删除原始值。
    """
    issues = []
    if frame.empty:
        return [
            _issue("ERROR", "kline_1d", "", "", "请求区间没有任何日线记录")
        ]
    duplicate = frame.duplicated(["code", "trade_date"], keep=False)
    for _, row in frame[duplicate].iterrows():
        issues.append(_issue("ERROR", "kline_1d", row["code"], row["trade_date"], "证券和交易日主键重复"))
    for _, row in frame.iterrows():
        values = [row.get("open"), row.get("high"), row.get("low"), row.get("close")]
        finite = [value for value in values if pd.notna(value)]
        if len(finite) != 4:
            issues.append(_issue("WARNING", "kline_1d", row["code"], row["trade_date"], "OHLC 存在缺失值"))
        elif row["high"] < max(row["open"], row["close"], row["low"]):
            issues.append(_issue("ERROR", "kline_1d", row["code"], row["trade_date"], "最高价低于开收低价格之一"))
        elif row["low"] > min(row["open"], row["close"], row["high"]):
            issues.append(_issue("ERROR", "kline_1d", row["code"], row["trade_date"], "最低价高于开收高价格之一"))
        for column in ("volume", "amount"):
            value = row.get(column)
            if pd.notna(value) and value < 0:
                issues.append(_issue("ERROR", "kline_1d", row["code"], row["trade_date"], "{0} 为负数".format(column)))
    return issues


def find_missing_kline(frame, symbols, trade_dates):
    """按实际交易日检查股票批次中的日线缺口。

    参数：
        frame: 所有批次合并后的标准日线数据。
        symbols: 本次请求的完整证券代码列表。
        trade_dates: 从全批次日线并集得到的实际交易日序列，不包含自然日周末。

    返回：
        每个填充后仍缺失的证券日一条 ``WARNING``；停牌日应由 QMT 的
        ``suspend_flag=1`` 补齐行表示，不会进入缺口日志。
    """
    existing = set()
    if not frame.empty:
        existing = set(zip(frame["code"].astype(str), frame["trade_date"].astype(str)))
    issues = []
    for code in symbols:
        for trade_date in trade_dates:
            if (code, trade_date) not in existing:
                issues.append(_issue("WARNING", "kline_1d", code, trade_date, "填充后仍无日线；可能为未上市、退市或本地缓存缺失"))
    return issues


def _issue(level, dataset, code, date_value, message):
    """构造质量检查问题字典。

    参数：
        level: 问题严重级别。
        dataset: 数据集名称。
        code: 相关证券代码。
        date_value: 相关交易日。
        message: 供用户排查的说明。

    返回：
        与外部问题报告列结构一致的字典。
    """
    return {
        "level": level,
        "dataset": dataset,
        "code": code,
        "date": date_value,
        "message": message,
    }
