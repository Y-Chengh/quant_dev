# -*- coding: utf-8 -*-
"""对已抽取数据执行质量检查；仅有的价格关系修正必须显式留痕。"""

import pandas as pd


PRICE_COLUMNS = ["open", "high", "low", "close"]
QUANTITY_COLUMNS = ["volume", "amount"]


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
    for _, row in frame[_suspect_price_rows(frame)].iterrows():
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


def _suspect_price_rows(frame):
    """用向量化掩码筛出可能存在价格或成交量问题的行。

    掩码只用于缩小逐行检查范围，判定结论仍由逐行逻辑给出，因此掩码允许多选、
    不允许漏选。列缺失或不是数值类型时无法用向量化比较复现逐行语义（例如数字
    字符串按字典序比较），此时整表退回逐行检查。

    参数：
        frame: 标准化后的非空日线 ``DataFrame``。

    返回：
        与 ``frame`` 行对齐的布尔 ``Series``。
    """
    columns = PRICE_COLUMNS + QUANTITY_COLUMNS
    if any(
        column not in frame.columns or frame[column].dtype.kind not in "ifb"
        for column in columns
    ):
        return pd.Series(True, index=frame.index)
    complete = frame[PRICE_COLUMNS].notna().all(axis=1)
    suspect = ~complete
    suspect = suspect | (
        complete & (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
    )
    suspect = suspect | (
        complete & (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    return suspect | (frame[QUANTITY_COLUMNS] < 0).any(axis=1)


def correct_kline_prices(frame):
    """修正源数据中最高最低价与开收价关系不成立的行并留痕。

    大 QMT 历史缓存偶见 ``high < max(open, close, low)`` 之类的脏行（例如
    1994 年个别股票），且重新下载结果不变；此处把最高价抬到开收低三者
    最大值、最低价压到开收高三者最小值，其余字段保持原值。

    参数：
        frame: 标准化后的日线 ``DataFrame``，应包含代码、交易日和 OHLC 字段。

    返回：
        ``(corrected, records, issues)`` 三元组：修正后的新表（原表不改动）、
        供 ``reports/line_correct`` 报告落盘的修正记录列表，以及与之一一对应
        的 ``WARNING`` 问题列表；无脏行时后两者为空列表。
    """
    if frame.empty:
        return frame, [], []
    corrected = frame.copy()
    records = []
    issues = []
    complete = corrected[["open", "high", "low", "close"]].notna().all(axis=1)
    high_floor = corrected[["open", "close", "low"]].max(axis=1)
    bad_high = complete & (corrected["high"] < high_floor)
    for index, row in corrected.loc[bad_high].iterrows():
        records.append(
            _correction(row, "high", row["high"], high_floor[index], "最高价低于开收低价格之一")
        )
        issues.append(
            _issue("WARNING", "kline_1d", row["code"], row["trade_date"], "最高价低于开收低价格之一，已修正为开收低最大值")
        )
    corrected.loc[bad_high, "high"] = high_floor[bad_high]
    low_cap = corrected[["open", "close", "high"]].min(axis=1)
    bad_low = complete & (corrected["low"] > low_cap)
    for index, row in corrected.loc[bad_low].iterrows():
        records.append(
            _correction(row, "low", row["low"], low_cap[index], "最低价高于开收高价格之一")
        )
        issues.append(
            _issue("WARNING", "kline_1d", row["code"], row["trade_date"], "最低价高于开收高价格之一，已修正为开收高最小值")
        )
    corrected.loc[bad_low, "low"] = low_cap[bad_low]
    return corrected, records, issues


def _correction(row, field, original_value, corrected_value, message):
    """构造与 ``reports/line_correct`` 报告列结构一致的修正记录。

    参数：
        row: 被修正的日线行。
        field: 被修正的价格字段名。
        original_value: 源数据中的原始值。
        corrected_value: 修正后的值。
        message: 触发修正的价格关系描述。

    返回：
        修正记录字典。
    """
    return {
        "dataset": "kline_1d",
        "code": str(row["code"]),
        "trade_date": str(row["trade_date"]),
        "field": field,
        "original_value": original_value,
        "corrected_value": corrected_value,
        "message": message,
    }


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
