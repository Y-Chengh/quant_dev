# -*- coding: utf-8 -*-
"""将原始财报转换为严格按公告日可见的日频财务快照。"""

import pandas as pd


def materialize_finance_daily(table_frames, trade_dates, symbols):
    """构造每个交易日收盘时可安全使用的财务快照。

    参数：
        table_frames: 按财务表名映射的原始财报 ``DataFrame``；每行包含报告期和公告日。
        trade_dates: 需要生成快照的八位交易日序列。
        symbols: 证券代码序列；结果对每个证券和交易日分别计算。

    返回：
        宽表 ``DataFrame``。仅使用 ``announce_date < trade_date`` 的记录，避免把当天盘后
        公告错误地用于当天收盘特征。
    """
    dates = sorted(set(str(value) for value in trade_dates if value))
    codes = sorted(set(str(value) for value in symbols if value))
    snapshots = {}
    for table_name, frame in table_frames.items():
        snapshots[table_name] = _build_table_snapshots(frame, dates, codes)

    rows = []
    for code in codes:
        for trade_date in dates:
            row = {"code": code, "trade_date": trade_date}
            has_finance = False
            for table_name in sorted(snapshots):
                latest = snapshots[table_name].get((code, trade_date))
                if latest is None:
                    continue
                row["{0}_report_date".format(table_name)] = latest["report_date"]
                row["{0}_announce_date".format(table_name)] = latest["announce_date"]
                for column in latest:
                    if column in ("code", "report_date", "announce_date"):
                        continue
                    row["{0}_{1}".format(table_name, column)] = latest[column]
                has_finance = True
            row["has_finance"] = 1 if has_finance else 0
            rows.append(row)
    return pd.DataFrame(rows)


def _build_table_snapshots(frame, trade_dates, symbols):
    """以单次游标扫描构造一张财务表的证券日快照。

    参数：
        frame: 单张原始财务表，包含代码、报告期和公告日。
        trade_dates: 已排序的八位交易日列表。
        symbols: 已排序的证券代码列表。

    返回：
        以 ``(code, trade_date)`` 为键、最新可见财报字典为值的映射。每条财报只扫描
        一次，避免对每个证券日重复执行 pandas 过滤。
    """
    if frame.empty or "announce_date" not in frame.columns:
        return {}
    current = frame[frame["announce_date"].notna()].copy()
    if current.empty:
        return {}
    current["announce_date"] = current["announce_date"].astype(str)
    current["report_date"] = current["report_date"].astype(str)
    output = {}
    for code in symbols:
        records = current[current["code"] == code].sort_values(
            ["announce_date", "report_date"], kind="mergesort"
        ).to_dict("records")
        pointer = 0
        latest = None
        for trade_date in trade_dates:
            while pointer < len(records) and records[pointer]["announce_date"] < trade_date:
                candidate = records[pointer]
                if latest is None or candidate["report_date"] >= latest["report_date"]:
                    latest = candidate
                pointer += 1
            if latest is not None:
                output[(code, trade_date)] = latest
    return output


def finance_daily_columns(table_frames):
    """根据已配置财务表生成稳定的日快照列顺序。

    参数：
        table_frames: 按财务表名映射的原始财报 ``DataFrame``。

    返回：
        包含代码、交易日、各表来源日期、业务字段和可用标志的列名列表。
    """
    columns = ["code", "trade_date"]
    for table_name in sorted(table_frames):
        columns.extend(
            [
                "{0}_report_date".format(table_name),
                "{0}_announce_date".format(table_name),
            ]
        )
        for column in table_frames[table_name].columns:
            if column in ("code", "report_date", "announce_date"):
                continue
            columns.append("{0}_{1}".format(table_name, column))
    columns.append("has_finance")
    return columns
