# -*- coding: utf-8 -*-
"""人工核实的 QMT 源数据勘误表：装载、按行应用与转成宽表供入库 SQL 使用。

勘误表记录的是人工确认为 QMT 原始数据本身有误（而非本仓库落盘或自检逻辑的 bug）
的具体字段取值，格式见 docs/qmt_source_data_errata.csv：每行覆盖一只证券、一个
交易日的一个字段，列为 symbol、trade_date、field、value（其余列只用于审计追溯，
不参与覆盖逻辑）。self_check 和 market_data 入库都会在读完源数据之后、写出结果
之前应用同一份勘误表，避免两处各自维护一份口径不一致的修正。

本模块属于 quant.qmt_downloader 的顶层模块，会被大 QMT 内置 Python 直接导入，
因此只允许使用标准库和 pandas，不写 __future__ 导入，不写变量注解。
"""

from pathlib import Path

import pandas as pd

#: 勘误表允许覆盖的日线字段；必须是 gateway 日线列中的数值字段子集。
ERRATA_OVERRIDABLE_FIELDS = (
    "open", "high", "low", "close", "pre_close", "volume", "amount", "suspend_flag",
)

#: 勘误表必须具备的列；缺列时视为格式错误而不是静默忽略。
ERRATA_REQUIRED_COLUMNS = ("symbol", "trade_date", "field", "value")


def load_errata_overrides(path):
    """读取勘误表 CSV，按 (证券代码, 交易日) 聚合成字段覆盖字典。

    参数：
        path: 勘误表 CSV 路径；可以是 ``None`` 或不存在的路径，此时视为没有
            任何勘误记录，不报错。

    返回：
        以 ``(symbol, trade_date)`` 为键、``{field: value}`` 为值的字典。
        ``symbol`` 统一转为去空白的大写，``trade_date`` 统一转为去空白字符串，
        ``value`` 保持原始字符串，由调用方按目标列类型转换。
    """

    if path is None:
        return {}
    csv_path = Path(path)
    if not csv_path.is_file():
        return {}
    frame = pd.read_csv(str(csv_path), encoding="utf-8-sig", dtype=str)
    missing_columns = [name for name in ERRATA_REQUIRED_COLUMNS if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            "勘误表 {0} 缺少必需列: {1}".format(csv_path, ", ".join(missing_columns))
        )
    overrides = {}
    for _, row in frame.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        trade_date = str(row["trade_date"]).strip()
        field = str(row["field"]).strip()
        if not symbol or not trade_date:
            continue
        if field not in ERRATA_OVERRIDABLE_FIELDS:
            raise ValueError(
                "勘误表 {0} 中字段 {1} 不在允许覆盖的日线字段集合 {2} 内".format(
                    csv_path, field, ERRATA_OVERRIDABLE_FIELDS
                )
            )
        value = row["value"]
        value = "" if pd.isna(value) else str(value).strip()
        overrides.setdefault((symbol, trade_date), {})[field] = value
    return overrides


def apply_errata_overrides(frame, overrides, code_column="code", date_column="trade_date"):
    """把勘误覆盖应用到已读入内存的日线 DataFrame。

    参数：
        frame: 待修正的日线 DataFrame；原地修改并返回，调用方不需要的话可以
            忽略返回值。
        overrides: ``load_errata_overrides`` 的返回值。
        code_column: 证券代码所在列名。
        date_column: 交易日所在列名，取值须与 ``overrides`` 键的
            ``trade_date`` 使用同一种字符串格式（八位 ``YYYYMMDD``）。

    返回：
        ``(frame, applied)`` 二元组：``frame`` 是应用覆盖后的 DataFrame（可能
        是原对象的原地修改结果）；``applied`` 是实际生效的字段覆盖次数，用于
        调用方记录日志。空表或没有任何勘误记录时直接返回原表与 0，不做遍历。
    """

    if not overrides or frame.empty:
        return frame, 0
    applied = 0
    codes = frame[code_column].astype(str).str.strip().str.upper()
    dates = frame[date_column].astype(str).str.strip()
    for position, index in enumerate(frame.index):
        key = (codes.iat[position], dates.iat[position])
        fields = overrides.get(key)
        if not fields:
            continue
        for field, value in fields.items():
            if field not in frame.columns:
                continue
            frame.at[index, field] = _coerce_value(value)
            applied += 1
    return frame, applied


def errata_pivot_frame(overrides):
    """把勘误覆盖字典转成宽表，供市场数据入库 SQL 按列 ``COALESCE`` 使用。

    参数：
        overrides: ``load_errata_overrides`` 的返回值。

    返回：
        含 ``code``、``trade_date`` 以及 ``ERRATA_OVERRIDABLE_FIELDS`` 全部字段
        的 DataFrame；未被覆盖的字段值为 ``None``（在 SQL 侧即 ``NULL``）。
        没有任何勘误记录时返回空表，但列和数值列的 ``float64`` 类型仍然齐备，
        使调用方可以无条件注册该表并在 SQL 里做类型转换。
    """

    columns = ("code", "trade_date") + ERRATA_OVERRIDABLE_FIELDS
    if not overrides:
        empty = pd.DataFrame(columns=columns)
        empty["code"] = empty["code"].astype(str)
        empty["trade_date"] = empty["trade_date"].astype(str)
        for field in ERRATA_OVERRIDABLE_FIELDS:
            empty[field] = empty[field].astype("float64")
        return empty
    rows = []
    for (symbol, trade_date), fields in overrides.items():
        row = {"code": symbol, "trade_date": trade_date}
        for field in ERRATA_OVERRIDABLE_FIELDS:
            row[field] = float(fields[field]) if field in fields else None
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _coerce_value(value):
    """把勘误表里的字符串值转换成适合写回日线数值列的类型。

    参数：
        value: ``load_errata_overrides`` 保留下来的原始字符串值。

    返回：
        能转换成 ``float`` 时返回浮点数；否则原样返回字符串，留给调用方
        自行判断（当前允许覆盖的字段均为数值字段，理论上总能转换成功）。
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return value
