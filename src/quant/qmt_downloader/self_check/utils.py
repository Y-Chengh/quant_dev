# -*- coding: utf-8 -*-
"""与自检器状态无关的解析、校验与文件摘要工具。"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..dates import normalize_date as _qmt_normalize_date


def _read_calendar_csv(path: Path) -> list[str]:
    """从一列或含标准日期列的 CSV 读取去重交易日。

    参数：
        path: QMT 导出的交易日历 CSV，优先读取 ``trade_date``、``date`` 或 ``time`` 列。

    返回：
        严格升序且去重的八位交易日期列表。
    """

    calendar_path = Path(path)
    try:
        frame = pd.read_csv(str(calendar_path), encoding="utf-8-sig", dtype=str)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("无法读取交易日历 {0}: {1}".format(calendar_path, exc)) from exc
    column = next((name for name in ("trade_date", "date", "time") if name in frame.columns), None)
    if column is None and len(frame.columns) == 1:
        column = str(frame.columns[0])
    if column is None:
        raise ValueError("交易日历必须包含 trade_date、date、time 之一或仅有一列")
    dates = [_normalize_date_text(value) for value in frame[column].tolist()]
    invalid = [str(value) for value, normalized in zip(frame[column].tolist(), dates) if normalized is None]
    if invalid:
        raise ValueError("交易日历包含非法日期，示例: {0}".format(_sample(invalid)))
    return sorted(set(value for value in dates if value is not None))


def _normalize_date_text(value: Any) -> str | None:
    """将 CSV 日期、时间戳或数字文本规范为八位日期。

    参数：
        value: 可能来自 QMT CSV、JSON 或 pandas 的日期值。

    返回：
        合法的 ``YYYYMMDD`` 日期；空值、无日期哨兵或非法值返回 ``None``。
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "99999999", "0"}:
        return None
    qmt_date = _qmt_normalize_date(value)
    if qmt_date is not None:
        try:
            datetime.strptime(qmt_date, "%Y%m%d")
            return qmt_date
        except ValueError:
            return None
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 8:
        candidate = digits[:8]
        try:
            datetime.strptime(candidate, "%Y%m%d")
            return candidate
        except ValueError:
            pass
    try:
        return pd.Timestamp(text).strftime("%Y%m%d")
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_lifecycle_value(value: Any) -> tuple[str | None, bool]:
    """解析上市或退市日期并区分合法空值与非法非空文本。

    参数：
        value: instrument_info 中的上市日期或退市日期原始值。

    返回：
        ``(日期, 是否为非法非空值)``；空值、99999999 以及 QMT 常见的
        19700101/19700427 无期限哨兵返回 ``(None, False)``。
    """

    if value is None:
        return None, False
    text = str(value).strip()
    if not text or text.lower() in {
        "nan", "nat", "none", "99999999", "0", "19700101", "19700427"
    }:
        return None, False
    normalized = _normalize_date_text(value)
    return normalized, normalized is None


def _optional_int(value: Any) -> int | None:
    """把完成标记中的行数转换为整数并识别非法值。

    参数：
        value: ``_SUCCESS.json`` 中的 rows 原始值。

    返回：
        非负整数；缺失、布尔值或非法文本返回 ``None``。
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted >= 0 else None


def _optional_date(value: str | None, field_name: str) -> None:
    """校验可选八位日期参数。

    参数：
        value: 可为空的 ``YYYYMMDD`` 日期字符串。
        field_name: 用于错误消息的配置字段名称。

    返回：
        无返回值；日期非法时抛出 ``ValueError``。
    """

    if value is not None and _normalize_date_text(value) != value:
        raise ValueError("{0} 必须是 YYYYMMDD 日期".format(field_name))


def _finite_float(value: Any) -> float | None:
    """把行情字段转换为有限浮点数并排除 QMT 最大双精度哨兵值。

    参数：
        value: CSV 中的价格、成交量、成交额或停牌标志原始值。

    返回：
        有限浮点数；空值、无穷、非数字或哨兵值返回 ``None``。
    """

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted) or abs(converted) >= 1.7976931348623157e308:
        return None
    return converted


def _file_sha256(path: Path) -> str:
    """分块计算本地文件 SHA-256。

    参数：
        path: 需要校验内容是否与完成标记一致的文件路径。

    返回：
        六十四位小写十六进制摘要。
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    """读取 CSV 物理行数并扣除一行表头。

    参数：
        path: 不应含嵌入换行字段的 QMT 标准分区 CSV。

    返回：
        不含表头的非负物理行数。
    """

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _sample(values: Iterable[Any], limit: int = 10) -> str:
    """把较长代码或日期序列压缩为可读示例。

    参数：
        values: 需要在错误证据中展示的任意可迭代值。
        limit: 最多展示的元素数量，缺省为十个。

    返回：
        逗号分隔的示例文本；超出上限时追加省略说明。
    """

    items = [str(value) for value in values]
    shown = items[:limit]
    suffix = " ... 共 {0} 项".format(len(items)) if len(items) > limit else ""
    return ", ".join(shown) + suffix
