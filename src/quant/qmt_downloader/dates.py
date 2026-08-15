# -*- coding: utf-8 -*-
"""QMT 时间戳、日期索引与有限数值的标准化函数。"""

import datetime
import math


def normalize_date(value):
    """将 QMT 日期、秒或毫秒时间戳统一为八位日期。

    参数：
        value: QMT 返回的日期索引、日期对象、秒时间戳或毫秒时间戳。

    返回：
        ``YYYYMMDD`` 字符串；无法识别时返回 ``None``。
    """
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, datetime.date):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 8 and digits[:2] in ("19", "20"):
        return digits[:8]
    try:
        timestamp = float(value)
        if timestamp > 100000000000:
            timestamp /= 1000.0
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
    except (TypeError, ValueError, OverflowError):
        return None


def finite_number(value):
    """将 QMT 数值转换为有限浮点数并识别最大双精度哨兵值。

    参数：
        value: 行情或财务接口返回的标量；允许数字字符串和空值。

    返回：
        有限 ``float``；缺失、无穷或 QMT 哨兵值返回 ``None``。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or abs(number) > 1e100:
        return None
    return number
