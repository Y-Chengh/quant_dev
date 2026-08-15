# -*- coding: utf-8 -*-
"""运行器使用的无状态工具：生命周期窗口、分批、任务键与问题判定。"""

import hashlib
import json

import pandas as pd

from ..gateway import FINANCE_FIELDS


def _lifecycle_windows(instrument_info):
    """把上市退市信息表转成证券生命周期映射。

    参数：
        instrument_info: ``fetch_instrument_info`` 返回的证券生命周期信息表。

    返回：
        ``code -> (open_date, expire_date)`` 字典；日期缺失时用空字符串表示该侧
        不设边界，调用方据此判定某交易日是否处于存续期。
    """
    lifecycle = {}
    if instrument_info is None or instrument_info.empty:
        return lifecycle
    for row in instrument_info.to_dict("records"):
        open_date = row.get("open_date")
        expire_date = row.get("expire_date")
        lifecycle[str(row.get("code"))] = (
            "" if pd.isna(open_date) else str(open_date or ""),
            "" if pd.isna(expire_date) else str(expire_date or ""),
        )
    return lifecycle


def _codes_alive_on(symbols, lifecycle, trade_date):
    """筛出在指定交易日处于存续期的证券代码。

    参数：
        symbols: 当前证券池代码序列。
        lifecycle: ``_lifecycle_windows`` 生成的生命周期映射；为空时不做筛选。
        trade_date: 八位交易日。

    返回：
        该交易日应当有行情的代码列表；判定口径与 ``_filter_kline_issues`` 一致，
        因此生成阶段跳过的证券日与兜底过滤会删除的完全相同。
    """
    if not lifecycle:
        return list(symbols)
    output = []
    for code in symbols:
        open_date, expire_date = lifecycle.get(str(code), ("", ""))
        if open_date and trade_date < open_date:
            continue
        if expire_date and trade_date > expire_date:
            continue
        output.append(code)
    return output


def _iter_batches(values, batch_size):
    """按固定大小顺序切分证券池。

    参数：
        values: 已稳定排序的证券代码列表。
        batch_size: 每批最多证券数量。

    返回：
        逐批产生证券代码列表的生成器。
    """
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def _format_elapsed(seconds):
    """把耗时秒数格式化为便于阅读的 ``H:MM:SS`` 文本。

    全市场长区间回溯常以小时计，因此不折算为天，小时位直接累加且不补零；秒数向下
    取整，避免摘要里出现与日志时间戳对不上的进位。

    参数：
        seconds: 非负耗时秒数；负值按 0 处理，防止时钟回拨产生负号文本。

    返回：
        形如 ``0:03:21`` 或 ``17:05:44`` 的耗时文本。
    """
    total = int(max(0.0, seconds))
    return "{0}:{1:02d}:{2:02d}".format(total // 3600, total % 3600 // 60, total % 60)


def _make_job_key(config, symbols):
    """根据影响抽取结果的配置生成稳定任务标识。

    参数：
        config: 当前 ``DownloaderConfig``。
        symbols: 已解析并排序的最终证券池。

    返回：
        ``qmt_`` 前缀加 SHA-256 前二十位的稳定字符串。
    """
    payload = {
        "schema_version": 2,
        "mode": config.mode,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "finance_lookback_start": config.finance_lookback_start,
        "symbols": list(symbols),
        "datasets": list(config.datasets),
        "download_kline": config.download_kline,
        "batch_size": config.batch_size,
        "allow_partial_finance": config.allow_partial_finance,
        "calendar_symbol": config.calendar_symbol,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "qmt_{0}".format(hashlib.sha256(encoded).hexdigest()[:20])


def _ensure_columns(frame, columns):
    """为 staging 读取结果补齐规范列并保持额外列。

    参数：
        frame: 从一个或多个 staging CSV 合并的数据表。
        columns: 当前数据集必须包含的规范列顺序。

    返回：
        规范列在前、额外列在后的新 ``DataFrame``。
    """
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.Series(index=output.index, dtype="object")
    extras = [column for column in output.columns if column not in columns]
    return output[list(columns) + sorted(extras)]


def _finance_templates():
    """构造所有财务表的空表和稳定列结构。

    返回：
        按逻辑财务表名映射的空 ``DataFrame``，用于流式片段补列和最终表头。
    """
    output = {}
    for table_name, fields in FINANCE_FIELDS.items():
        columns = ["code", "report_date", "announce_date"] + [
            field.split(".", 1)[1] for field in fields[2:]
        ]
        output[table_name] = pd.DataFrame(columns=columns)
    return output


def _contains_error(issues):
    """判断一次网关调用是否返回接口级错误。

    参数：
        issues: 网关返回的结构化问题字典列表。

    返回：
        任一问题严重级别为 ``ERROR`` 时返回 ``True``。
    """
    return any(item.get("level") == "ERROR" for item in issues)


def _contains_finance_missing(issues):
    """判断财务问题中是否存在部分表、证券或公告日缺失。

    参数：
        issues: 财务网关返回的结构化问题字典列表。

    返回：
        存在 ``finance_raw`` 警告时返回 ``True``，用于严格完整性模式保持可重试。
    """
    return any(
        item.get("level") == "WARNING"
        and str(item.get("dataset", "")).startswith("finance_raw/")
        for item in issues
    )
