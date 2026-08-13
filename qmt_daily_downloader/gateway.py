# -*- coding: utf-8 -*-
"""封装大 QMT 内置 Python 的行情、财务和除权接口。"""

import time
from datetime import datetime

import pandas as pd

from .dates import finite_number, normalize_date


KLINE_COLUMNS = [
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "suspend_flag",
]

KLINE_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preClose": "pre_close",
    "volume": "volume",
    "amount": "amount",
    "suspendFlag": "suspend_flag",
}

FINANCE_FIELDS = {
    "balance": [
        "ASHAREBALANCESHEET.m_timetag",
        "ASHAREBALANCESHEET.m_anntime",
        "ASHAREBALANCESHEET.tot_assets",
        "ASHAREBALANCESHEET.tot_liab",
    ],
    "income": [
        "ASHAREINCOME.m_timetag",
        "ASHAREINCOME.m_anntime",
        "ASHAREINCOME.revenue",
        "ASHAREINCOME.net_profit_excl_min_int_inc",
    ],
    "cashflow": [
        "ASHARECASHFLOW.m_timetag",
        "ASHARECASHFLOW.m_anntime",
        "ASHARECASHFLOW.net_cash_flows_oper_act",
    ],
    "capital": [
        "CAPITALSTRUCTURE.m_timetag",
        "CAPITALSTRUCTURE.m_anntime",
        "CAPITALSTRUCTURE.total_capital",
        "CAPITALSTRUCTURE.circulating_capital",
        "CAPITALSTRUCTURE.free_float_capital",
    ],
    "pershareindex": [
        "PERSHAREINDEX.m_timetag",
        "PERSHAREINDEX.m_anntime",
        "PERSHAREINDEX.s_fa_bps",
        "PERSHAREINDEX.s_fa_eps_basic",
        "PERSHAREINDEX.du_return_on_equity",
        "PERSHAREINDEX.inc_revenue_rate",
        "PERSHAREINDEX.inc_net_profit_rate",
        "PERSHAREINDEX.gear_ratio",
    ],
}

CORPORATE_ACTION_COLUMNS = [
    "code",
    "ex_date",
    "cash_dividend_per_share",
    "bonus_share_per_share",
    "capitalization_per_share",
    "rights_issue_per_share",
    "rights_issue_price",
    "share_reform_flag",
    "adjustment_factor",
]


class QmtGateway(object):
    """将大 QMT ContextInfo 返回值转换为稳定的表格结构。"""

    def __init__(self, context, history_downloader, logger, retry_count=3):
        """保存大 QMT 上下文及历史行情下载函数。

        参数：
            context: 大 QMT 策略传入的 ``ContextInfo`` 对象。
            history_downloader: 大 QMT 内置全局 ``download_history_data`` 函数。
            logger: 已配置终端和文件处理器的日志对象。
            retry_count: 单只证券接口失败后的最大尝试次数。
        """
        self.context = context
        self.history_downloader = history_downloader
        self.logger = logger
        self.retry_count = int(retry_count)

    def resolve_symbols(self, symbols, sector):
        """从显式代码列表或大 QMT 板块解析证券池。

        参数：
            symbols: 配置中显式填写的证券代码序列；非空时优先使用。
            sector: 大 QMT 客户端中的板块名称，仅在 ``symbols`` 为空时使用。

        返回：
            去重、排序并转为大写的证券代码列表。
        """
        if symbols:
            resolved = list(symbols)
        else:
            resolved = self.context.get_stock_list_in_sector(sector) or []
        output = sorted(set(str(code).strip().upper() for code in resolved if str(code).strip()))
        if not output:
            raise RuntimeError("证券池为空，请检查 symbols 或大 QMT 板块名称")
        invalid = [code for code in output if "." not in code]
        if invalid:
            raise ValueError("证券代码必须包含市场后缀: {0}".format(",".join(invalid)))
        return output

    def fetch_kline(self, symbols, start_date, end_date, download_first=True):
        """补充并读取指定证券的日 K 线。

        参数：
            symbols: 当前批次证券代码列表。
            start_date: 八位起始交易日，区间两端均包含。
            end_date: 八位结束交易日，区间两端均包含。
            download_first: 是否先调用大 QMT 历史行情下载接口更新本地缓存。

        返回：
            ``(DataFrame, issues)``，前者每行是一只证券一天的未复权行情，后者为问题列表。
        """
        issues = []
        available = []
        for code in symbols:
            try:
                if download_first:
                    self._retry(
                        lambda current=code: self.history_downloader(
                            current, "1d", start_date, end_date
                        ),
                        "下载日线 {0}".format(code),
                    )
                available.append(code)
            except Exception as error:
                issues.append(_issue("ERROR", "kline_1d", code, "", str(error)))
        if not available:
            return pd.DataFrame(columns=KLINE_COLUMNS), issues

        fields = list(KLINE_FIELD_MAP.keys())
        try:
            result = self._retry(
                lambda: self.context.get_market_data_ex(
                    fields,
                    available,
                    period="1d",
                    start_time=start_date,
                    end_time=end_date,
                    count=-1,
                    dividend_type="none",
                    fill_data=True,
                    subscribe=False,
                ),
                "读取日线批次",
            )
        except Exception as error:
            for code in available:
                issues.append(_issue("ERROR", "kline_1d", code, "", str(error)))
            return pd.DataFrame(columns=KLINE_COLUMNS), issues

        rows = []
        result = result or {}
        for code in available:
            raw = result.get(code)
            if raw is None or len(raw) == 0:
                # fill_data=True 时，QMT 会为可识别的停牌日返回 suspendFlag=1 的补齐行。
                # 完全没有返回记录的代码留给全批次缺口检查，避免把停牌误报为缓存缺失。
                continue
            frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            for index_value, values in frame.iterrows():
                trade_date = normalize_date(index_value)
                if trade_date is None:
                    time_value = values.get("time") if "time" in values else None
                    trade_date = normalize_date(time_value)
                if trade_date is None or trade_date < start_date or trade_date > end_date:
                    continue
                row = {"code": code, "trade_date": trade_date}
                for source, target in KLINE_FIELD_MAP.items():
                    row[target] = finite_number(values.get(source))
                rows.append(row)
        return pd.DataFrame(rows, columns=KLINE_COLUMNS), issues

    def fetch_trading_dates(self, calendar_symbol, start_date, end_date):
        """从大 QMT 交易日接口读取请求区间的预期交易日。

        参数：
            calendar_symbol: 用作交易日历基准的证券代码，默认可使用上证综指。
            start_date: 八位区间起始日。
            end_date: 八位区间结束日。

        返回：
            ``(dates, issues)``；日期已去重排序，接口异常时返回空列表和 ``ERROR``。
        """
        try:
            count = (
                datetime.strptime(end_date, "%Y%m%d")
                - datetime.strptime(start_date, "%Y%m%d")
            ).days + 1
            raw = self._retry(
                lambda: self.context.get_trading_dates(
                    calendar_symbol, start_date, end_date, max(count, 1), "1d"
                ),
                "读取交易日历 {0}".format(calendar_symbol),
            )
        except Exception as error:
            return [], [
                _issue("ERROR", "trading_calendar", calendar_symbol, "", str(error))
            ]
        dates = sorted(
            set(
                date_value
                for date_value in (normalize_date(value) for value in (raw or []))
                if date_value is not None and start_date <= date_value <= end_date
            )
        )
        return dates, []

    def fetch_finance(self, symbols, start_date, end_date):
        """读取大 QMT 本地缓存中的原始财务记录。

        参数：
            symbols: 当前批次证券代码列表。
            start_date: 八位报告期起始日；为生成历史快照通常应留足回看区间。
            end_date: 八位报告期结束日。

        返回：
            ``(table_frames, issues)``；表格按财务表名分组，记录同时保留报告期和公告日。
        """
        table_frames = {}
        issues = []
        for table_name, fields in FINANCE_FIELDS.items():
            try:
                raw = self._retry(
                    lambda current_fields=fields: self.context.get_raw_financial_data(
                        current_fields,
                        list(symbols),
                        start_date,
                        end_date,
                        report_type="report_time",
                    ),
                    "读取财务表 {0}".format(table_name),
                )
            except Exception as error:
                issues.append(_issue("ERROR", "finance_raw/{0}".format(table_name), "", "", str(error)))
                table_frames[table_name] = pd.DataFrame(columns=_finance_columns(fields))
                continue
            frame, parse_issues = _parse_financial_result(raw or {}, symbols, table_name, fields)
            table_frames[table_name] = frame
            issues.extend(parse_issues)
        return table_frames, issues

    def fetch_corporate_actions(self, symbols, start_date, end_date):
        """读取历史除权除息、送转和配股记录。

        参数：
            symbols: 当前批次证券代码列表。
            start_date: 八位除权日起始日。
            end_date: 八位除权日结束日。

        返回：
            ``(DataFrame, issues)``；数量和金额均按大 QMT 的每股口径原样保存。
        """
        rows = []
        issues = []
        for code in symbols:
            try:
                raw = self._retry(
                    lambda current=code: self.context.get_divid_factors(current),
                    "读取除权记录 {0}".format(code),
                )
            except Exception as error:
                issues.append(_issue("ERROR", "corporate_actions", code, "", str(error)))
                continue
            for timestamp, values in (raw or {}).items():
                ex_date = normalize_date(timestamp)
                if ex_date is None or ex_date < start_date or ex_date > end_date:
                    continue
                if not isinstance(values, (list, tuple)) or len(values) < 7:
                    issues.append(_issue("WARNING", "corporate_actions", code, ex_date, "除权记录字段数量不足 7 个"))
                    continue
                rows.append(
                    {
                        "code": code,
                        "ex_date": ex_date,
                        "cash_dividend_per_share": finite_number(values[0]),
                        "bonus_share_per_share": finite_number(values[1]),
                        "capitalization_per_share": finite_number(values[2]),
                        "rights_issue_per_share": finite_number(values[3]),
                        "rights_issue_price": finite_number(values[4]),
                        "share_reform_flag": finite_number(values[5]),
                        "adjustment_factor": finite_number(values[6]),
                    }
                )
        return pd.DataFrame(rows, columns=CORPORATE_ACTION_COLUMNS), issues

    def _retry(self, action, description):
        """按配置重试一个同步大 QMT 调用。

        参数：
            action: 不接收参数、执行一次大 QMT 调用的可调用对象。
            description: 写入日志的简短操作说明。

        返回：
            最后一次成功调用的返回值；全部失败时重新抛出最终异常。
        """
        last_error = None
        for attempt in range(1, self.retry_count + 1):
            try:
                return action()
            except Exception as error:
                last_error = error
                self.logger.warning(
                    "%s 失败，尝试 %d/%d: %s",
                    description,
                    attempt,
                    self.retry_count,
                    error,
                )
                if attempt < self.retry_count:
                    time.sleep(min(attempt, 3))
        raise last_error


def _parse_financial_result(raw, symbols, table_name, fields):
    """把嵌套财务字典转换为按报告期排列的数据表。

    参数：
        raw: ``get_raw_financial_data`` 返回的嵌套字典。
        symbols: 请求的证券代码序列，用于识别整只证券缺失。
        table_name: 当前逻辑财务表名。
        fields: 当前请求的完整大 QMT 字段列表。

    返回：
        ``(DataFrame, issues)``，字段名去掉表名前缀并附加证券、报告期和公告日。
    """
    rows = []
    issues = []
    report_field = fields[0]
    announce_field = fields[1]
    for code in symbols:
        values_by_field = (raw or {}).get(code) or {}
        keys = set()
        for field in fields:
            field_values = values_by_field.get(field) or {}
            if isinstance(field_values, dict):
                keys.update(field_values.keys())
        if not keys:
            issues.append(
                _issue(
                    "WARNING",
                    "finance_raw/{0}".format(table_name),
                    code,
                    "",
                    "未返回财务记录；请先在大 QMT 数据管理中下载财务数据",
                )
            )
            continue
        for key in sorted(keys, key=lambda item: str(item)):
            report_raw = _field_value(values_by_field, report_field, key)
            announce_raw = _field_value(values_by_field, announce_field, key)
            report_date = normalize_date(report_raw) or normalize_date(key)
            announce_date = normalize_date(announce_raw)
            if report_date is None:
                issues.append(_issue("WARNING", "finance_raw/{0}".format(table_name), code, "", "财务记录缺少可识别报告期"))
                continue
            if announce_date is None:
                issues.append(_issue("WARNING", "finance_raw/{0}".format(table_name), code, report_date, "财务记录缺少公告日，将保存到 unknown 分区且不进入日快照"))
            row = {
                "code": code,
                "report_date": report_date,
                "announce_date": announce_date,
            }
            for field in fields[2:]:
                column = field.split(".", 1)[1]
                row[column] = finite_number(
                    _field_value(values_by_field, field, key)
                )
            missing_columns = [
                field.split(".", 1)[1]
                for field in fields[2:]
                if row.get(field.split(".", 1)[1]) is None
            ]
            if missing_columns:
                if len(missing_columns) == len(fields[2:]):
                    issue_dataset = "finance_raw/{0}".format(table_name)
                    message = "该财务记录所有业务字段均缺失: {0}".format(
                        ",".join(missing_columns)
                    )
                else:
                    issue_dataset = "finance_field/{0}".format(table_name)
                    message = "该财务记录部分业务字段缺失: {0}".format(
                        ",".join(missing_columns)
                    )
                issues.append(
                    _issue(
                        "WARNING",
                        issue_dataset,
                        code,
                        report_date,
                        message,
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows, columns=_finance_columns(fields)), issues


def _finance_columns(fields):
    """生成一个财务表的规范输出列。

    参数：
        fields: 当前财务表请求的大 QMT 完整字段列表，前两项为报告期和公告日。

    返回：
        证券代码、报告期、公告日和业务字段组成的列名列表。
    """
    return ["code", "report_date", "announce_date"] + [
        field.split(".", 1)[1] for field in fields[2:]
    ]


def _field_value(values_by_field, field, key):
    """安全读取某一财务字段在指定报告键上的值。

    参数：
        values_by_field: 单只证券按完整字段名分组的字典。
        field: 大 QMT 完整财务字段名。
        key: 当前财务记录的报告键。

    返回：
        原始字段值；字段或键不存在时返回 ``None``。
    """
    values = values_by_field.get(field) or {}
    return values.get(key) if isinstance(values, dict) else None


def _issue(level, dataset, code, date_value, message):
    """构造统一的问题记录。

    参数：
        level: ``WARNING`` 或 ``ERROR`` 严重级别。
        dataset: 问题所属数据集。
        code: 相关证券代码。
        date_value: 相关八位日期；未知时为空。
        message: 面向用户的错误或缺失说明。

    返回：
        可直接写入问题报告的字典。
    """
    return {
        "level": level,
        "dataset": dataset,
        "code": code,
        "date": date_value,
        "message": message,
    }
