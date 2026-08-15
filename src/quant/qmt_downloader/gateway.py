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

INSTRUMENT_INFO_COLUMNS = [
    "code",
    "instrument_name",
    "open_date",
    "expire_date",
    "is_trading",
    "instrument_status",
]

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

    def __init__(
        self,
        context,
        history_downloader,
        logger,
        retry_count=3,
        batch_history_downloader=None,
    ):
        """保存大 QMT 上下文及历史行情下载函数。

        参数：
            context: 大 QMT 策略传入的 ``ContextInfo`` 对象。
            history_downloader: 大 QMT 内置全局 ``download_history_data`` 函数。
            logger: 已配置终端和文件处理器的日志对象。
            retry_count: 单只证券接口失败后的最大尝试次数。
            batch_history_downloader: 可选的批量历史下载函数，签名为
                ``(codes, period, start_time, end_time)``（大 QMT 的
                ``download_history_data2``）。为 ``None`` 时始终逐只下载。
        """
        self.context = context
        self.history_downloader = history_downloader
        self.logger = logger
        self.retry_count = int(retry_count)
        self.batch_history_downloader = batch_history_downloader

    def resolve_symbols(self, symbols, sector, expired_sectors=()):
        """从显式代码列表、大 QMT 板块和过期（退市）板块解析证券池。

        参数：
            symbols: 配置中显式填写的证券代码序列；非空时优先于 ``sector``。
            sector: 大 QMT 客户端中的板块名称，仅在 ``symbols`` 为空时使用；两者
                都为空时只按 ``expired_sectors`` 组池，不会拿空板块名去查接口。
            expired_sectors: 过期（退市）板块名称序列，例如 ``过期沪深A股``。这是
                附加项：无论存续证券来自 ``symbols`` 还是 ``sector`` 都会并入，用于
                消除回测的幸存者偏差。留空时行为与不带该参数时完全一致。

        返回：
            去重、排序并转为大写的证券代码列表。以下情况抛 ``RuntimeError``：
            ``sector`` 没有成分证券、过期板块无成分且不在客户端板块列表中、配置的
            过期板块合计没带回任何证券、最终证券池为空；单个过期板块存在但无成分只
            记 ``WARNING``。代码缺少市场后缀时抛 ``ValueError``。
        """
        if symbols:
            living = _normalize_codes(symbols)
        elif sector:
            living = _normalize_codes(self.context.get_stock_list_in_sector(sector) or [])
            if not living:
                # 单独判定存续板块：过期板块会让整体证券池非空，板块名写错就再也碰不到
                # 下面的空池判定，最终静默跑出一份只含退市标的的数据集。
                raise RuntimeError(
                    "板块 {0} 没有返回任何证券，请检查大 QMT 板块名称".format(sector)
                )
        else:
            # 只配过期板块时没有存续板块可查；空板块名对大 QMT 没有意义，某些版本会直接抛错。
            living = set()
        # 先物化并去空白：``expired_sectors`` 是公开参数，传入生成器时逐处判定会二次
        # 消费，错误消息里的板块名会变成空串；空白板块名同样不能拿去查接口。
        expired_names = [
            str(item).strip() for item in expired_sectors if str(item).strip()
        ]
        expired = set()
        known_sectors = self._sector_names() if expired_names else None
        for name in expired_names:
            codes = _normalize_codes(self.context.get_stock_list_in_sector(name) or [])
            if codes:
                self.logger.info("过期板块解析完成 sector=%s codes=%d", name, len(codes))
                expired |= codes
                continue
            if known_sectors is not None and name in known_sectors:
                # 板块存在但确实没有成分，例如该市场尚无退市标的。这不是配置错误，
                # 阻断整次运行只会逼用户删掉一个本来正确的板块名。
                self.logger.warning(
                    "过期板块 %s 存在但没有成分证券，本次不并入该板块的退市标的", name
                )
                continue
            # 板块名写错或客户端未下载过期合约列表。静默放行会让证券池悄悄退回只含
            # 存续标的，回测重新带上幸存者偏差且毫无提示，因此直接失败。
            raise RuntimeError(
                "过期板块 {0} 没有返回任何证券，也不在客户端板块列表中；请在大 QMT "
                "界面端“数据管理 → 过期合约数据 → 过期合约列表”下载并重启客户端，"
                "再用 get_sector_list() 核对板块名".format(name)
            )
        if expired_names and not expired:
            # 单个板块合法为空可以放行，但配置的过期板块一个退市标的都没带回来时，
            # 证券池实际退回了只含存续标的的状态，幸存者偏差原封不动地回来了。
            raise RuntimeError(
                "配置的过期板块（{0}）全部没有成分证券，证券池只剩存续标的；"
                "请确认已在大 QMT 界面端“数据管理 → 过期合约数据 → 过期合约列表”"
                "下载并重启客户端".format("、".join(expired_names))
            )
        output = sorted(living | expired)
        if not output:
            raise RuntimeError("证券池为空，请检查 symbols 或大 QMT 板块名称")
        invalid = [code for code in output if "." not in code]
        if invalid:
            raise ValueError("证券代码必须包含市场后缀: {0}".format(",".join(invalid)))
        if expired_names:
            self.logger.info(
                "证券池解析完成 存续=%d 过期=%d 合计=%d",
                len(living),
                len(expired - living),
                len(output),
            )
        return output

    def _sector_names(self):
        """读取客户端板块名集合，用于区分“板块不存在”和“板块存在但为空”。

        返回：
            去空白后的板块名集合；接口缺失、抛错或返回空时返回 ``None``，调用方
            据此退回“板块为空即失败”的严格判定，不会因为这一步不可用而放过缩池。
        """
        getter = getattr(self.context, "get_sector_list", None)
        if getter is None:
            return None
        try:
            names = getter() or []
        except Exception as error:
            self.logger.warning(
                "板块列表读取异常 stage=get_sector_list error_type=%s error=%s",
                type(error).__name__,
                error,
            )
            return None
        output = set(str(name).strip() for name in names if str(name).strip())
        return output or None

    def _download_kline_history(self, symbols, start_date, end_date):
        """补充本地日线缓存，可用时优先走批量下载接口。

        批量接口把整批证券压缩成一次请求，省掉逐只调用的往返开销；一旦批量调用
        失败，就退回逐只下载，把失败归因到具体证券，行为与改造前一致，并对本次
        运行的其余批次直接禁用批量接口。批量调用成功但个别证券实际未补齐时，后
        续的缺口检查和定向补下载仍会兜住。

        参数：
            symbols: 当前批次证券代码列表。
            start_date: 八位起始交易日。
            end_date: 八位结束交易日。

        返回：
            ``(available, issues)``；``available`` 为可以继续读取行情的代码列表。
        """
        codes = list(symbols)
        if self.batch_history_downloader is not None and len(codes) > 1:
            started_at = time.perf_counter()
            self.logger.info(
                "日线下载开始 stage=download_history mode=batch symbols=%d start=%s end=%s",
                len(codes),
                start_date,
                end_date,
            )
            try:
                # 逐只下载随时可以顶上，因此批量调用不重试：重试只会在每个批次前
                # 白等数秒，而连接类故障不会在同一次运行内自愈。
                self.batch_history_downloader(codes, "1d", start_date, end_date)
            except Exception as error:
                self.logger.warning(
                    "批量下载日线失败，本次运行改用逐只下载 stage=download_history_data2 symbols=%d "
                    "start=%s end=%s elapsed=%.2fs error_type=%s error=%s",
                    len(codes),
                    start_date,
                    end_date,
                    time.perf_counter() - started_at,
                    type(error).__name__,
                    error,
                )
                # 首次失败后禁用，避免其余批次重复付出同样的失败等待。
                self.batch_history_downloader = None
            else:
                self.logger.info(
                    "日线下载完成 stage=download_history mode=batch symbols=%d elapsed=%.2fs",
                    len(codes),
                    time.perf_counter() - started_at,
                )
                return codes, []

        started_at = time.perf_counter()
        self.logger.info(
            "日线下载开始 stage=download_history mode=per_symbol symbols=%d start=%s end=%s",
            len(codes),
            start_date,
            end_date,
        )
        available = []
        issues = []
        for code in codes:
            try:
                self._retry(
                    lambda current=code: self.history_downloader(
                        current, "1d", start_date, end_date
                    ),
                    "下载日线 {0}".format(code),
                )
                available.append(code)
            except Exception as error:
                self.logger.exception(
                    "日线下载异常 stage=download_history_data code=%s period=1d start=%s end=%s error_type=%s error=%s",
                    code,
                    start_date,
                    end_date,
                    type(error).__name__,
                    error,
                )
                issues.append(
                    _issue(
                        "ERROR",
                        "kline_1d",
                        code,
                        "",
                        "stage=download_history_data period=1d start={0} end={1} error_type={2} error={3}".format(
                            start_date, end_date, type(error).__name__, error
                        ),
                    )
                )
        self.logger.info(
            "日线下载完成 stage=download_history mode=per_symbol symbols=%d available=%d elapsed=%.2fs",
            len(codes),
            len(available),
            time.perf_counter() - started_at,
        )
        return available, issues

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
        if download_first:
            available, issues = self._download_kline_history(symbols, start_date, end_date)
        else:
            available, issues = list(symbols), []
        if not available:
            return pd.DataFrame(columns=KLINE_COLUMNS), issues

        fields = list(KLINE_FIELD_MAP.keys())
        market_started_at = time.perf_counter()
        self.logger.info(
            "日线批量读取开始 stage=get_market_data_ex symbols=%d start=%s end=%s fill_data=True",
            len(available),
            start_date,
            end_date,
        )
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
            result_rows = 0
            unsized_results = 0
            if isinstance(result, dict):
                for value in result.values():
                    if value is None:
                        continue
                    try:
                        result_rows += len(value)
                    except TypeError:
                        # 非标准 QMT 返回值不应影响已成功的行情读取结果。
                        unsized_results += 1
            if unsized_results:
                self.logger.warning(
                    "日线批量读取存在 %d 个无法计数的非常规返回值，rows 统计不含这部分",
                    unsized_results,
                )
            self.logger.info(
                "日线批量读取完成 stage=get_market_data_ex symbols=%d rows=%d elapsed=%.2fs",
                len(available),
                result_rows,
                time.perf_counter() - market_started_at,
            )
        except Exception as error:
            self.logger.exception(
                "日线读取异常 stage=get_market_data_ex symbols=%s period=1d start=%s end=%s fill_data=True subscribe=False elapsed=%.2fs error_type=%s error=%s",
                ",".join(available),
                start_date,
                end_date,
                time.perf_counter() - market_started_at,
                type(error).__name__,
                error,
            )
            for code in available:
                issues.append(
                    _issue(
                        "ERROR",
                        "kline_1d",
                        code,
                        "",
                        "stage=get_market_data_ex symbols={0} period=1d start={1} end={2} fill_data=True subscribe=False error_type={3} error={4}".format(
                            ",".join(available), start_date, end_date, type(error).__name__, error
                        ),
                    )
                )
            return pd.DataFrame(columns=KLINE_COLUMNS), issues

        parse_started_at = time.perf_counter()
        parts = []
        date_cache = {}
        result = result or {}
        for code in available:
            raw = result.get(code)
            if raw is None or len(raw) == 0:
                # fill_data=True 时，QMT 会为可识别的停牌日返回 suspendFlag=1 的补齐行。
                # 完全没有返回记录的代码留给全批次缺口检查，避免把停牌误报为缓存缺失。
                continue
            frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            trade_dates = [_cached_normalize_date(value, date_cache) for value in frame.index]
            if "time" in frame.columns and any(value is None for value in trade_dates):
                fallback = list(frame["time"])
                trade_dates = [
                    value
                    if value is not None
                    else _cached_normalize_date(fallback[position], date_cache)
                    for position, value in enumerate(trade_dates)
                ]
            keep = [
                position
                for position, value in enumerate(trade_dates)
                if value is not None and start_date <= value <= end_date
            ]
            if not keep:
                continue
            selected = frame.iloc[keep]
            columns = {
                "code": [code] * len(keep),
                "trade_date": [trade_dates[position] for position in keep],
            }
            for source, target in KLINE_FIELD_MAP.items():
                if source in selected.columns:
                    columns[target] = _finite_values(selected[source])
                else:
                    columns[target] = [float("nan")] * len(keep)
            parts.append(pd.DataFrame(columns, columns=KLINE_COLUMNS))
        if parts:
            output = pd.concat(parts, ignore_index=True)
        else:
            output = pd.DataFrame(columns=KLINE_COLUMNS)
        returned_start = output["trade_date"].min() if not output.empty else ""
        returned_end = output["trade_date"].max() if not output.empty else ""
        self.logger.info(
            "日线数据解析完成 stage=parse_kline requested_start=%s requested_end=%s returned_start=%s returned_end=%s symbols=%d returned_symbols=%d rows=%d elapsed=%.2fs issues=%d",
            start_date,
            end_date,
            returned_start,
            returned_end,
            len(available),
            output["code"].nunique() if not output.empty else 0,
            len(output),
            time.perf_counter() - parse_started_at,
            len(issues),
        )
        return output, issues

    def fetch_trading_dates(self, calendar_symbol, start_date, end_date):
        """从大 QMT 交易日接口读取请求区间的预期交易日。

        参数：
            calendar_symbol: 用作交易日历基准的证券代码，默认可使用上证综指。
            start_date: 八位区间起始日。
            end_date: 八位区间结束日。

        返回：
            ``(dates, issues)``；日期已去重排序，接口异常时返回空列表和 ``ERROR``。
        """
        count = 0
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
            self.logger.exception(
                "交易日历读取异常 stage=get_trading_dates calendar_symbol=%s start=%s end=%s count=%s period=1d error_type=%s error=%s",
                calendar_symbol,
                start_date,
                end_date,
                max(count, 1),
                type(error).__name__,
                error,
            )
            return [], [
                _issue(
                    "ERROR",
                    "trading_calendar",
                    calendar_symbol,
                    "",
                    "stage=get_trading_dates start={0} end={1} count={2} period=1d error_type={3} error={4}".format(
                        start_date, end_date, max(count, 1), type(error).__name__, error
                    ),
                )
            ]
        dates = sorted(
            set(
                date_value
                for date_value in (normalize_date(value) for value in (raw or []))
                if date_value is not None and start_date <= date_value <= end_date
            )
        )
        return dates, []

    def fetch_instrument_info(self, symbols):
        """逐只读取证券上市、退市和当前交易状态信息。

        参数：
            symbols: 需要查询的证券代码序列；每个代码必须包含市场后缀。

        返回：
            ``(DataFrame, issues)``；数据包含代码、名称、上市日期、退市日期、当前交易
            状态和停牌状态，接口失败的证券写入错误列表。
        """
        rows = []
        issues = []
        getter = getattr(self.context, "get_instrument_detail", None)
        if getter is None:
            getter = getattr(self.context, "get_instrumentdetail", None)
        if getter is None:
            issue = _issue(
                "ERROR",
                "instrument_info",
                "",
                "",
                "大 QMT ContextInfo 不支持 get_instrument_detail/get_instrumentdetail",
            )
            return pd.DataFrame(columns=INSTRUMENT_INFO_COLUMNS), [issue]
        for code in symbols:
            try:
                detail = getter(code) or {}
                if not isinstance(detail, dict) or not detail:
                    raise ValueError("未返回合约基础信息")
                rows.append(
                    {
                        "code": code,
                        "instrument_name": detail.get("InstrumentName"),
                        "open_date": _normalize_lifecycle_date(detail.get("OpenDate")),
                        "expire_date": _normalize_lifecycle_date(detail.get("ExpireDate")),
                        "is_trading": detail.get("IsTrading"),
                        "instrument_status": detail.get("InstrumentStatus"),
                    }
                )
            except Exception as error:
                self.logger.exception(
                    "证券生命周期读取异常 stage=get_instrument_detail code=%s error_type=%s error=%s",
                    code,
                    type(error).__name__,
                    error,
                )
                issues.append(
                    _issue(
                        "ERROR",
                        "instrument_info",
                        code,
                        "",
                        "stage=get_instrument_detail code={0} error_type={1} error={2}".format(
                            code, type(error).__name__, error
                        ),
                    )
                )
        return pd.DataFrame(rows, columns=INSTRUMENT_INFO_COLUMNS), issues

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
                self.logger.exception(
                    "财务读取异常 stage=get_raw_financial_data table=%s symbols=%s start=%s end=%s report_type=report_time error_type=%s error=%s",
                    table_name,
                    ",".join(symbols),
                    start_date,
                    end_date,
                    type(error).__name__,
                    error,
                )
                issues.append(
                    _issue(
                        "ERROR",
                        "finance_raw/{0}".format(table_name),
                        "",
                        "",
                        "stage=get_raw_financial_data table={0} symbols={1} start={2} end={3} report_type=report_time error_type={4} error={5}".format(
                            table_name,
                            ",".join(symbols),
                            start_date,
                            end_date,
                            type(error).__name__,
                            error,
                        ),
                    )
                )
                table_frames[table_name] = pd.DataFrame(columns=_finance_columns(fields))
                continue
            try:
                frame, parse_issues = _parse_financial_result(raw or {}, symbols, table_name, fields)
            except Exception as error:
                self.logger.exception(
                    "财务解析异常 stage=parse_financial_result table=%s symbols=%s start=%s end=%s error_type=%s error=%s",
                    table_name,
                    ",".join(symbols),
                    start_date,
                    end_date,
                    type(error).__name__,
                    error,
                )
                issues.append(
                    _issue(
                        "ERROR",
                        "finance_raw/{0}".format(table_name),
                        "",
                        "",
                        "stage=parse_financial_result table={0} symbols={1} start={2} end={3} error_type={4} error={5}".format(
                            table_name,
                            ",".join(symbols),
                            start_date,
                            end_date,
                            type(error).__name__,
                            error,
                        ),
                    )
                )
                table_frames[table_name] = pd.DataFrame(columns=_finance_columns(fields))
                continue
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
                self.logger.exception(
                    "除权记录读取异常 stage=get_divid_factors code=%s start=%s end=%s error_type=%s error=%s",
                    code,
                    start_date,
                    end_date,
                    type(error).__name__,
                    error,
                )
                issues.append(
                    _issue(
                        "ERROR",
                        "corporate_actions",
                        code,
                        "",
                        "stage=get_divid_factors code={0} start={1} end={2} error_type={3} error={4}".format(
                            code, start_date, end_date, type(error).__name__, error
                        ),
                    )
                )
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


def _normalize_codes(values):
    """把一份原始证券代码序列规整为去重的大写代码集合。

    参数：
        values: 显式配置或板块接口返回的代码序列，允许含空白项。

    返回：
        去除首尾空白并转为大写的代码集合；空白项被丢弃。
    """
    return {
        str(code).strip().upper() for code in values if str(code).strip()
    }


def _cached_normalize_date(value, cache):
    """带缓存的 ``normalize_date``；一个批次内各证券的日期索引高度重复。

    参数：
        value: QMT 日期索引值。
        cache: 调用方持有的缓存字典，生命周期应限定在单个批次内。

    返回：
        与 ``normalize_date`` 完全一致的结果；不可哈希的值直接绕过缓存。
    """
    try:
        if value in cache:
            return cache[value]
    except TypeError:
        return normalize_date(value)
    normalized = normalize_date(value)
    cache[value] = normalized
    return normalized


def _finite_values(values):
    """按列向量化实现 ``finite_number``。

    参数：
        values: 单只证券某个行情字段的 ``Series``。

    返回：
        ``numpy`` 浮点数组；非数值、无穷和 QMT 哨兵值统一为 ``NaN``，与
        ``finite_number`` 返回 ``None`` 在数据表中等价。
    """
    numbers = pd.to_numeric(values, errors="coerce").astype("float64")
    return numbers.where(numbers.abs() <= 1e100).values


def _normalize_lifecycle_date(value):
    """标准化上市或退市日期并屏蔽 QMT 的无日期哨兵值。

    参数：
        value: ``OpenDate`` 或 ``ExpireDate`` 返回的日期、整数或字符串。

    返回：
        ``YYYYMMDD`` 日期；``0``、``99999999`` 等无明确日期的值返回 ``None``。
    """
    text = str(value).strip() if value is not None else ""
    if text in (
        "",
        "0",
        "0.0",
        "99999999",
        "99999999.0",
        "19700101",
        "19700102",
        "19700103",
        "19700104",
        "19700105",
        "19700106",
    ):
        return None
    return normalize_date(value)


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
