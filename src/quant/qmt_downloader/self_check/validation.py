# -*- coding: utf-8 -*-
"""行级字段校验、数值异常检测与日线连续性校验。"""

from __future__ import annotations

import math
import statistics
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

from .base import _CheckerState
from .models import AuditIssue, _ScanState, _SymbolStats
from .utils import _finite_float, _normalize_date_text


class _RowValidationMixin(_CheckerState):
    """校验行级主键与数值字段，并检测缺失区间与统计异常。"""

    def _validate_rows(
        self,
        frame: pd.DataFrame,
        date_value: str,
        directory: Path | None,
        expected_symbols: set[str],
        lifecycle: dict[str, tuple[str, str | None]],
        corporate_actions: set[tuple[str, str]],
        stats: dict[str, _SymbolStats],
        state: _ScanState,
    ) -> set[str]:
        """验证单日全部行情行及跨日价格、成交量连续性。

        参数：
            frame: 当前交易日日线表。
            date_value: 当前八位交易日期。
            directory: 当前分区目录；整日缺失时为 ``None``。
            expected_symbols: 分区口径声明的完整证券池集合。
            lifecycle: 证券代码到上市退市日期的映射。
            corporate_actions: 可解释价格断层的公司行为主键集合。
            stats: 按证券累计覆盖率和异常数量的可变统计字典。
            state: 跨日期保留上一行、缺口和成交量历史的状态。

        返回：
            当前分区实际出现且主键可解析的证券代码集合。
        """

        if frame.empty:
            return set()
        required = {
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
        }
        missing_columns = sorted(required.difference(frame.columns))
        path = (
            directory
            if directory is not None and self.source_mode == "staging"
            else (directory / "data.csv") if directory is not None else Path("")
        )
        if missing_columns:
            self._add_issue(
                "KLINE_COLUMNS_MISSING",
                "ERROR",
                "kline_1d",
                "日线分区缺少必需字段。",
                date=date_value,
                expected=", ".join(sorted(required)),
                actual="缺少: {0}".format(", ".join(missing_columns)),
                possible_causes="下载器版本不一致、CSV 表头损坏或人工编辑",
                suggested_action="确认字段版本并使用 repair 重建分区",
                source_file=str(path),
            )
            for column in missing_columns:
                frame[column] = math.nan
        normalized_codes = frame["code"].astype(str).str.strip().str.upper()
        duplicated = frame.assign(_code=normalized_codes).duplicated(
            ["_code", "trade_date"], keep=False
        )
        for index in frame.index[duplicated]:
            code = normalized_codes.loc[index]
            duplicate_path = path
            duplicate_index: Any = index
            if self.source_mode == "staging":
                source_value = frame.loc[index].get("_source_file")
                if isinstance(source_value, str) and source_value:
                    duplicate_path = Path(source_value)
                try:
                    duplicate_index = int(frame.loc[index].get("_source_row")) - 2
                except (TypeError, ValueError):
                    pass
            self._row_issue(
                "KLINE_DUPLICATE_KEY",
                "ERROR",
                "同一证券和交易日出现重复日线主键。",
                code,
                date_value,
                "code,trade_date",
                "每个主键恰好一行",
                "存在重复记录",
                "批次合并重复、文件被追加或修复时未覆盖旧记录",
                "重建该日期分区，不要任意保留重复行",
                duplicate_path,
                duplicate_index,
            )
        present: set[str] = set()
        # 逐行取值是整轮自检的绝对热点：一次全量审计要处理数千万行，而 iterrows 每行
        # 都要新建一个 Series，行内十余次 row.get 又都走 Series 的索引查找。一次性转成
        # 纯 dict 记录后按位置遍历，取值退化为字典查找，字段语义和遍历顺序保持不变。
        labels = frame.index.tolist()
        codes = normalized_codes.tolist()
        records = frame.to_dict("records")
        # 同一分区内 trade_date 几乎全部相同，而日期规范化每次都要走 strptime，
        # 按原始值缓存可以把整日的解析次数压到不同取值的个数。
        date_cache: dict[Any, str | None] = {}
        for position, row in enumerate(records):
            index = labels[position]
            code = codes[position]
            # 来源路径必须逐行从分区级默认值重新算起：一旦将来出现某些行带
            # ``_source_file``、某些行没有的混合帧，沿用上一行留下的值会把这些行的
            # 问题指向一个与它无关的批次文件，排查时直接跑偏。
            row_path = path
            source_path = row.get("_source_file")
            if isinstance(source_path, str) and source_path:
                row_path = Path(source_path)
                try:
                    index = int(row.get("_source_row")) - 2
                except (TypeError, ValueError):
                    pass
            raw_date = row.get("trade_date")
            try:
                row_date = date_cache[raw_date]
            except (KeyError, TypeError):
                row_date = _normalize_date_text(raw_date)
                try:
                    date_cache[raw_date] = row_date
                except TypeError:
                    pass
            if not code or code.lower() == "nan":
                self._row_issue(
                    "KLINE_CODE_MISSING",
                    "ERROR",
                    "日线记录缺少证券代码。",
                    "",
                    date_value,
                    "code",
                    "非空的 code.market",
                    str(row.get("code", "")),
                    "CSV 字段错位或写入前代码丢失",
                    "重新下载该日期分区",
                    row_path,
                    index,
                )
                continue
            if code not in expected_symbols:
                self._row_issue(
                    "KLINE_SYMBOL_OUTSIDE_SCOPE",
                    "ERROR",
                    "日线记录的证券不属于分区声明的证券池。",
                    code,
                    date_value,
                    "code",
                    "code 属于 partition_scope.symbols",
                    code,
                    "证券池切换后混写分区或代码映射错误",
                    "核对证券池并使用统一口径重建分区",
                    row_path,
                    index,
                )
                continue
            if row_date != date_value:
                self._row_issue(
                    "KLINE_DATE_PARTITION_MISMATCH",
                    "ERROR",
                    "日线记录的 trade_date 与所在日期分区不一致。",
                    code,
                    date_value,
                    "trade_date",
                    date_value,
                    str(row.get("trade_date", "")),
                    "分区合并错误、日期标准化错误或文件被移动",
                    "按记录真实日期核对源数据后重建相关分区",
                    row_path,
                    index,
                )
                continue
            present.add(code)
            open_date, expire_date = lifecycle.get(code, (date_value, None))
            in_lifecycle = date_value >= open_date and (
                expire_date is None or date_value <= expire_date
            )
            # QMT 会在部分证券上市前用 suspend_flag=1 的停牌占位行填充历史日期，
            # 这类行不是真实行情归属错误，不计入 DATA_BEFORE_LISTING。
            suspend_flag = _finite_float(row.get("suspend_flag"))
            if (
                date_value < open_date
                and suspend_flag != 1.0
                and (code, "before") not in self._lifecycle_issue_keys
            ):
                self._lifecycle_issue_keys.add((code, "before"))
                self._row_issue(
                    "DATA_BEFORE_LISTING",
                    "ERROR",
                    "证券上市前出现日线行情。",
                    code,
                    date_value,
                    "trade_date",
                    "trade_date >= {0}".format(open_date),
                    date_value,
                    "上市日期错误、证券代码复用或行情归属错误",
                    "核对 QMT OpenDate 和证券代码，确认前不要自动删除行情",
                    row_path,
                    index,
                )
            if expire_date is not None and date_value > expire_date and (code, "after") not in self._lifecycle_issue_keys:
                self._lifecycle_issue_keys.add((code, "after"))
                self._row_issue(
                    "DATA_AFTER_DELISTING",
                    "ERROR",
                    "证券退市后仍存在日线行情。",
                    code,
                    date_value,
                    "trade_date",
                    "trade_date <= {0}".format(expire_date),
                    date_value,
                    "退市日期口径错误、证券代码复用或历史数据混入",
                    "核对 QMT ExpireDate 和证券代码，确认前不要自动删除行情",
                    row_path,
                    index,
                )
            if not in_lifecycle:
                # 上市前/退市后的 QMT 占位行只用于生命周期审计；其零价格和零成交量
                # 不应再次被当作行情字段损坏，从而避免全样本报告产生百万级重复错误。
                continue
            invalid = self._validate_numeric_row(
                row, code, date_value, row_path, index, stats
            )
            if invalid:
                stats[code].invalid_rows += 1
            self._validate_continuity(
                row,
                code,
                date_value,
                row_path,
                index,
                corporate_actions,
                state,
            )
        return present

    def _validate_numeric_row(
        self,
        row: dict[str, Any],
        code: str,
        date_value: str,
        path: Path,
        index: Any,
        stats: dict[str, _SymbolStats],
    ) -> bool:
        """验证一行 OHLC、成交量、成交额和停牌标志。

        参数：
            row: 当前证券日线原始字段，为该行的列名到原始值映射。
            code: 当前证券代码。
            date_value: 当前交易日期。
            path: 当前记录来源 CSV 路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。
            stats: 按证券累计停牌和零成交异常的可变统计字典。

        返回：
            任一确定性数值校验失败时返回 ``True``。
        """

        values = {name: _finite_float(row.get(name)) for name in (
            "open", "high", "low", "close", "pre_close", "volume", "amount", "suspend_flag"
        )}
        invalid = False
        for field_name in ("open", "high", "low", "close", "pre_close", "volume", "amount", "suspend_flag"):
            if values[field_name] is None:
                invalid = True
                self._row_issue(
                    "KLINE_FIELD_INVALID",
                    "ERROR",
                    "日线必需数值字段缺失、无穷或无法转换为有限数。",
                    code,
                    date_value,
                    field_name,
                    "有限数值",
                    str(row.get(field_name, "")),
                    "QMT 哨兵值、CSV 字段错位、空值或非数字文本",
                    "核对 QMT 原始返回并重新下载该证券日期",
                    path,
                    index,
                )
        if any(values[name] is None for name in values):
            return True
        open_price = values["open"]
        high = values["high"]
        low = values["low"]
        close = values["close"]
        pre_close = values["pre_close"]
        volume = values["volume"]
        amount = values["amount"]
        suspend = values["suspend_flag"]
        assert None not in (open_price, high, low, close, pre_close, volume, amount, suspend)
        # NON_POSITIVE_PRICE 检查已按维护者要求禁用：pre_close 字段本身可能存在
        # 误差（已验证），且下游统一改用 close 计算涨跌幅，不再依赖该检查。
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            invalid = True
            self._row_issue(
                "INVALID_OHLC_RELATION",
                "ERROR",
                "日线最高价或最低价与开收盘价关系不合法。",
                code,
                date_value,
                "open,high,low,close",
                "high >= max(open, low, close) 且 low <= min(open, high, close)",
                "open={0}, high={1}, low={2}, close={3}".format(open_price, high, low, close),
                "字段错位、部分字段来自不同记录或源行情损坏",
                "核对 QMT 原始记录并重建该日期分区",
                path,
                index,
            )
        if volume < 0 or amount < 0:
            invalid = True
            self._row_issue(
                "NEGATIVE_TURNOVER",
                "ERROR",
                "成交量或成交额为负数。",
                code,
                date_value,
                "volume,amount",
                "volume >= 0 且 amount >= 0",
                "volume={0}, amount={1}".format(volume, amount),
                "数值溢出、字段转换错误或行情记录损坏",
                "核对 QMT 原始记录并重建该日期分区",
                path,
                index,
            )
        if suspend not in (0.0, 1.0):
            invalid = True
            self._row_issue(
                "INVALID_SUSPEND_FLAG",
                "ERROR",
                "停牌标志不是 0 或 1。",
                code,
                date_value,
                "suspend_flag",
                "0=未停牌，1=停牌",
                str(suspend),
                "QMT 字段映射错误、CSV 错位或未知接口版本",
                "核对 suspendFlag 原始值及下载器字段映射",
                path,
                index,
            )
        elif suspend == 0.0:
            if volume <= 0:
                invalid = True
                stats[code].active_zero_volume_rows += 1
                self._row_issue(
                    "ACTIVE_ZERO_VOLUME",
                    "ERROR",
                    "QMT 标记为未停牌，但成交量不大于零。",
                    code,
                    date_value,
                    "volume",
                    "suspend_flag=0 时 volume > 0",
                    "suspend_flag=0, volume={0}, amount={1}".format(volume, amount),
                    "停牌标志错误、行情补齐异常或本地行情未完整下载",
                    "核对 QMT 原始 suspendFlag 与 volume，并重新下载该证券日期",
                    path,
                    index,
                )
            if amount <= 0:
                invalid = True
                self._row_issue(
                    "ACTIVE_ZERO_AMOUNT",
                    "ERROR",
                    "QMT 标记为未停牌，但成交额不大于零。",
                    code,
                    date_value,
                    "amount",
                    "suspend_flag=0 时 amount > 0",
                    "suspend_flag=0, volume={0}, amount={1}".format(volume, amount),
                    "成交额字段缺失、停牌标志错误或行情补齐异常",
                    "核对 QMT 原始 amount，并重新下载该证券日期",
                    path,
                    index,
                )
        else:
            stats[code].suspended_rows += 1
            if volume != 0 or amount != 0:
                invalid = True
                self._row_issue(
                    "SUSPENDED_WITH_TURNOVER",
                    "ERROR",
                    "QMT 标记为停牌，但记录仍包含成交量或成交额。",
                    code,
                    date_value,
                    "suspend_flag,volume,amount",
                    "suspend_flag=1 时 volume=0 且 amount=0",
                    "suspend_flag=1, volume={0}, amount={1}".format(volume, amount),
                    "停牌字段错位、批次拼接错误或 QMT 返回口径异常",
                    "查看 QMT 原始返回；确认前不要自动修改停牌标志",
                    path,
                    index,
                )
            if not (open_price == high == low == close):
                self._row_issue(
                    "SUSPENDED_PRICE_NOT_FLAT",
                    "WARNING",
                    "停牌补齐行的开高低收并不完全相等。",
                    code,
                    date_value,
                    "open,high,low,close",
                    "fill_data=True 的停牌补齐行通常满足 open=high=low=close",
                    "open={0}, high={1}, low={2}, close={3}".format(open_price, high, low, close),
                    "不同 QMT 版本补齐口径、停牌状态边界或源数据异常",
                    "在 QMT 客户端核对该日原始停牌行情；确认口径后决定是否修复",
                    path,
                    index,
                )
        return invalid

    def _validate_continuity(
        self,
        row: dict[str, Any],
        code: str,
        date_value: str,
        path: Path,
        index: Any,
        corporate_actions: set[tuple[str, str]],
        state: _ScanState,
    ) -> None:
        """检查跨日昨收连续性、极端涨跌和成交量数量级突变。

        参数：
            row: 当前证券日线原始字段，为该行的列名到原始值映射。
            code: 当前证券代码。
            date_value: 当前交易日期。
            path: 当前记录来源 CSV 路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。
            corporate_actions: 可解释昨收断层的公司行为主键集合。
            state: 保存上一行及滚动成交量历史的跨日状态。

        返回：
            无返回值；统计异常以 ``WARNING`` 或 ``INFO`` 追加到问题集合。
        """

        current = {
            name: _finite_float(row.get(name))
            for name in ("close", "pre_close", "volume", "suspend_flag")
        }
        previous = state.last_row.get(code)
        pre_close = current["pre_close"]
        if previous is not None and pre_close and previous.get("close"):
            relative = abs(pre_close / previous["close"] - 1.0)
            if relative > 1e-6:
                has_action = (code, date_value) in corporate_actions
                self._row_issue(
                    "PRE_CLOSE_DISCONTINUITY_EXPLAINED" if has_action else "PRE_CLOSE_DISCONTINUITY",
                    "INFO",
                    "当日 pre_close 与上一条实际日线 close 不一致。",
                    code,
                    date_value,
                    "pre_close",
                    "上一实际日线 close={0}".format(previous["close"]),
                    "pre_close={0}, 相对差异={1:.4%}".format(pre_close, relative),
                    "当日存在公司行为记录" if has_action else "除权除息、数据缺口、复权口径变化或行情异常",
                    "无需修复，保留公司行为解释" if has_action else "核对公司行为和 QMT 复权口径，再判断是否重下数据",
                    path,
                    index,
                )
        close = current["close"]
        if close and pre_close:
            change = close / pre_close - 1.0
            if abs(change) > self.config.price_jump_warning_ratio:
                self._row_issue(
                    "ABNORMAL_PRICE_JUMP",
                    "WARNING",
                    "单日收盘相对昨收变化超过统计告警阈值。",
                    code,
                    date_value,
                    "close,pre_close",
                    "abs(close / pre_close - 1) <= {0:.2%}".format(
                        self.config.price_jump_warning_ratio
                    ),
                    "close={0}, pre_close={1}, 涨跌={2:.2%}".format(close, pre_close, change),
                    "真实极端行情、除权口径、价格单位变化或数据损坏",
                    "结合涨跌停规则和公司行为核对，不应仅凭统计告警修改数据",
                    path,
                    index,
                )
        volume = current["volume"]
        suspend = current["suspend_flag"]
        history = state.volume_history.setdefault(
            code, deque(maxlen=self.config.volume_history_window)
        )
        if volume is not None and volume > 0 and suspend == 0.0:
            if len(history) >= self.config.volume_history_min_periods:
                median = statistics.median(history)
                if median > 0:
                    ratio = max(volume / median, median / volume)
                    if ratio >= self.config.volume_scale_warning_ratio:
                        self._row_issue(
                            "ABNORMAL_VOLUME_SCALE",
                            "WARNING",
                            "成交量相对近期中位数发生异常数量级变化。",
                            code,
                            date_value,
                            "volume",
                            "与前 {0} 个有效交易日中位数的倍数 < {1:g}".format(
                                len(history), self.config.volume_scale_warning_ratio
                            ),
                            "volume={0}, 历史中位数={1}, 最大倍数={2:.2f}".format(
                                volume, median, ratio
                            ),
                            "真实放量/缩量、成交量单位改变、重复累计或数据源版本变化",
                            "核对异常日期边界两侧的 QMT 原始成交量单位，不要自动缩放",
                            path,
                            index,
                        )
            history.append(volume)
        state.last_row[code] = {
            "date": date_value,
            "close": close,
        }

    def _close_missing_span(
        self,
        code: str,
        end_position: int,
        end_date: str,
        output: list[dict[str, Any]],
        state: _ScanState,
    ) -> None:
        """关闭一段连续交易日缺口并同时生成详细错误。

        参数：
            code: 存在连续缺口的证券代码。
            end_position: 缺失区间末日在审计日历中的零基位置。
            end_date: 缺失区间末日的八位交易日期。
            output: 累积缺失区间报告行的可变列表。
            state: 保存每只证券当前缺口起点的跨日状态。

        返回：
            无返回值；若该证券没有打开的缺口则不做任何操作。
        """

        opened = state.missing_open.pop(code, None)
        if opened is None or end_position < opened[0]:
            return
        start_position, start_date = opened
        days = end_position - start_position + 1
        evidence = "证券属于分区证券池，且这些日期位于上市退市有效区间；停牌日也应有补齐行"
        output.append(
            {
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": days,
                "evidence": evidence,
            }
        )
        self._add_issue(
            "KLINE_MISSING_SPAN",
            "ERROR",
            "kline_1d",
            "证券在生命周期内存在连续交易日日线缺失。",
            code=code,
            start_date=start_date,
            end_date=end_date,
            expected="连续 {0} 个交易日均有一条日线记录".format(days),
            actual="{0} 至 {1} 共缺失 {2} 个交易日".format(start_date, end_date, days),
            evidence=evidence,
            possible_causes="QMT 本地行情未下载完整、下载批次失败或日期分区写入不完整",
            suggested_action="核对 missing_spans.csv，并使用 repair 模式重建涉及的日期分区",
            source_file=str(self._expected_partition_path(end_date)),
        )

    def _row_issue(
        self,
        issue_code: str,
        level: str,
        message: str,
        code: str,
        date_value: str,
        field_name: str,
        expected: str,
        actual: str,
        possible_causes: str,
        suggested_action: str,
        path: Path,
        index: Any,
    ) -> None:
        """生成包含证券、日期、字段、源文件和 CSV 行号的逐行问题。

        参数：
            issue_code: 稳定的机器可读错误编号。
            level: ``ERROR``、``WARNING`` 或 ``INFO`` 严重级别。
            message: 对问题本身的简明中文说明。
            code: 当前证券代码。
            date_value: 当前八位交易日期。
            field_name: 触发问题的一个或多个字段名称。
            expected: 业务规则要求的理论结果。
            actual: 当前记录的实际值和必要上下文。
            possible_causes: 不擅自修正数据前可供排查的可能原因。
            suggested_action: 推荐的人工核对或可恢复修复动作。
            path: 当前记录来源 CSV 文件路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。

        返回：
            无返回值；问题会追加到当前自检器的问题集合。
        """

        try:
            source_row: int | str = int(index) + 2
        except (TypeError, ValueError):
            source_row = ""
        self._add_issue(
            issue_code,
            level,
            "kline_1d",
            message,
            code=code,
            date=date_value,
            field=field_name,
            expected=expected,
            actual=actual,
            evidence="源记录已按证券代码、交易日期和字段规则逐项校验",
            possible_causes=possible_causes,
            suggested_action=suggested_action,
            source_file=str(path),
            source_row=source_row,
        )

    def _add_issue(
        self,
        issue_code: str,
        level: str,
        dataset: str,
        message: str,
        **details: Any,
    ) -> None:
        """追加一条字段完整且严重级别合法的问题。

        参数：
            issue_code: 稳定的机器可读错误编号。
            level: ``ERROR``、``WARNING`` 或 ``INFO`` 严重级别。
            dataset: 发生问题的数据集名称。
            message: 面向使用者的中文错误说明。
            **details: ``AuditIssue`` 其余定位、证据、原因和建议字段。

        返回：
            无返回值；严重级别非法时抛出 ``ValueError``。
        """

        if level not in {"ERROR", "WARNING", "INFO"}:
            raise ValueError("未知问题级别: {0}".format(level))
        if level == "ERROR":
            details.setdefault("expected", "请结合 issue_code 核对对应业务规则")
            details.setdefault("actual", "未提供原始值；请查看 source_file 并复核源数据")
            details.setdefault(
                "evidence", "该问题由全样本结构、字段或生命周期规则触发"
            )
            details.setdefault(
                "possible_causes", "数据下载失败、文件损坏、字段口径变化或配置不一致"
            )
            details.setdefault(
                "suggested_action", "查看 errors.csv 的证据后重新核对或 repair 相关数据"
            )
            details.setdefault("source_file", str(self.root))
        self.issues.append(
            AuditIssue(
                issue_code=issue_code,
                level=level,
                dataset=dataset,
                message=message,
                **details,
            )
        )
