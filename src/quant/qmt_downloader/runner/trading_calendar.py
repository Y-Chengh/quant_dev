# -*- coding: utf-8 -*-
"""交易日历快照的累积合并与落表。"""

import pandas as pd

from ..dates import normalize_date
from .base import _RunnerState

TRADING_CALENDAR_COLUMNS = ("trade_date", "calendar_symbol")


class _TradingCalendarMixin(_RunnerState):
    """把本次请求区间的交易日并入 ``trading_calendar`` 快照。"""

    def _write_trading_calendar(self, trade_dates):
        """把本次交易日合并进 ``trading_calendar/snapshot=latest`` 并整体覆盖。

        与 ``instrument_info`` 一样按快照而非按日分区保存：交易日历是一张整体覆盖
        的表，没有“某一天的日历”这种业务分区。但它必须**累积**而不是直接覆盖：日
        增量每次只请求一两天，若用本次结果直接替换，历史日历会被截成一天，自检就再
        也拿不到完整审计区间。因此这里读回已有快照，与本次区间求并集后重写；本次区
        间内的日期一律以本次结果为准，区间外的历史日期原样保留。

        注意合并只增不删：某个日期一旦写入，即使大 QMT 后来不再把它视为交易日也不会
        被移除。需要彻底重建时删除该快照目录后以覆盖同一区间的配置重跑。

        参数：
            trade_dates: 本次已从大 QMT 交易日历接口取得的升序八位交易日列表。

        返回：
            分区存储层返回的状态、行数和文件路径字典。
        """
        merged = {}
        existing = self.store.read_partition(
            "trading_calendar", "snapshot", "latest", dtype={"trade_date": str}
        )
        dropped = 0
        if existing is not None and "trade_date" in existing.columns:
            symbols = (
                existing["calendar_symbol"]
                if "calendar_symbol" in existing.columns
                else pd.Series([""] * len(existing), index=existing.index)
            )
            for raw_date, raw_symbol in zip(existing["trade_date"], symbols):
                date_value = normalize_date(raw_date)
                if date_value is None:
                    # 单行损坏不应丢弃整份历史日历，计数后跳过并在日志中说明。
                    dropped += 1
                    continue
                merged[date_value] = "" if pd.isna(raw_symbol) else str(raw_symbol)
        if dropped:
            self.logger.warning(
                "交易日历快照存在无法解析的日期 rows=%d，已跳过这些行", dropped
            )
        for date_value in trade_dates:
            merged[str(date_value)] = self.config.calendar_symbol
        ordered = sorted(merged)
        frame = pd.DataFrame(
            {
                "trade_date": ordered,
                "calendar_symbol": [merged[value] for value in ordered],
            },
            columns=list(TRADING_CALENDAR_COLUMNS),
        )
        result = self._write_partition(
            "trading_calendar",
            "snapshot",
            "latest",
            frame,
            TRADING_CALENDAR_COLUMNS,
            ["trade_date"],
            ["trade_date"],
            overwrite=True,
        )
        self.logger.info(
            "[trading_calendar] 交易日历落表 dates=%d first=%s last=%s path=%s",
            len(ordered),
            ordered[0] if ordered else "",
            ordered[-1] if ordered else "",
            result["path"],
        )
        return result
