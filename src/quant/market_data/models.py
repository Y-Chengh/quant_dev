from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

import pandas as pd


class KlinePeriod(StrEnum):
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    MIN_60 = "60m"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"


@dataclass(frozen=True, slots=True)
class KlineQuery:
    code: str
    start: datetime
    end: datetime
    period: KlinePeriod = KlinePeriod.MIN_5


@dataclass(frozen=True, slots=True)
class RawBarQuery:
    code: str
    trade_date: date
    start_time: str | None = None
    end_time: str | None = None
    min_close: float | None = None
    max_close: float | None = None
    min_volume: int | None = None
    max_volume: int | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    page: int = 1
    page_size: int = 100


@dataclass(frozen=True, slots=True)
class PagedBars:
    data: pd.DataFrame
    total: int
    page: int
    page_size: int
