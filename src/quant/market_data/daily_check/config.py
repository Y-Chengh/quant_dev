"""日线库合法性审计的配置与阈值。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

#: ``--price-limit-level`` 的可选值。
PRICE_LIMIT_LEVELS = ("warning", "error")


@dataclass(frozen=True)
class DailyCheckConfig:
    """一次日线库审计所需的路径、范围与阈值。

    参数：
        database: ``qmt_daily.duckdb`` 路径。
        start_date: 审计起点，格式 ``YYYYMMDD``；``None`` 表示从库中最早交易日开始。
        end_date: 审计终点，格式 ``YYYYMMDD``；``None`` 表示到库中最晚交易日为止。
        codes: 只审计这些证券；空元组表示全市场。
        report_dir: 本次报告目录；``None`` 表示写入
            ``<库所在目录>/reports/market_check/<时间戳>``。
        cross_check_5m: 是否与 5 分钟库做聚合交叉校验。
        market_database: 5 分钟库 ``market.duckdb`` 路径；只在交叉校验时使用，
            ``None`` 表示按 ``MARKET_DB_PATH`` 环境变量解析。
        coverage_error_threshold: 单日覆盖率低于该值时报告大范围缺失，
            取值需落在 ``(0, 1]``。
        pre_close_tolerance: 判定 ``pre_close`` 与上一交易日收盘是否一致的相对容差。
        adjust_factor_tolerance: 判定除权因子与行情是否自洽的相对容差。
        price_limit_tolerance: 涨跌停判定的额外容差，用于吸收四舍五入。
        price_limit_level: 涨跌停越界问题的级别。缺省 ``warning``：板块规则本身
            随时间变化，个别历史行情仍可能被误判。
        apply_st_limit: 是否按**当前** ST 状态把主板与老创业板的涨跌停收紧到 5%。
            缺省 ``False``——历史 ST 状态不可知，收紧会产生大量误报。
        cross_check_tolerance: 交叉校验的相对误差阈值。
    """

    database: Path
    start_date: str | None = None
    end_date: str | None = None
    codes: tuple[str, ...] = ()
    report_dir: Path | None = None
    cross_check_5m: bool = False
    market_database: Path | None = None
    coverage_error_threshold: float = 0.95
    pre_close_tolerance: float = 1e-4
    adjust_factor_tolerance: float = 1e-4
    price_limit_tolerance: float = 0.005
    price_limit_level: str = "warning"
    apply_st_limit: bool = False
    cross_check_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        """校验日期格式、阈值范围与级别取值。

        返回：
            无返回值；配置非法时抛出 ``ValueError``。
        """
        _optional_date(self.start_date, "start_date")
        _optional_date(self.end_date, "end_date")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date 不能晚于 end_date")
        if not 0.0 < self.coverage_error_threshold <= 1.0:
            raise ValueError("coverage_error_threshold 必须位于 (0, 1] 范围")
        for name in ("pre_close_tolerance", "adjust_factor_tolerance", "price_limit_tolerance"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负数")
        if self.price_limit_level not in PRICE_LIMIT_LEVELS:
            raise ValueError("price_limit_level 必须是 warning 或 error")
        if self.cross_check_tolerance <= 0:
            raise ValueError("cross_check_tolerance 必须大于 0")

    def resolved_report_dir(self, started_at: datetime) -> Path:
        """解析本次审计的报告目录。

        参数：
            started_at: 审计开始时间，用于生成默认目录名中的时间戳。

        返回：
            显式配置的目录，或 ``<库所在目录>/reports/market_check/<时间戳>``。
        """
        if self.report_dir is not None:
            return Path(self.report_dir)
        stamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
        return Path(self.database).parent / "reports" / "market_check" / stamp


def _optional_date(value: str | None, field_name: str) -> None:
    """校验可选的八位日期文本。

    参数：
        value: 待校验的日期字符串；``None`` 或空串视为未设置。
        field_name: 字段名，仅用于生成可定位的错误消息。

    返回：
        无返回值；格式非法时抛出 ``ValueError``。
    """
    if value is None or value == "":
        return
    try:
        datetime.strptime(value, "%Y%m%d")
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} 必须是 YYYYMMDD 日期")
