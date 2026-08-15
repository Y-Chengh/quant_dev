"""日线库数据合法性审计的命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant.config import default_market_database, default_qmt_daily_database
from quant.market_data.daily.ingest.cli_support import add_sync_arguments, sync_from_args
from quant.market_data.daily_check import (
    PRICE_LIMIT_LEVELS,
    DailyCheckConfig,
    run_daily_market_check,
)

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """构造日线库审计命令的解析器。

    返回：
        已注册数据库、区间、证券、报告目录、各类阈值与交叉校验参数的解析器。
    """
    parser = argparse.ArgumentParser(
        description="全量检查日线库的缺失区间、停牌与成交一致性、除权自洽性和涨跌停越界"
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="日线库路径；缺省按 QMT_DAILY_DB_PATH 环境变量解析",
    )
    parser.add_argument("--start-date", help="审计起点，格式 YYYYMMDD")
    parser.add_argument("--end-date", help="审计终点，格式 YYYYMMDD")
    parser.add_argument("--codes", nargs="+", help="只审计这些证券；缺省审计全市场")
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="本次唯一报告目录；缺省写入 <库所在目录>/reports/market_check/时间戳",
    )
    parser.add_argument(
        "--coverage-error-threshold",
        type=float,
        default=0.95,
        help="单日覆盖率低于该值时报告大范围缺失，缺省 0.95",
    )
    parser.add_argument(
        "--pre-close-tolerance",
        type=float,
        default=1e-4,
        help="前收与上一交易日收盘的相对容差，缺省 1e-4",
    )
    parser.add_argument(
        "--adjust-factor-tolerance",
        type=float,
        default=1e-4,
        help="除权因子与行情自洽性的相对容差，缺省 1e-4",
    )
    parser.add_argument(
        "--price-limit-tolerance",
        type=float,
        default=0.005,
        help="涨跌停判定的额外容差，用于吸收四舍五入，缺省 0.005",
    )
    parser.add_argument(
        "--price-limit-level",
        choices=PRICE_LIMIT_LEVELS,
        default="warning",
        help="涨跌停越界问题的级别；缺省 warning",
    )
    parser.add_argument(
        "--apply-st-limit",
        action="store_true",
        help="按当前 ST 状态把主板涨跌停收紧到 5%%；历史 ST 状态不可知，会大量误报",
    )
    parser.add_argument(
        "--cross-check-5m",
        action="store_true",
        help="把 5 分钟行情聚合成日频后与日线库交叉校验",
    )
    parser.add_argument(
        "--market-database",
        type=Path,
        help="5 分钟库 market.duckdb 路径；仅在交叉校验时使用",
    )
    parser.add_argument(
        "--cross-check-tolerance",
        type=float,
        default=1e-3,
        help="交叉校验的相对误差阈值，缺省 1e-3",
    )
    add_sync_arguments(parser)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志等级，默认 INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行、执行审计并打印报告位置。

    参数：
        argv: 不含程序名的命令行参数列表；缺省时读取当前进程命令行。

    返回：
        无硬错误返回 0，发现数据错误返回 1，配置或执行失败返回 2。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=DEFAULT_LOG_FORMAT)

    database = args.database if args.database is not None else default_qmt_daily_database()
    try:
        # 审计前的同步一律走完整扫描：水位短路会漏掉「分区被原地重写」的情况，
        # 而审计恰恰不能建立在一份可能已经过期的库上。
        if getattr(args, "sync_mode", "auto") == "auto":
            args.sync_mode = "full"
        sync_from_args(args, database=database)
        result = run_daily_market_check(
            DailyCheckConfig(
                database=database,
                start_date=args.start_date,
                end_date=args.end_date,
                codes=tuple(args.codes or ()),
                report_dir=args.report_dir,
                cross_check_5m=args.cross_check_5m,
                market_database=args.market_database
                if args.market_database is not None
                else (default_market_database() if args.cross_check_5m else None),
                coverage_error_threshold=args.coverage_error_threshold,
                pre_close_tolerance=args.pre_close_tolerance,
                adjust_factor_tolerance=args.adjust_factor_tolerance,
                price_limit_tolerance=args.price_limit_tolerance,
                price_limit_level=args.price_limit_level,
                apply_st_limit=args.apply_st_limit,
                cross_check_tolerance=args.cross_check_tolerance,
            )
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"日线库合法性审计未完成: {error}")
        return 2

    print(
        "日线库合法性审计{0}: errors={1} warnings={2} report={3}".format(
            "未通过" if result.exit_code else "通过",
            result.summary["errors"],
            result.summary["warnings"],
            result.report_dir,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
