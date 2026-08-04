from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from factor_research.data import load_market_service
from factor_research.dataset import build_direction_dataset
from factor_research.experiment import DirectionExperiment
from factor_research.factors import DEFAULT_FEATURES, available_factors, build_daily_features
from factor_research.models.registry import (
    add_model_selection_argument,
    add_selected_model_arguments,
    model_factory_from_args,
)
from factor_research.reporting import write_evaluation_report
from factor_research.timing import log_elapsed


DEFAULT_DATABASE = Path(os.getenv("MARKET_DB_PATH", r"D:\量化\market.duckdb"))
DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_VALIDATION_YEARS = 1
DEFAULT_SYMBOL_LIMIT = 20
DEFAULT_FACTOR_CACHE = Path(".factor_cache")
DEFAULT_LOG_DIR = Path("logs")
logger = logging.getLogger(__name__)


def resolve_window(
    metadata: dict,
    start: str | None = None,
    end: str | None = None,
) -> tuple[datetime, datetime]:
    """默认以数据集末端为终点，向前取三年。"""
    dataset_end = pd.Timestamp(metadata["last_time"])
    end_time = pd.Timestamp(end) if end else dataset_end
    start_time = pd.Timestamp(start) if start else end_time - pd.DateOffset(years=DEFAULT_LOOKBACK_YEARS)
    dataset_start = pd.Timestamp(metadata["first_time"])
    start_time = max(start_time, dataset_start)
    end_time = min(end_time, dataset_end)
    if start_time >= end_time:
        raise ValueError(f"无效研究窗口: {start_time} 至 {end_time}")
    return start_time.to_pydatetime(), end_time.to_pydatetime()


def resolve_run_output_paths(
    log_dir: Path,
    started_at: datetime,
    run_id: str,
) -> tuple[Path, Path, Path]:
    """Place all artifacts from one run in its start-date archive directory."""
    archive_dir = log_dir / started_at.strftime("%Y-%m-%d")
    stem = f"factor_demo_{started_at:%Y%m%d_%H%M%S}_{run_id}"
    return (
        archive_dir / f"{stem}.log",
        archive_dir / f"{stem}.md",
        archive_dir / f"{stem}_accuracy.svg",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # 第一阶段只识别模型名称；第二阶段只加载该模型自己的参数定义。
    model_parser = argparse.ArgumentParser(add_help=False)
    add_model_selection_argument(model_parser)
    selected, _ = model_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="通过market service预测下一交易日涨跌")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="market.duckdb路径")
    parser.add_argument("--start", help="研究开始时间，默认数据末端向前3年")
    parser.add_argument("--end", help="研究结束时间，默认数据库最后时间")
    parser.add_argument("--codes", nargs="+", help="股票代码列表，默认取代码表前20只")
    parser.add_argument("--symbol-limit", type=int, default=DEFAULT_SYMBOL_LIMIT)
    parser.add_argument(
        "--validation-start",
        help="滚动验证开始日期；默认从研究结束日期往前1年，例如 2024-01-01",
    )
    add_model_selection_argument(parser)
    add_selected_model_arguments(parser, selected.model)
    parser.add_argument(
        "--factors",
        nargs="+",
        choices=available_factors(),
        default=DEFAULT_FEATURES,
        help="运行时选择使用的因子；默认使用全部已注册因子",
    )
    parser.add_argument(
        "--factor-cache-dir",
        type=Path,
        default=DEFAULT_FACTOR_CACHE,
        help="因子Parquet缓存目录；默认 .factor_cache",
    )
    parser.add_argument("--no-factor-cache", action="store_true", help="禁用因子缓存")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志等级，默认 INFO",
    )
    parser.add_argument("--debug", action="store_true", help="开启调试模式并使用 DEBUG 日志等级")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="日志目录，默认 logs；文件名由时间戳和随机ID自动生成",
    )
    return parser.parse_args(argv)


@log_elapsed(logger, "程序运行")
def main() -> None:
    try:
        from market_service.client import MarketDataClient
    except ModuleNotFoundError as exc:
        if exc.name == "duckdb":
            raise SystemExit(
                "当前Python环境缺少duckdb。请先运行: "
                "python -m pip install -r requirements-factor-research.txt"
            ) from exc
        raise

    args = parse_args()
    if args.debug:
        args.log_level = "DEBUG"
    run_id = uuid4().hex[:8]
    run_started_at = datetime.now()
    log_file, report_file, chart_file = resolve_run_output_paths(args.log_dir, run_started_at, run_id)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_formatter)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=[console_handler, file_handler],
        force=True,
    )
    run_arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    logger.info("运行参数: %s", json.dumps(run_arguments, ensure_ascii=False, sort_keys=True))
    logger.info("运行ID: %s，日志文件: %s", run_id, log_file.resolve())
    if args.symbol_limit < 1 or args.symbol_limit > 100:
        raise ValueError("symbol-limit必须在1至100之间")

    client = MarketDataClient(args.database)
    metadata = client.get_metadata()
    start, end = resolve_window(metadata, args.start, args.end)
    validation_start = (
        pd.Timestamp(args.validation_start)
        if args.validation_start
        else pd.Timestamp(end) - pd.DateOffset(years=DEFAULT_VALIDATION_YEARS)
    )
    if validation_start <= pd.Timestamp(start) or validation_start > pd.Timestamp(end):
        raise ValueError(
            f"validation-start 必须晚于研究开始日期且不晚于结束日期: "
            f"{start:%Y-%m-%d} < validation-start <= {end:%Y-%m-%d}"
        )
    codes = args.codes or client.search_symbols("", limit=args.symbol_limit)
    if not codes:
        raise RuntimeError("market service未返回可研究的股票代码")

    logger.info("数据窗口: %s 至 %s", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    logger.info("滚动验证开始: %s", validation_start.strftime("%Y-%m-%d"))
    logger.info("股票数量: %d", len(codes))
    bars = load_market_service(client, codes, start, end)
    logger.info("分钟行情行数: %d", len(bars))
    if bars.empty:
        raise RuntimeError("指定窗口内没有5分钟行情")

    cache_dir = None if args.no_factor_cache else args.factor_cache_dir
    daily = build_daily_features(bars, feature_columns=args.factors, cache_dir=cache_dir)
    logger.info("开始构建方向预测数据集")
    dataset = build_direction_dataset(daily, feature_columns=args.factors, args=args)
    logger.info("方向预测数据集行数: %d，开始滚动训练验证", len(dataset))
    result = DirectionExperiment(
        validation_start=validation_start,
        feature_columns=args.factors,
        args=args,
        model_factory=model_factory_from_args(args),
    ).run(dataset)
    logger.info("滚动验证指标: %s", result.metrics)
    logger.info("日级预估准度变化趋势:\n%s", result.daily_accuracy_trend.to_string(index=False))
    logger.info("因子重要性:\n%s", result.feature_importance.to_string())
    write_evaluation_report(result, report_file, chart_file, run_id, run_arguments)
    logger.info("评估报告: %s，准确率趋势图: %s", report_file.resolve(), chart_file.resolve())


if __name__ == "__main__":
    main()
