from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

from factor_research.backtesting import run_top_n_intraday_backtest
from factor_research.data import load_market_service
from factor_research.dataset import build_direction_dataset
from factor_research.experiment import (
    DirectionExperiment,
    PREDICTION_TASKS,
    TRAINING_MODES,
)
from factor_research.factors import (
    DEFAULT_FEATURES,
    available_factors,
    build_daily_features,
    parse_factor_expressions,
)
from factor_research.models.registry import (
    add_model_selection_argument,
    add_selected_model_arguments,
    available_models,
    model_factory_from_args,
)
from factor_research.reporting import write_evaluation_report
from factor_research.timing import log_elapsed


DEFAULT_DATABASE = Path(os.getenv("MARKET_DB_PATH", r"D:\量化\market.duckdb"))
DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_VALIDATION_YEARS = 1
DEFAULT_SYMBOL_LIMIT = 80
DEFAULT_FACTOR_CACHE = Path(".factor_cache")
DEFAULT_LOG_DIR = Path("logs")
logger = logging.getLogger(__name__)


def _positive_integer(value: str) -> int:
    """解析必须大于零的整数命令行参数。

    参数：
        value: 命令行或 YAML 中待转换的整数字符串。

    返回：
        严格大于零的整数。
    """

    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return converted


def _cost_bps(value: str) -> float:
    """解析合法的单边交易成本基点数。

    参数：
        value: 命令行或 YAML 中的基点数，允许零但必须小于 10000。

    返回：
        位于 ``[0, 10000)`` 的有限浮点基点数。
    """

    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted < 10_000.0:
        raise argparse.ArgumentTypeError("必须是 [0, 10000) 范围内的有限数值")
    return converted


def resolve_equity_chart_path(accuracy_chart_path: Path) -> Path:
    """根据准确率图路径生成同目录、同运行标识的收益曲线路径。

    参数：
        accuracy_chart_path: 本次运行的准确率 SVG 路径。

    返回：
        文件名后缀由 ``_accuracy`` 替换为 ``_equity`` 的 SVG 路径。
    """

    stem = accuracy_chart_path.stem
    if stem.endswith("_accuracy"):
        stem = stem[: -len("_accuracy")]
    return accuracy_chart_path.with_name(f"{stem}_equity{accuracy_chart_path.suffix}")


def _factor_expression_argument(value: str) -> str:
    """校验并规范化一个命令行或 YAML 中的 DSL 因子表达式。

    参数：
        value: 搜索报告输出的 ``canonical``/``expression_str`` 文本。

    返回：
        重新解析后生成的稳定规范字符串。
    """

    try:
        return parse_factor_expressions([value])[0].to_string()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_yaml_config(path: Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """读取 YAML 配置，并要求顶层为参数名到参数值的映射。"""
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except OSError as exc:
        parser.error(f"无法读取配置文件 {path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"YAML 配置文件格式错误 {path}: {exc}")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        parser.error("YAML 配置的顶层必须是参数名到参数值的映射")
    invalid_keys = sorted(
        key for key in loaded if re.fullmatch(r"[a-z][a-z0-9_]*", key) is None
    )
    if invalid_keys:
        parser.error(
            "YAML 参数名必须使用 snake_case: " + ", ".join(invalid_keys)
        )
    return loaded


def _config_defaults(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    ignored_unknown: set[str] | None = None,
) -> dict[str, Any]:
    """按 argparse action 校验并转换 YAML 配置值。"""
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    ignored_unknown = ignored_unknown or set()
    unknown = sorted(set(config) - set(actions) - ignored_unknown)
    if unknown:
        parser.error(f"YAML 配置包含未知参数: {', '.join(unknown)}")

    defaults: dict[str, Any] = {}
    for name, value in config.items():
        if name in ignored_unknown and name not in actions:
            continue
        action = actions[name]
        if name == "config":
            parser.error("YAML 配置中不能再次指定 config")
        if value is None:
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                parser.error(f"YAML 参数 {name} 必须是布尔值")
            defaults[name] = value
            continue

        is_sequence = action.nargs in ("+", "*") or isinstance(action.nargs, int)
        if is_sequence:
            if not isinstance(value, list):
                parser.error(f"YAML 参数 {name} 必须是列表")
            if action.nargs == "+" and not value:
                parser.error(f"YAML 参数 {name} 不能为空列表")
            if isinstance(action.nargs, int) and len(value) != action.nargs:
                parser.error(
                    f"YAML 参数 {name} 必须包含 {action.nargs} 个值，"
                    f"实际为 {len(value)} 个"
                )
            values = value
        else:
            if isinstance(value, (list, dict)):
                parser.error(f"YAML 参数 {name} 必须是单个值")
            values = [value]

        converted: list[Any] = []
        for item in values:
            try:
                # YAML 原生数字、日期等先还原为命令行文本，再走 argparse 的类型转换。
                # 例如 int("True") 会报错，避免把 YAML true 静默转换成整数 1。
                token = str(item)
                converted_item = action.type(token) if action.type else token
            except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
                parser.error(f"YAML 参数 {name} 的值 {item!r} 无效: {exc}")
            if action.choices is not None and converted_item not in action.choices:
                parser.error(
                    f"YAML 参数 {name} 的值 {converted_item!r} 不在可选范围 "
                    f"{list(action.choices)} 内"
                )
            converted.append(converted_item)
        defaults[name] = converted if is_sequence else converted[0]
    return defaults


def _model_argument_names(model_name: str) -> set[str]:
    """返回指定模型声明的参数名，用于识别切换模型后的失效配置。"""
    parser = argparse.ArgumentParser(add_help=False)
    add_selected_model_arguments(parser, model_name)
    return {action.dest for action in parser._actions}


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
    """解析主实验通用参数、回测参数及所选模型的专属参数。

    参数：
        argv: 不含程序名的命令行参数列表；缺省时由 ``argparse`` 读取当前
            进程命令行。命令行值优先于 YAML 中的同名配置。

    返回：
        完成类型、取值范围及模型兼容性初步校验的参数命名空间。
    """

    # 第一阶段读取配置文件和模型名称；第二阶段只加载该模型自己的参数定义。
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument("--config", type=Path)
    add_model_selection_argument(model_parser)
    preliminary, _ = model_parser.parse_known_args(argv)
    config = (
        _load_yaml_config(preliminary.config, model_parser)
        if preliminary.config is not None
        else {}
    )
    configured_model = config.get("model")
    if configured_model is not None:
        if not isinstance(configured_model, str):
            model_parser.error("YAML 参数 model 必须是字符串")
        if configured_model not in available_models():
            model_parser.error(
                f"YAML 参数 model 的值 {configured_model!r} 不在可选范围 "
                f"{available_models()} 内"
            )
        model_parser.set_defaults(model=configured_model)
    selected, _ = model_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description="通过market service预测下一交易日开盘至收盘涨跌")
    parser.add_argument("--config", type=Path, help="YAML 配置文件；命令行参数优先")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="market.duckdb路径")
    parser.add_argument("--start", default='2024-01-01', help="研究开始时间，默认数据末端向前3年")
    parser.add_argument("--end", default='2026-01-01', help="研究结束时间，默认数据库最后时间")
    parser.add_argument(
        "--validation-start",
        default='2025-06-01',
        help="验证集开始日期；默认从研究结束日期往前1年，例如 2024-01-01",
    )
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default="rolling",
        help="训练方式：rolling 为逐日扩展窗口训练，single 为训练集仅拟合一次",
    )
    parser.add_argument(
        "--task",
        choices=PREDICTION_TASKS,
        default="classification",
        help="预测任务：classification 为涨跌二分类，regression 为连续涨跌幅",
    )
    parser.add_argument("--codes", nargs="+", help="股票代码列表，默认取代码表前20只")
    parser.add_argument(
        "--symbol-limit",
        type=int,
        default=DEFAULT_SYMBOL_LIMIT,
        help="股票数量，默认取20只",
    )
    parser.add_argument(
        "--backtest-top-n",
        type=_positive_integer,
        default=10,
        help="回测每日按模型分数买入的最多证券数，默认 10",
    )
    parser.add_argument(
        "--slippage-bps",
        type=_cost_bps,
        default=0.0,
        help="回测单边滑点基点数，买卖两边分别应用，默认 0",
    )
    parser.add_argument(
        "--commission-bps",
        type=_cost_bps,
        default=0.0,
        help="回测单边手续费基点数，买卖两边分别收取，默认 0",
    )

    add_model_selection_argument(parser)
    add_selected_model_arguments(parser, selected.model)
    parser.add_argument(
        "--factors",
        nargs="*",
        choices=available_factors(),
        default=DEFAULT_FEATURES,
        help="运行时选择使用的因子；默认使用全部已注册因子",
    )
    parser.add_argument(
        "--factor-expressions",
        nargs="+",
        type=_factor_expression_argument,
        default=[],
        help=(
            "搜索输出的 canonical/expression_str 因子表达式；"
            "可传多个，并在已注册因子之外加入本次模型"
        ),
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
    ignored_model_arguments: set[str] = set()
    if configured_model is not None and configured_model != selected.model:
        ignored_model_arguments = _model_argument_names(configured_model)
    parser.set_defaults(
        **_config_defaults(
            parser,
            config,
            ignored_unknown=ignored_model_arguments,
        )
    )
    args = parser.parse_args(argv)
    if not args.factors and not args.factor_expressions:
        parser.error("factors 与 factor-expressions 不能同时为空")
    return args


@log_elapsed(logger, "程序运行")
def main() -> None:
    """执行行情加载、因子计算、模型验证、Top N 回测及报告生成。

    返回：
        无；运行产物写入配置的日志归档目录，异常时向调用方抛出错误。
    """

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
    yaml_config_snapshot = (
        args.config.read_text(encoding="utf-8") if args.config is not None else None
    )
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
    # if args.symbol_limit < 1 or args.symbol_limit > 100:
    #     raise ValueError("symbol-limit必须在1至100之间")

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
    logger.info("验证集开始: %s", validation_start.strftime("%Y-%m-%d"))
    logger.info("训练方式: %s", args.training_mode)
    logger.info("预测任务: %s", args.task)
    logger.info("股票数量: %d", len(codes))
    bars = load_market_service(client, codes, start, end)
    logger.info("分钟行情行数: %d", len(bars))
    if bars.empty:
        raise RuntimeError("指定窗口内没有5分钟行情")

    cache_dir = None if args.no_factor_cache else args.factor_cache_dir
    expression_nodes = parse_factor_expressions(
        getattr(args, "factor_expressions", ())
    )
    model_features = [*args.factors, *(node.factor_id for node in expression_nodes)]
    daily = build_daily_features(
        bars,
        feature_columns=args.factors,
        cache_dir=cache_dir,
        factor_expressions=expression_nodes,
    )
    logger.info("开始构建方向预测数据集")
    dataset = build_direction_dataset(daily, feature_columns=model_features, args=args)
    logger.info("方向预测数据集行数: %d，开始模型训练验证", len(dataset))
    result = DirectionExperiment(
        validation_start=validation_start,
        feature_columns=model_features,
        args=args,
        model_factory=model_factory_from_args(args),
        training_mode=args.training_mode,
        task=args.task,
    ).run(dataset)
    logger.info("验证指标: %s", result.metrics)
    score_column = (
        "up_probability" if args.task == "classification" else "predicted_return"
    )
    backtest = run_top_n_intraday_backtest(
        result.predictions,
        score_column=score_column,
        top_n=getattr(args, "backtest_top_n", 10),
        slippage_bps=getattr(args, "slippage_bps", 0.0),
        commission_bps=getattr(args, "commission_bps", 0.0),
    )
    logger.info("Top N 日内策略回测指标: %s", backtest.metrics)
    if result.feature_importance is None:
        logger.info("当前模型未提供因子重要性")
    else:
        logger.info("因子重要性:\n%s", result.feature_importance.to_string())
    write_evaluation_report(
        result,
        report_file,
        chart_file,
        run_id,
        run_arguments,
        yaml_config=yaml_config_snapshot,
        backtest=backtest,
        equity_chart_path=resolve_equity_chart_path(chart_file),
    )
    logger.info("评估报告: %s", report_file.resolve())
    logger.info("HTML 报告: %s", report_file.with_suffix(".html").resolve())


if __name__ == "__main__":
    main()
