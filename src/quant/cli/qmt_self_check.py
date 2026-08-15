"""QMT 全样本日线数据内部质量自检命令行入口。"""

# python src\quant\cli\qmt_self_check.py --config configs\qmt_downloader\kline_only.backfill.json
# 增量自检
# python -m quant.cli.qmt_self_check `
#   --config configs\qmt_downloader\incremental.example.json `
#   --start-date 20260801 --end-date 20260815

# 全量自检
# python -m quant.cli.qmt_self_check --config configs\qmt_downloader\incremental.example.json

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from quant.config import default_qmt_config_path, default_qmt_errata_path
from quant.qmt_downloader.config import strip_jsonc
from quant.qmt_downloader.self_check import (
    LOGGER_NAME,
    SelfCheckConfig,
    run_full_sample_self_check,
)

DEFAULT_CONFIG = default_qmt_config_path("kline_only.backfill.json")
#: 可选的进度日志级别；``DEBUG`` 会逐个分区、逐个交易日输出。
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def build_parser() -> argparse.ArgumentParser:
    """构造全样本数据自检命令行解析器。

    返回：
        已注册配置、日历、日期范围、报告目录、异常阈值和进度日志参数的解析器。
    """

    parser = argparse.ArgumentParser(
        description="全量检查 QMT 日线分区、证券生命周期、数据缺口、停牌成交量和异常跳变"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="QMT 下载器 JSON 配置；用于读取 output_root 和缺省审计区间",
    )
    parser.add_argument("--output-root", type=Path, help="覆盖配置中的 QMT 数据根目录")
    parser.add_argument(
        "--staging-root",
        type=Path,
        help="直接审计 staging 作业目录，或包含唯一 qmt_<job_key> 子目录的 staging 父目录",
    )
    parser.add_argument("--start-date", help="覆盖配置的审计起点，格式 YYYYMMDD")
    parser.add_argument("--end-date", help="覆盖配置的审计终点，格式 YYYYMMDD")
    parser.add_argument(
        "--calendar-csv",
        type=Path,
        help="QMT 交易日历 CSV；应含 trade_date/date/time 列或仅有一列",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="本次唯一报告目录；缺省写入 output_root/reports/self_check/时间戳",
    )
    parser.add_argument(
        "--coverage-error-threshold",
        type=float,
        default=0.95,
        help="单日覆盖率低于该值时额外报告大范围缺失，缺省 0.95",
    )
    parser.add_argument(
        "--price-jump-warning-ratio",
        type=float,
        default=0.50,
        help="单日绝对涨跌超过该比例时报告统计告警，缺省 0.50",
    )
    parser.add_argument(
        "--volume-scale-warning-ratio",
        type=float,
        default=100.0,
        help="成交量相对近期中位数达到该倍数时报告统计告警，缺省 100",
    )
    parser.add_argument(
        "--errata-csv",
        type=Path,
        default=default_qmt_errata_path(),
        help="人工核实的源数据勘误表 CSV，格式见 docs/qmt_source_data_errata.md；"
        "文件不存在时视为没有勘误记录",
    )
    parser.add_argument(
        "--verify-staging-hash",
        action="store_true",
        help="额外计算 staging 每个批次 CSV 的 SHA-256 并与 .meta.json 比对",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="进度日志级别；DEBUG 逐个分区和交易日输出，WARNING 及以上只保留告警",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        # argparse 会对 help 做 % 插值，字面百分号必须写成 %%。
        help="每处理多少个分区或交易日输出一条进度；缺省 0 表示按总量的 5%% 自动选择",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="把同样的进度日志追加写入该 UTF-8 文件；缺省只输出到终端",
    )
    return parser


def configure_progress_logging(level: str, log_file: Path | None = None) -> None:
    """把自检进度日志接到标准输出，并可选同时写入日志文件。

    自检本身是库层代码，只向名为 ``quant.qmt_downloader.self_check`` 的日志器写入
    而不配置处理器；是否输出、输出到哪里由本入口决定，避免被库层调用时污染宿主的
    日志配置。

    参数：
        level: ``LOG_LEVELS`` 中的日志级别名称，决定进度粒度；其它取值抛
            ``ValueError``，避免拼错级别时静默按默认级别运行。
        log_file: 可选的日志文件路径；缺省只写终端，给定时以追加方式写 UTF-8 文本，
            并自动创建父目录。目录不可写或路径非法时抛 ``OSError``。

    返回：
        无返回值；重复调用会先清空该日志器已有处理器，不会重复输出。
    """

    if level not in LOG_LEVELS:
        raise ValueError("日志级别必须是 {0} 之一".format("、".join(LOG_LEVELS)))
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config_defaults(path: Path) -> dict[str, object]:
    """读取下载器 JSONC 中与自检相关的根目录和日期范围。

    参数：
        path: QMT 下载器配置文件路径；与下载器本身一致，允许 ``//`` 行注释、
            ``/* */`` 块注释和末尾多余逗号。

    返回：
        至少可能包含 ``output_root``、``start_date`` 和 ``end_date`` 的配置字典。
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        text = handle.read()
    try:
        values = json.loads(strip_jsonc(text))
    except ValueError as error:
        raise ValueError(f"下载器配置 {config_path} 解析失败：{error}")
    if not isinstance(values, dict):
        raise ValueError("下载器配置根节点必须是 JSON 对象")
    return values


def main(argv: list[str] | None = None) -> int:
    """解析命令行、配置进度日志、执行全样本自检并打印报告位置。

    参数：
        argv: 可选命令行参数列表；缺省时读取当前进程命令行。

    返回：
        无硬错误返回 0，发现数据错误返回 1，配置或执行失败由入口返回 2。
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every 不能为负数")
    try:
        # 日志装配放在同一个 try 内：--log-file 指向只读目录或非法路径时，应与配置
        # 错误一样返回 2 并给出中文提示，而不是抛出未捕获的 OSError 栈。
        configure_progress_logging(args.log_level, args.log_file)
        defaults = load_config_defaults(args.config)
        configured_output_root = defaults.get("output_root")
        if args.output_root is None and not configured_output_root:
            parser.error("配置或 --output-root 必须提供 QMT 数据根目录")
        output_root = args.output_root or Path(str(configured_output_root))
        config_start = str(defaults.get("start_date", ""))
        config_end = str(defaults.get("end_date", ""))
        start_date = args.start_date or (config_start if config_start.isdigit() else None)
        end_date = args.end_date or (config_end if config_end.isdigit() else None)
        result = run_full_sample_self_check(
            SelfCheckConfig(
                output_root=output_root,
                staging_root=args.staging_root,
                start_date=start_date,
                end_date=end_date,
                calendar_csv=args.calendar_csv,
                errata_csv=args.errata_csv,
                report_dir=args.report_dir,
                coverage_error_threshold=args.coverage_error_threshold,
                price_jump_warning_ratio=args.price_jump_warning_ratio,
                volume_scale_warning_ratio=args.volume_scale_warning_ratio,
                verify_staging_hash=args.verify_staging_hash,
                progress_every=args.progress_every,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"QMT 全样本数据自检未完成: {exc}")
        return 2
    print(
        "QMT 全样本数据自检{0}: errors={1} warnings={2} report={3}".format(
            "未通过" if result.exit_code else "通过",
            result.summary["errors"],
            result.summary["warnings"],
            result.report_dir,
        )
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
