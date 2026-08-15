# -*- coding: utf-8 -*-
"""QMT 全样本日线数据内部质量自检命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.config import default_qmt_config_path
from quant.qmt_downloader.self_check import SelfCheckConfig, run_full_sample_self_check


DEFAULT_CONFIG = default_qmt_config_path("kline_only.backfill.json")


def build_parser() -> argparse.ArgumentParser:
    """构造全样本数据自检命令行解析器。

    返回：
        已注册配置、日历、日期范围、报告目录和异常阈值参数的解析器。
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
        "--verify-staging-hash",
        action="store_true",
        help="额外计算 staging 每个批次 CSV 的 SHA-256 并与 .meta.json 比对",
    )
    return parser


def load_config_defaults(path: Path) -> dict[str, object]:
    """读取下载器 JSON 中与自检相关的根目录和日期范围。

    参数：
        path: QMT 下载器 JSON 配置文件路径。

    返回：
        至少可能包含 ``output_root``、``start_date`` 和 ``end_date`` 的配置字典。
    """

    with Path(path).open("r", encoding="utf-8-sig") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError("下载器配置根节点必须是 JSON 对象")
    return values


def main(argv: list[str] | None = None) -> int:
    """解析命令行、执行全样本自检并打印报告位置。

    参数：
        argv: 可选命令行参数列表；缺省时读取当前进程命令行。

    返回：
        无硬错误返回 0，发现数据错误返回 1，配置或执行失败由入口返回 2。
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
                report_dir=args.report_dir,
                coverage_error_threshold=args.coverage_error_threshold,
                price_jump_warning_ratio=args.price_jump_warning_ratio,
                volume_scale_warning_ratio=args.volume_scale_warning_ratio,
                verify_staging_hash=args.verify_staging_hash,
            )
        )
    except (OSError, ValueError) as exc:
        print("QMT 全样本数据自检未完成: {0}".format(exc))
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
