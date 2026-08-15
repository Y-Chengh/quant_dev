"""下载同花顺 iFinD 分钟行情并保存为 CSV。"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from iFinDPy import THS_HF, THS_iFinDLogin, THS_iFinDLogout

DEFAULT_INDICATORS = (
    "volume;amount;change;changeRatio;turnoverRatio;open;high;low;close"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 iFinD 分钟级行情")
    parser.add_argument("--code", default="600000.SH", help="证券代码")
    parser.add_argument("--start", default="2026-03-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-30", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--start-time", default="09:15:00", help="每日开始时间")
    parser.add_argument("--end-time", default="15:15:00", help="每日结束时间")
    parser.add_argument("--indicators", default=DEFAULT_INDICATORS, help="分号分隔指标")
    parser.add_argument("--params", default="Fill:Original", help="THS_HF 参数")
    parser.add_argument("--output", default="data/600000_SH_202603", help="输出目录")
    parser.add_argument("--retry", type=int, default=3, help="单日失败重试次数")
    return parser.parse_args()


def iter_weekdays(start: date, end: date):
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)


def credentials() -> tuple[str, str]:
    """优先读取环境变量，缺失时安全地交互输入。"""
    username = os.getenv("IFIND_USERNAME") or input("iFinD 账号: ").strip()
    password = os.getenv("IFIND_PASSWORD") or getpass.getpass("iFinD 密码: ")
    if not username or not password:
        raise ValueError("账号和密码不能为空")
    return username, password


def fetch_one_day(args: argparse.Namespace, trading_day: date) -> pd.DataFrame:
    begin = f"{trading_day:%Y-%m-%d} {args.start_time}"
    end = f"{trading_day:%Y-%m-%d} {args.end_time}"

    for attempt in range(1, args.retry + 1):
        result = THS_HF(
            args.code,
            args.indicators,
            args.params,
            begin,
            end,
            "format:dataframe",
        )
        if result.errorcode == 0:
            frame = result.data
            if frame is None or frame.empty:
                return pd.DataFrame()
            frame = frame.copy()
            frame.insert(0, "trade_date", trading_day.isoformat())
            return frame

        if attempt < args.retry:
            print(
                f"  请求失败（{result.errorcode}: {result.errmsg}），"
                f"2 秒后重试 {attempt}/{args.retry}..."
            )
            time.sleep(2)
        else:
            raise RuntimeError(
                f"{trading_day:%Y-%m-%d} 获取失败："
                f"{result.errorcode} {result.errmsg}"
            )

    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        raise ValueError("开始日期不能晚于结束日期")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    username, password = credentials()
    login_code = THS_iFinDLogin(username, password)
    if login_code != 0:
        raise RuntimeError(f"iFinD 登录失败，错误码：{login_code}")

    frames: list[pd.DataFrame] = []
    try:
        for trading_day in iter_weekdays(start, end):
            print(f"获取 {trading_day:%Y-%m-%d} ...", flush=True)
            frame = fetch_one_day(args, trading_day)
            if frame.empty:
                print("  无数据，跳过")
                continue

            daily_file = output_dir / f"{args.code.replace('.', '_')}_{trading_day:%Y%m%d}.csv"
            frame.to_csv(daily_file, index=False, encoding="utf-8-sig")
            frames.append(frame)
            print(f"  已保存 {len(frame)} 行 -> {daily_file}")
    finally:
        THS_iFinDLogout()

    if not frames:
        print("指定日期范围内没有获得任何数据。")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined_file = output_dir / (
        f"{args.code.replace('.', '_')}_{start:%Y%m%d}_{end:%Y%m%d}_all.csv"
    )
    combined.to_csv(combined_file, index=False, encoding="utf-8-sig")
    print(f"完成：共 {len(frames)} 个交易日、{len(combined)} 行 -> {combined_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
