from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from factor_research.data import validate_bars


DEFAULT_DATABASE = Path(os.getenv("MARKET_DB_PATH", r"D:\量化\market.duckdb"))
DEFAULT_SYMBOL_LIMIT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用最近一个月行情测试单因子 return_1d")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="market.duckdb 路径")
    parser.add_argument("--codes", nargs="+", help="股票代码；默认取代码表前20只")
    parser.add_argument("--symbol-limit", type=int, default=DEFAULT_SYMBOL_LIMIT)
    parser.add_argument("--output", type=Path, help="可选：将结果保存为 CSV")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    return parser.parse_args()


def build_return_1d(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """从5分钟行情只计算日频 return_1d 因子。"""
    bars = validate_bars(bars)
    daily = (
        bars.groupby(["code", "trade_date"], sort=True)
        .agg(close=("close", "last"))
        .reset_index()
        .sort_values(["code", "trade_date"])
    )
    daily["return_1d"] = daily.groupby("code", sort=False)["close"].pct_change()
    return daily[["trade_date", "code", "return_1d"]].sort_values(
        ["trade_date", "code"]
    ).reset_index(drop=True)


def main() -> None:
    from market_service.client import MarketDataClient

    args = parse_args()
    if not 1 <= args.symbol_limit <= 100:
        raise ValueError("symbol-limit 必须在1至100之间")

    client = MarketDataClient(args.database)
    metadata = client.get_metadata()
    end = pd.Timestamp(metadata["last_time"])
    start = max(end - pd.DateOffset(months=1), pd.Timestamp(metadata["first_time"]))
    codes = args.codes or client.search_symbols("", limit=args.symbol_limit)
    if not codes:
        raise RuntimeError("数据库中没有可测试的股票代码")

    bars = client.get_klines_5m(codes, start.to_pydatetime(), end.to_pydatetime())
    if bars.empty:
        raise RuntimeError("最近一个月没有5分钟行情")

    factor = build_return_1d(bars, args)
    print(f"测试区间: {start:%Y-%m-%d %H:%M:%S} 至 {end:%Y-%m-%d %H:%M:%S}")
    print(f"股票数量: {len(codes)}，因子行数: {len(factor)}")
    print(factor.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        factor.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"结果已保存: {args.output.resolve()}")


if __name__ == "__main__":
    main()
