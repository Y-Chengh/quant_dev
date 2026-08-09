from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import TopNBacktestResult
from .experiment import ExperimentResult


METRIC_LABELS = {
    "samples": "样本数",
    "positive_rate": "实际上涨比例",
    "accuracy": "准确率",
    "balanced_accuracy": "平衡准确率",
    "auc": "ROC AUC",
    "ic": "IC",
    "rank_ic": "Rank IC",
    "icir": "ICIR",
    "ic_win_rate": "IC 胜率",
    "pooled_ic": "全样本 Pearson IC",
    "ic_dates": "有效 IC 交易日数",
    "rank_ic_dates": "有效 Rank IC 交易日数",
    "brier_score": "Brier 分数",
    "log_loss": "Log Loss",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "direction_accuracy": "方向准确率",
}


def _format_number(value: Any) -> str:
    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.6f}"


def render_accuracy_trend_svg(trend: pd.DataFrame, output_path: Path) -> None:
    """Render daily accuracy and its 20-day moving average as a standalone SVG."""
    if trend.empty:
        raise ValueError("Cannot render an accuracy trend without daily observations")

    width, height = 1000, 440
    left, right, top, bottom = 72, 28, 38, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    accuracy = trend["accuracy"].to_numpy(dtype=float)
    rolling = trend["accuracy"].rolling(window=20, min_periods=1).mean().to_numpy(dtype=float)
    count = len(trend)

    def x_at(index: int) -> float:
        return left + (plot_width * index / max(1, count - 1))

    def y_at(value: float) -> float:
        return top + (1.0 - value) * plot_height

    def points(values: np.ndarray) -> str:
        return " ".join(f"{x_at(index):.2f},{y_at(value):.2f}" for index, value in enumerate(values))

    dates = pd.to_datetime(trend["target_date"])
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">日级预估准确率趋势</title>',
        '<desc id="desc">每日准确率及二十日移动平均折线图</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.daily{fill:none;stroke:#93c5fd;stroke-width:1.5;opacity:.9}.rolling{fill:none;stroke:#2563eb;stroke-width:3}</style>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_at(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.0%}</text>')
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<polyline class="daily" points="{points(accuracy)}"/>',
            f'<polyline class="rolling" points="{points(rolling)}"/>',
        ]
    )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(dates.iloc[int(index)].strftime("%Y-%m-%d"))
        svg.append(f'<text x="{x:.2f}" y="{height - bottom + 25}" font-size="12" text-anchor="middle">{label}</text>')
    svg.extend(
        [
            f'<line class="daily" x1="{width - 278}" y1="19" x2="{width - 242}" y2="19"/>',
            f'<text x="{width - 234}" y="23" font-size="12">每日准确率</text>',
            f'<line class="rolling" x1="{width - 142}" y1="19" x2="{width - 106}" y2="19"/>',
            f'<text x="{width - 98}" y="23" font-size="12">20 日均线</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(svg), encoding="utf-8")


def render_equity_curve_svg(
    daily_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """将 Top N 及三组横截面对照的日度净值曲线渲染为独立 SVG。

    参数：
        daily_returns: 按目标交易日排序且至少含日期与 Top N 累计净值的日度回测
            结果；全市场平均、Bottom N 和 Mid N 净值列可选，以兼容旧结果。
        output_path: SVG 收益曲线的写入路径。

    返回：
        无；函数将 SVG 内容写入 ``output_path``。
    """

    equity_columns = {
        "equity": ("Top N", "equity"),
        "universe_equity": ("全市场平均", "universe"),
        "bottom_equity": ("Bottom N", "bottom"),
        "mid_equity": ("Mid N", "mid"),
    }
    required = {"target_date", "equity"}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise ValueError(f"收益曲线缺少列: {sorted(missing)}")
    if daily_returns.empty:
        raise ValueError("无法为没有日度收益的回测绘制收益曲线")

    width, height = 1000, 440
    left, right, top, bottom = 82, 28, 38, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    available_equity_columns = {
        column: metadata
        for column, metadata in equity_columns.items()
        if column in daily_returns.columns
    }
    closing_equities = daily_returns.loc[
        :, list(available_equity_columns)
    ].to_numpy(
        dtype=float
    )
    if not np.isfinite(closing_equities).all():
        raise ValueError("收益曲线净值包含 NaN 或无穷值")
    # 显式加入期初净值，确保单日回测也能画出一条可见线段。
    equities = {
        column: np.concatenate(
            ([1.0], daily_returns[column].to_numpy(dtype=float))
        )
        for column in available_equity_columns
    }
    count = len(next(iter(equities.values())))
    all_equities = np.concatenate(list(equities.values()))
    lower = min(1.0, float(all_equities.min()))
    upper = max(1.0, float(all_equities.max()))
    padding = max((upper - lower) * 0.08, max(abs(lower), abs(upper), 1.0) * 0.01)
    y_min, y_max = lower - padding, upper + padding

    def x_at(index: int) -> float:
        """将净值观测序号映射为绘图区横坐标。

        参数：
            index: 从期初零开始的净值观测序号。

        返回：
            当前观测在 SVG 绘图区内的像素横坐标。
        """

        return left + plot_width * index / max(1, count - 1)

    def y_at(value: float) -> float:
        """将策略净值映射为绘图区纵坐标。

        参数：
            value: 需要绘制的累计净值。

        返回：
            当前净值在 SVG 绘图区内的像素纵坐标。
        """

        return top + (y_max - value) / (y_max - y_min) * plot_height

    curve_points = {
        column: " ".join(
            f"{x_at(index):.2f},{y_at(value):.2f}"
            for index, value in enumerate(values)
        )
        for column, values in equities.items()
    }
    date_labels = [
        "期初",
        *(date.strftime("%Y-%m-%d") for date in pd.to_datetime(daily_returns["target_date"])),
    ]
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    y_ticks = np.linspace(y_min, y_max, 5)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Top N 与横截面对照收益曲线</title>',
        '<desc id="desc">Top N、全市场平均、Bottom N 和 Mid N 计入相同双边成本后的累计净值</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.baseline{stroke:#94a3b8;stroke-width:1;stroke-dasharray:5 4}.equity,.universe,.bottom,.mid{fill:none;stroke-width:2.3}.equity{stroke:#059669}.universe{stroke:#2563eb}.bottom{stroke:#dc2626}.mid{stroke:#d97706}</style>',
    ]
    for value in y_ticks:
        y = y_at(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_at(1.0):.2f}" x2="{width - right}" y2="{y_at(1.0):.2f}"/>',
        ]
    )
    for column, (_, css_class) in available_equity_columns.items():
        svg.append(
            f'<polyline class="{css_class}" points="{curve_points[column]}"/>'
        )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(date_labels[int(index)])
        svg.append(
            f'<text x="{x:.2f}" y="{height - bottom + 25}" font-size="12" text-anchor="middle">{label}</text>'
        )
    legend_x = width - 430
    for position, (_, (label, css_class)) in enumerate(
        available_equity_columns.items()
    ):
        x = legend_x + position * 108
        svg.append(
            f'<line class="{css_class}" x1="{x}" y1="19" x2="{x + 28}" y2="19"/>'
        )
        svg.append(f'<text x="{x + 34}" y="23" font-size="12">{label}</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_evaluation_report(
    result: ExperimentResult,
    report_path: Path,
    chart_path: Path,
    run_id: str,
    run_arguments: dict[str, Any],
    yaml_config: str | None = None,
    backtest: TopNBacktestResult | None = None,
    equity_chart_path: Path | None = None,
) -> None:
    """写入模型评估、可选 Top N 回测以及对应 SVG 图表。

    参数：
        result: 模型验证结果及逐证券预测。
        report_path: Markdown 评估报告写入路径。
        chart_path: 日级预测准确率 SVG 图表写入路径。
        run_id: 当前实验的唯一运行标识。
        run_arguments: 已生效的命令行及 YAML 合并参数。
        yaml_config: 原始 YAML 配置文本；未使用配置文件时为 ``None``。
        backtest: Top N 日内策略结果；缺省时不输出回测章节。
        equity_chart_path: 收益曲线 SVG 路径；提供回测结果时必须同时提供。

    返回：
        无；函数写入 Markdown 报告及配置的 SVG 图表。
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    render_accuracy_trend_svg(result.daily_accuracy_trend, chart_path)
    if backtest is not None:
        if equity_chart_path is None:
            raise ValueError("提供 backtest 时必须同时提供 equity_chart_path")
        render_equity_curve_svg(backtest.daily_returns, equity_chart_path)

    predictions = result.predictions.copy()
    if "prediction" in predictions:
        predictions["predicted_up"] = predictions["prediction"].astype(bool)
    else:
        predictions["predicted_up"] = predictions["up_probability"] >= 0.5
    daily_summary = (
        predictions.groupby("target_date", as_index=False, sort=True)
        .agg(
            samples=("label", "size"),
            actual_up_rate=("label", "mean"),
            predicted_up_rate=("predicted_up", "mean"),
        )
        .merge(result.daily_accuracy_trend, on=["target_date", "samples"], how="left")
    )

    lines = [
        (
            "# 涨跌幅预测评估报告"
            if result.task == "regression"
            else "# 方向预测评估报告"
        ),
        "",
        f"- 运行 ID：`{run_id}`",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 验证区间：{pd.Timestamp(daily_summary['target_date'].min()).date()} 至 {pd.Timestamp(daily_summary['target_date'].max()).date()}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in result.metrics.items():
        lines.append(f"| {METRIC_LABELS.get(key, key)} | {_format_number(value)} |")

    lines.extend(
        [
            "",
            "## 准确率趋势",
            "",
            f"![日级预估准确率趋势]({chart_path.name})",
            "",
            "浅色线为每日准确率，深色线为 20 日移动平均。",
        ]
    )

    if backtest is not None:
        metric_labels = {
            "trading_days": "交易日数",
            "total_return": "累计收益率",
            "annualized_return": "年化收益率",
            "annualized_volatility": "年化波动率",
            "sharpe_ratio": "夏普比率",
        }
        lines.extend(
            [
                "",
                "## Top N 日内策略回测",
                "",
                f"- 每日选股数：最多 {backtest.top_n} 只",
                f"- 排序分数：`{backtest.score_column}`",
                f"- 单边滑点：{backtest.slippage_bps:.4f} bps",
                f"- 单边手续费：{backtest.commission_bps:.4f} bps",
                "- 交易口径：目标日开盘等权买入、收盘全部卖出，成本在买卖两边分别计取。",
                "",
                "| 回测指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.metrics.items():
            lines.append(
                f"| {metric_labels.get(key, key)} | {_format_number(value)} |"
            )
        lines.extend(
            [
                "",
                "## Top N 与横截面对照收益曲线",
                "",
                f"![Top N、全市场平均、Bottom N 与 Mid N 收益曲线]({equity_chart_path.name})",
                "",
                "四组均按目标日开盘等权买入、收盘卖出并采用相同双边成本；Mid N 为预测排序居中的最多 N 只。",
                "",
                "## Top N 基准与横截面对照",
                "",
                "等权及随机组合采用与 Top N 相同的双边成本；随机基准按固定种子独立逐日抽样。",
                "",
                "| 基准指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        benchmark_labels = {
            "equal_weight_total_return": "全股票等权累计收益率",
            "equal_weight_annualized_return": "全股票等权年化收益率",
            "equal_weight_sharpe_ratio": "全股票等权夏普比率",
            "random_simulations": "随机 Top N 模拟次数",
            "random_annualized_p05": "随机 Top N 年化收益率 P05",
            "random_annualized_median": "随机 Top N 年化收益率中位数",
            "random_annualized_p95": "随机 Top N 年化收益率 P95",
            "strategy_random_percentile": "策略在随机基准中的百分位",
        }
        for key, value in backtest.benchmark_metrics.items():
            lines.append(
                f"| {benchmark_labels.get(key, key)} | {_format_number(value)} |"
            )

        lines.extend(
            [
                "",
                "## Top N 超额与多空价差",
                "",
                "收益差采用每日毛收益之差和算术年化；该口径用于检验选股能力，不作为可复利长仓净值。",
                "",
                "| 价差指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        relative_labels = {
            "top_minus_universe_annualized_return": "Top N - 全股票等权：算术年化收益",
            "top_minus_universe_annualized_volatility": "Top N - 全股票等权：年化波动率",
            "top_minus_universe_sharpe_ratio": "Top N - 全股票等权：夏普比率",
            "top_minus_bottom_annualized_return": "Top N - Bottom N：算术年化收益",
            "top_minus_bottom_annualized_volatility": "Top N - Bottom N：年化波动率",
            "top_minus_bottom_sharpe_ratio": "Top N - Bottom N：夏普比率",
        }
        for key, value in backtest.relative_metrics.items():
            lines.append(
                f"| {relative_labels.get(key, key)} | {_format_number(value)} |"
            )

        lines.extend(
            [
                "",
                "## 预测分数十分位收益",
                "",
                "十分位 1 为最低预测分数组，十分位 10 为最高预测分数组；收益未扣成本。",
                "",
                "| 十分位 | 样本数 | 交易日数 | 平均日收益 | 算术年化收益 | 上涨比例 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in backtest.decile_returns.itertuples(index=False):
            lines.append(
                f"| {row.predicted_decile} | {row.samples} | {row.trading_days} | "
                f"{_format_number(row.average_return)} | "
                f"{_format_number(row.annualized_arithmetic_return)} | "
                f"{_format_number(row.hit_rate)} |"
            )

        selection_labels = {
            "top_samples": "Top N 样本数",
            "top_hit_rate": "Top N 上涨比例",
            "top_average_return": "Top N 平均日收益",
            "other_samples": "其余股票样本数",
            "other_hit_rate": "其余股票上涨比例",
            "other_average_return": "其余股票平均日收益",
            "top_minus_other_average_return": "Top N 相对其余股票平均日收益差",
        }
        lines.extend(
            [
                "",
                "## Top N 与其余股票命中对照",
                "",
                "| 选股指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.selection_metrics.items():
            lines.append(
                f"| {selection_labels.get(key, key)} | {_format_number(value)} |"
            )

    lines.extend(["", "## 因子重要性", ""])
    if result.feature_importance is None:
        lines.append("当前模型未提供因子重要性。")
    else:
        lines.extend(["| 因子 | 重要性 |", "| --- | ---: |"])
        for factor, importance in result.feature_importance.items():
            lines.append(f"| `{factor}` | {float(importance):.6f} |")

    lines.extend(["", "## 运行参数", "", "```text"])
    lines.extend(f"{key}={value}" for key, value in sorted(run_arguments.items()))
    lines.extend(["```", "", "## YAML 配置", ""])
    if yaml_config is None:
        lines.append("未使用 YAML 配置文件。")
    else:
        config_text = yaml_config.rstrip("\r\n")
        fence = "```"
        while fence in config_text:
            fence += "`"
        lines.extend([f"{fence}yaml", config_text, fence])

    if not result.daily_ic_trend.empty:
        lines.extend(
            [
                "",
                "## 每日横截面 IC",
                "",
                "| 目标日期 | 有效样本数 | IC | Rank IC |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in result.daily_ic_trend.itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(row.target_date).date()} | {row.samples} | "
                f"{_format_number(row.ic)} | {_format_number(row.rank_ic)} |"
            )

    lines.extend(
        [
            "",
            "## 每日预估汇总",
            "",
            "| 目标日期 | 样本数 | 实际上涨比例 | 预测上涨比例 | 准确率 | 较前日变化 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in daily_summary.itertuples(index=False):
        change = "-" if pd.isna(row.accuracy_change) else f"{row.accuracy_change:+.2%}"
        lines.append(
            f"| {pd.Timestamp(row.target_date).date()} | {row.samples} | {row.actual_up_rate:.2%} | "
            f"{row.predicted_up_rate:.2%} | {row.accuracy:.2%} | {change} |"
        )

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
