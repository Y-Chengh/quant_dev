"""评估报告的落盘编排：拼装正文、渲染图表并写出双格式文件。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtesting import TopNBacktestResult
from ..experiment import ExperimentResult
from .charts import (
    render_accuracy_trend_svg,
    render_equity_curve_svg,
    render_ic_trend_svg,
)
from .formatting import _format_number
from .labels import (
    IC_TREND_LONG_WINDOW,
    IC_TREND_SHORT_WINDOW,
    METRIC_LABELS,
)
from .markdown import _render_top_selection_tables, render_markdown_report_html


def write_evaluation_report(
    result: ExperimentResult,
    report_path: Path,
    chart_path: Path,
    run_id: str,
    run_arguments: dict[str, Any],
    yaml_config: str | None = None,
    backtest: TopNBacktestResult | None = None,
    equity_chart_path: Path | None = None,
    ic_chart_path: Path | None = None,
) -> None:
    """写入模型评估、可选 Top N 回测、HTML 副本以及对应 SVG 图表。

    参数：
        result: 模型验证结果及逐证券预测。
        report_path: Markdown 评估报告写入路径。
        chart_path: 日级预测准确率 SVG 图表写入路径。
        run_id: 当前实验的唯一运行标识。
        run_arguments: 已生效的命令行及 YAML 合并参数。
        yaml_config: 原始 YAML 配置文本；未使用配置文件时为 ``None``。
        backtest: Top N 日内策略结果；缺省时不输出回测章节。
        equity_chart_path: 收益曲线 SVG 路径；提供回测结果时必须同时提供。
        ic_chart_path: IC/Rank IC 双周期趋势 SVG 路径；缺省不输出趋势图，保留
            既有调用接口行为。

    返回：
        无；函数写入 Markdown、同名 HTML 报告及配置的 SVG 图表。
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    render_accuracy_trend_svg(result.daily_accuracy_trend, chart_path)
    if ic_chart_path is not None and not result.daily_ic_trend.empty:
        render_ic_trend_svg(result.daily_ic_trend, ic_chart_path)
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

    if ic_chart_path is not None and not result.daily_ic_trend.empty:
        lines.extend(
            [
                "",
                "## IC 与 Rank IC 滚动趋势",
                "",
                f"![IC 与 Rank IC 的短期及长期趋势]({ic_chart_path.name})",
                "",
                f"左列为 {IC_TREND_SHORT_WINDOW} 日趋势，用于观察近期变化；右列为 {IC_TREND_LONG_WINDOW} 日趋势，用于观察长期稳定性。四个小面板分别按自身滚动值动态缩放纵轴，避免长期趋势被短期波动压成近似直线。滚动窗口只使用当日及以前的逐日横截面指标。",
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
            _render_top_selection_tables(
                backtest.top_selections,
                backtest.score_column,
            )
        )
        lines.extend(
            [
                "",
                "### Top N 与横截面对照收益曲线",
                "",
                f"![Top N、全市场平均、Bottom N 与 Mid N 收益曲线]({equity_chart_path.name})",
                "",
                "四组均按目标日开盘等权买入、收盘卖出并采用相同双边成本；Mid N 为预测排序居中的最多 N 只。"
                "上面板为全区间累计净值；下面板按自然年把净值以上一年末（首年为期初）重定基为 1，"
                "段末净值减一即该自然年收益，并直接在图中标注各年份与四组当年收益。",
                "",
                "### Top N 基准与横截面对照",
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
                "### Top N 超额与多空价差",
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
                "### 预测分数十分位收益",
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
                "### Top N 与其余股票命中对照",
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
    markdown_text = "\n".join(lines)
    report_path.write_text(markdown_text, encoding="utf-8")
    render_markdown_report_html(markdown_text, report_path.with_suffix(".html"))
