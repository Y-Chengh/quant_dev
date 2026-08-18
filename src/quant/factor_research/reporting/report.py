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
from .drawdown_chart import render_drawdown_curve_svg
from .formatting import _format_integer, _format_number, _format_percentage
from .labels import (
    IC_TREND_LONG_WINDOW,
    IC_TREND_SHORT_WINDOW,
    METRIC_LABELS,
)
from .markdown import _render_top_selection_tables, render_markdown_report_html
from .slippage_chart import render_slippage_curves_svg

DRAWDOWN_METRIC_LABELS = {
    "max_drawdown": "最大回撤",
    "max_drawdown_decline_days": "最大回撤峰值至谷底交易日",
    "max_drawdown_recovery_days": "最大回撤谷底至修复交易日",
    "max_drawdown_total_days": "最大回撤区间交易日",
    "current_drawdown": "期末当前回撤",
    "average_drawdown": "日均水下深度",
    "drawdown_days_ratio": "处于回撤的交易日占比",
    "longest_drawdown_days": "最长回撤区间交易日",
    "drawdown_episodes": "回撤区间数",
    "recovered_drawdown_episodes": "已修复回撤区间数",
    "calmar_ratio": "Calmar 比率（年化收益/最大回撤）",
}
DRAWDOWN_PERCENTAGE_METRICS = frozenset(
    {
        "max_drawdown",
        "current_drawdown",
        "average_drawdown",
        "drawdown_days_ratio",
    }
)
DRAWDOWN_INTEGER_METRICS = frozenset(
    {
        "max_drawdown_decline_days",
        "max_drawdown_recovery_days",
        "max_drawdown_total_days",
        "longest_drawdown_days",
        "drawdown_episodes",
        "recovered_drawdown_episodes",
    }
)
DRAWDOWN_EPISODE_LIMIT = 10


def _format_drawdown_metric(key: str, value: Any) -> str:
    """按回撤指标的业务口径选择百分比、整数或通用数值格式。

    参数：
        key: 回撤指标键名，用于区分比率、交易日计数与比率型比值。
        value: 该指标的数值；未修复等无定义场景允许为 NaN。

    返回：
        与指标口径匹配的展示文本。
    """

    if key in DRAWDOWN_PERCENTAGE_METRICS:
        return _format_percentage(value)
    if key in DRAWDOWN_INTEGER_METRICS:
        return _format_integer(value)
    return _format_number(value)


def _render_drawdown_sections(
    backtest: TopNBacktestResult,
    drawdown_chart_path: Path | None,
) -> list[str]:
    """生成回撤诊断指标、回撤修复图与历次回撤区间明细的 Markdown 段落。

    回撤统一按扣除双边成本后的 Top N 净值曲线计算，期初净值为 1.0，取值不大于
    0；未修复的回撤区间修复日期与修复交易日数显示为缺失。旧版回测结果没有回撤
    字段时只输出回撤修复图，不虚构指标。

    参数：
        backtest: Top N 目标收益策略回测结果，读取其回撤指标与回撤区间明细。
        drawdown_chart_path: 回撤修复图 SVG 路径；为 ``None`` 时不插入图像段落。

    返回：
        可直接追加到评估报告的 Markdown 文本行。
    """

    lines: list[str] = []
    if backtest.drawdown_metrics:
        lines.extend(
            [
                "",
                "### 回撤诊断",
                "",
                "回撤按扣除双边成本后的 Top N 净值曲线计算，期初净值 1.0 也参与"
                "历史峰值统计；负值表示相对历史峰值的跌幅，0 表示当日创出新高。"
                "交易日数只统计验证区间内的目标交易日。日均水下深度是逐日回撤在"
                "全部交易日上的均值，创出新高的交易日按 0 计入，因此它明显小于"
                "历次回撤深度的平均值。",
                "",
                "| 回撤指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.drawdown_metrics.items():
            lines.append(
                f"| {DRAWDOWN_METRIC_LABELS.get(key, key)} | "
                f"{_format_drawdown_metric(key, value)} |"
            )
    if drawdown_chart_path is not None:
        lines.extend(
            [
                "",
                "### 历史回撤与修复",
                "",
                f"![Top N 累计净值、历史峰值与水下回撤修复过程]({drawdown_chart_path.name})",
                "",
                "上面板阴影为净值低于历史峰值的水下区间，下面板为逐日回撤深度；"
                "红点标注最大回撤谷底，虚线标注该轮回撤的修复位置，蓝线为全市场"
                "等权对照的回撤。",
            ]
        )
    if backtest.drawdown_metrics:
        lines.extend(["", "### 历次回撤区间", ""])
        episodes = backtest.drawdown_episodes
        if episodes.empty:
            lines.append("验证区间内净值未跌破历史峰值，没有回撤区间。")
            return lines
        ordered = episodes.sort_values("max_drawdown", kind="mergesort")
        shown = ordered.head(DRAWDOWN_EPISODE_LIMIT)
        lines.extend(
            [
                f"按回撤深度降序列出最多 {DRAWDOWN_EPISODE_LIMIT} 段区间，"
                f"共 {len(episodes)} 段；峰值日期为「期初」表示回撤自期初净值"
                "就开始，未修复区间的区间交易日只统计到验证区间末尾。",
                "",
                "| 峰值日期 | 谷底日期 | 修复日期 | 最大回撤 | 峰值至谷底交易日 |"
                " 谷底至修复交易日 | 区间交易日 | 是否修复 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in shown.itertuples(index=False):
            peak_date = (
                "期初"
                if pd.isna(row.peak_date)
                else f"{pd.Timestamp(row.peak_date).date()}"
            )
            recovery_date = (
                "未修复"
                if pd.isna(row.recovery_date)
                else f"{pd.Timestamp(row.recovery_date).date()}"
            )
            lines.append(
                f"| {peak_date} | {pd.Timestamp(row.trough_date).date()} | "
                f"{recovery_date} | {_format_percentage(row.max_drawdown)} | "
                f"{_format_integer(row.decline_days)} | "
                f"{_format_integer(row.recovery_days)} | "
                f"{_format_integer(row.total_days)} | "
                f"{'是' if bool(row.recovered) else '否'} |"
            )
    return lines


def _render_slippage_sections(
    backtest: TopNBacktestResult,
    slippage_chart_path: Path | None,
) -> list[str]:
    """生成不同滑点下的 Top N 收益曲线对比图与汇总指标 Markdown 段落。

    对比曲线复用基准回测的每日选股与手续费率，只替换单边滑点，因此差异可以完全
    归因于滑点。未配置滑点候选（或候选与基准重复被去重后为空）时返回空列表，
    报告不出现该小节，既有输出保持不变。

    参数：
        backtest: Top N 目标收益策略回测结果，读取其滑点对比曲线与汇总指标。
        slippage_chart_path: 滑点对比图 SVG 路径；为 ``None`` 时只输出指标表。

    返回：
        可直接追加到评估报告的 Markdown 文本行。
    """

    metrics = backtest.slippage_metrics
    if metrics.empty:
        return []
    lines = ["", "### 不同滑点下的收益曲线对比", ""]
    if slippage_chart_path is not None:
        lines.extend(
            [
                f"![不同单边滑点下的 Top N 累计净值对比]({slippage_chart_path.name})",
                "",
            ]
        )
    lines.extend(
        [
            f"各档曲线使用同一批每日选股，手续费保持 {backtest.commission_bps:.4f} bps 不变，"
            f"只替换单边滑点；基准为 `--slippage-bps` 的 {backtest.slippage_bps:.4f} bps，"
            "与基准重复的候选滑点已去重。",
            "",
            "| 单边滑点（bps） | 累计收益率 | 年化收益率 | 年化波动率 | 夏普比率 |"
            " 最大回撤 | 年化收益相对基准 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics.itertuples(index=False):
        label = f"{row.slippage_bps:.4f}" + ("（基准）" if bool(row.is_baseline) else "")
        lines.append(
            f"| {label} | {_format_percentage(row.total_return)} | "
            f"{_format_percentage(row.annualized_return)} | "
            f"{_format_percentage(row.annualized_volatility)} | "
            f"{_format_number(row.sharpe_ratio)} | "
            f"{_format_percentage(row.max_drawdown)} | "
            f"{_format_percentage(row.annualized_return_minus_baseline)} |"
        )
    return lines


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
    drawdown_chart_path: Path | None = None,
    slippage_chart_path: Path | None = None,
) -> None:
    """写入模型评估、可选 Top N 回测、HTML 副本以及对应 SVG 图表。

    参数：
        result: 模型验证结果及逐证券预测。
        report_path: Markdown 评估报告写入路径。
        chart_path: 日级预测准确率 SVG 图表写入路径。
        run_id: 当前实验的唯一运行标识。
        run_arguments: 已生效的命令行及 YAML 合并参数。
        yaml_config: 原始 YAML 配置文本；未使用配置文件时为 ``None``。
        backtest: Top N 目标收益策略结果；缺省时不输出回测章节。
        equity_chart_path: 收益曲线 SVG 路径；提供回测结果时必须同时提供。
        ic_chart_path: IC/Rank IC 双周期趋势 SVG 路径；缺省不输出趋势图，保留
            既有调用接口行为。
        drawdown_chart_path: 历史回撤与修复 SVG 路径；缺省不输出回撤图，保留
            既有调用接口行为，回撤指标表仍照常输出。
        slippage_chart_path: 不同滑点收益曲线对比 SVG 路径；缺省不输出对比图，
            指标表仍照常输出。回测未配置滑点候选时该图不会生成。

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
        if drawdown_chart_path is not None:
            render_drawdown_curve_svg(backtest.daily_returns, drawdown_chart_path)
        if slippage_chart_path is not None and not backtest.slippage_curves.empty:
            render_slippage_curves_svg(backtest.slippage_curves, slippage_chart_path)

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
                "## Top N 策略回测",
                "",
                f"- 每日选股数：最多 {backtest.top_n} 只",
                f"- 排序分数：`{backtest.score_column}`",
                f"- 单边滑点：{backtest.slippage_bps:.4f} bps",
                f"- 单边手续费：{backtest.commission_bps:.4f} bps",
                "- 交易口径：按配置目标的起止价等权买入和卖出，成本在买卖两边分别计取。",
                "",
                "| 回测指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.metrics.items():
            lines.append(
                f"| {metric_labels.get(key, key)} | {_format_number(value)} |"
            )
        lines.extend(_render_drawdown_sections(backtest, drawdown_chart_path))
        lines.extend(
            [
                "",
                "### Top N 与横截面对照收益曲线",
                "",
                f"![Top N、全市场平均、Bottom N 与 Mid N 收益曲线]({equity_chart_path.name})",
                "",
                "四组均按配置目标的起止价交易并采用相同双边成本；Mid N 为预测排序居中的最多 N 只。"
                "上面板为全区间累计净值；下面板按自然年把净值以上一年末（首年为期初）重定基为 1，"
                "段末净值减一即该自然年收益，并直接在图中标注各年份与四组当年收益。",
            ]
        )
        lines.extend(_render_slippage_sections(backtest, slippage_chart_path))
        lines.extend(
            [
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
            "equal_weight_max_drawdown": "全股票等权最大回撤",
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
            _render_top_selection_tables(
                backtest.top_selections,
                backtest.score_column,
            )
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
