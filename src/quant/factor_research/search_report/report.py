"""搜索结果的完整可审计报告落盘。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..factor_search import FactorSearchResult, SearchContext
from ..reporting import IC_TREND_SHORT_MIN_PERIODS, IC_TREND_SHORT_WINDOW
from .charts import _write_objective_chart, _write_rolling_ic_stability_chart
from .constants import ROLLING_IC_MIN_PERIODS, ROLLING_IC_WINDOW
from .formatting import _format_cell, _json_default, _markdown_table
from .reproduction import _reproduction_command
from .summaries import _build_top_daily_ic, _period_summary


def write_grid_search_report(
    result: FactorSearchResult,
    context: SearchContext,
    output_dir: Path,
    metadata: Mapping[str, Any],
    *,
    holdout_top_k: int,
) -> Path:
    """输出网格或遗传搜索主报告及完整、可复核的明细附件。

    参数：
        result: 搜索返回的候选、完整排行榜、错误表和 selection 排名目标。
        context: 包含日频特征、前向目标以及 selection/holdout 切分的上下文。
        output_dir: 本次运行的独立报告目录；如不存在会自动创建。
        metadata: 数据源、搜索配置、证券代码和运行耗时等可复现元数据。
        holdout_top_k: 搜索运行时预先指定的 holdout 披露上限；报告还会根据
            ``holdout_elapsed_seconds`` 限制为搜索阶段实际成功评价的候选。

    返回：
        已写入的 Markdown 主报告路径。
    """

    if holdout_top_k < 0:
        raise ValueError("holdout_top_k 不能为负数")
    output_dir.mkdir(parents=True, exist_ok=False)
    leaderboard_path = output_dir / "leaderboard.csv"
    errors_path = output_dir / "errors.csv"
    daily_ic_path = output_dir / "top_candidates_daily_ic.csv"
    best_values_path = output_dir / "best_factor_values.csv"
    candidates_path = output_dir / "candidates.json"
    metadata_path = output_dir / "run_metadata.json"
    chart_path = output_dir / "selection_ranking.svg"
    rolling_ic_chart_path = output_dir / "rolling_ic_stability.svg"
    history_path = output_dir / "evolution_history.csv"
    report_path = output_dir / "report.md"

    leaderboard = result.leaderboard.copy()
    for period in ("selection", "holdout"):
        for metric in ("icir", "rank_icir"):
            source = f"{period}_{metric}"
            target = f"{period}_oriented_{metric}"
            if source in leaderboard and "direction" in leaderboard:
                leaderboard[target] = leaderboard[source] * leaderboard["direction"]
    errors = result.errors.reindex(
        columns=["factor_id", "stage", "error", "elapsed_seconds"]
    )
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
    if "holdout_elapsed_seconds" in leaderboard:
        completed_holdout = leaderboard.loc[
            leaderboard["eligible"]
            & leaderboard["holdout_elapsed_seconds"].notna()
        ].head(holdout_top_k)
    else:
        completed_holdout = leaderboard.iloc[0:0]
    completed_holdout_ids = completed_holdout["factor_id"].astype(str).tolist()
    completed_holdout_mask = leaderboard["factor_id"].astype(str).isin(
        completed_holdout_ids
    )
    for metric in (
        "selection_oriented_rank_ic",
        "holdout_oriented_ic",
        "holdout_oriented_rank_ic",
    ):
        rank_column = f"{metric}_rank"
        leaderboard[rank_column] = float("nan")
        if metric in leaderboard and completed_holdout_mask.any():
            disclosed_values = pd.to_numeric(
                leaderboard.loc[completed_holdout_mask, metric], errors="coerce"
            )
            leaderboard.loc[completed_holdout_mask, rank_column] = (
                disclosed_values.rank(method="average", ascending=False)
            )
    top_daily_ic = _build_top_daily_ic(result, context, completed_holdout_ids)
    top_daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    if not top_daily_ic.empty:
        finite_daily_ic = top_daily_ic.loc[
            np.isfinite(top_daily_ic["rank_ic"])
        ].copy()
        oriented_ratios = (
            finite_daily_ic.assign(positive=finite_daily_ic["rank_ic"] > 0)
            .groupby(["factor_id", "period"])["positive"]
            .mean()
        )
        for period in ("selection", "holdout"):
            if period not in oriented_ratios.index.get_level_values("period"):
                continue
            period_ratios = oriented_ratios.xs(
                period, level="period", drop_level=True
            )
            leaderboard[f"{period}_oriented_positive_rank_ic_ratio"] = (
                leaderboard["factor_id"].map(period_ratios)
            )
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    _write_objective_chart(leaderboard, chart_path, result.objective)
    stability_factor_id = completed_holdout_ids[0] if completed_holdout_ids else None
    _write_rolling_ic_stability_chart(
        top_daily_ic,
        rolling_ic_chart_path,
        stability_factor_id,
    )
    history = getattr(result, "history", None)
    has_history = isinstance(history, pd.DataFrame) and not history.empty
    if has_history:
        history.to_csv(history_path, index=False, encoding="utf-8-sig")

    best = result.best_candidate
    best_daily = result.materialize(context, factor_id=best.factor_id, oriented=True)
    best_daily[["code", "trade_date", best.factor_id]].rename(
        columns={best.factor_id: "oriented_factor_value"}
    ).to_csv(best_values_path, index=False, encoding="utf-8-sig")
    candidate_payload = [
        {
            "factor_id": candidate.factor_id,
            "canonical": candidate.canonical,
            "expression_str": candidate.expression_str,
            "node_count": candidate.node_count,
            "depth": candidate.depth,
            "lookback": candidate.lookback,
            "expression": candidate.expression.to_dict(),
        }
        for candidate in result.candidates
    ]
    candidates_path.write_text(
        json.dumps(candidate_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    selection = _period_summary(context, context.selection_mask)
    holdout = _period_summary(context, context.holdout_mask)
    equivalence_columns = {
        "selection_oriented_value_fingerprint",
        "selection_oriented_rank_fingerprint",
    }
    equivalence_enabled = equivalence_columns.issubset(leaderboard.columns)
    screening_failures = (
        int(errors["stage"].eq("screening").sum())
        if "stage" in errors.columns
        else 0
    )
    equivalent_candidates_removed = (
        max(
            0,
            len(result.candidates) - screening_failures - len(leaderboard),
        )
        if equivalence_enabled
        else 0
    )
    enriched_metadata = {
        **dict(metadata),
        "objective": result.objective,
        "generated_candidates": len(result.candidates),
        "leaderboard_rows": len(leaderboard),
        "equivalence_deduplication_enabled": equivalence_enabled,
        "equivalent_candidates_removed": equivalent_candidates_removed,
        "eligible_candidates": int(leaderboard["eligible"].sum()),
        "failed_evaluations": len(errors),
        "successful_holdout_candidates": completed_holdout_ids,
        "rolling_ic_stability_factor_id": stability_factor_id,
        "rolling_ic_short_window": IC_TREND_SHORT_WINDOW,
        "rolling_ic_short_min_periods": IC_TREND_SHORT_MIN_PERIODS,
        "rolling_ic_window": ROLLING_IC_WINDOW,
        "rolling_ic_min_periods": ROLLING_IC_MIN_PERIODS,
        "best_factor_id": best.factor_id,
        "best_canonical": best.canonical,
        "best_expression_str": best.expression_str,
        "selection": selection,
        "holdout": holdout,
    }
    if has_history:
        enriched_metadata["evolution_generations"] = len(history)
        enriched_metadata["evolution_evaluations"] = int(
            history.iloc[-1]["total_evaluations"]
        )
    metadata_path.write_text(
        json.dumps(enriched_metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    best_row = leaderboard.loc[
        leaderboard["factor_id"].eq(best.factor_id)
    ].iloc[0]
    overview_rows: list[tuple[str, object]] = [
        ("候选表达式", len(result.candidates)),
        ("成功进入排行榜", len(leaderboard)),
    ]
    if equivalence_enabled:
        overview_rows.append(("selection 等价去重", equivalent_candidates_removed))
    overview_rows.extend(
        [
            ("合格候选", int(leaderboard["eligible"].sum())),
            ("失败评价", len(errors)),
            ("selection 排名目标", result.objective),
            ("最优因子", best.factor_id),
            ("最优目标值", best_row[result.objective]),
            ("报告 holdout 披露上限", holdout_top_k),
            ("成功完成 holdout 评价", len(completed_holdout_ids)),
        ]
    )
    overview = pd.DataFrame(overview_rows, columns=["项目", "结果"])
    split_table = pd.DataFrame(
        [
            {"period": "selection", **selection},
            {"period": "holdout", **holdout},
        ]
    )
    top_columns = [
        "factor_id",
        "expression_str",
        "canonical",
        "node_count",
        "depth",
        "lookback",
        "first_generation",
        "eligible",
        "fitness",
        "length_penalty",
        "selection_coverage",
        "selection_oriented_ic",
        "selection_oriented_rank_ic",
        "selection_oriented_rank_icir",
        "direction",
    ]
    holdout_columns = [
        "factor_id",
        "expression_str",
        "selection_oriented_rank_ic",
        "selection_oriented_rank_ic_rank",
        "holdout_coverage",
        "holdout_ic",
        "model_ic",
        "holdout_rank_ic",
        "model_rank_ic",
        "holdout_oriented_ic",
        "holdout_oriented_ic_rank",
        "holdout_oriented_rank_ic",
        "holdout_oriented_rank_ic_rank",
        "holdout_oriented_rank_icir",
        "holdout_oriented_positive_rank_ic_ratio",
    ]
    disclosed = leaderboard.loc[
        leaderboard["factor_id"].isin(completed_holdout_ids)
    ]
    model_metrics_available = (
        "model_ic" in disclosed
        and disclosed["model_ic"].notna().any()
    )
    model_note = (
        "`model_ic`/`model_rank_ic` 是 `factor_passthrough` 对原始候选值的主实验口径复验，"
        "应分别与未定向的 `holdout_ic`/`holdout_rank_ic` 一致。"
        if model_metrics_available
        else "本次结果没有成功完成 `factor_passthrough` 模型复验。"
    )
    error_summary = (
        errors.groupby("stage", dropna=False).size().rename("count").reset_index()
        if not errors.empty
        else pd.DataFrame(columns=["stage", "count"])
    )
    elapsed_seconds = float(metadata.get("elapsed_seconds", float("nan")))
    requested_codes = list(metadata.get("codes", []))
    evaluated_codes = max(int(selection["codes"]), int(holdout["codes"]))
    search_algorithm = str(metadata.get("search_algorithm", "grid"))
    report_title = (
        "因子遗传编程搜索报告"
        if search_algorithm == "genetic_programming"
        else "因子网格搜索报告"
    )
    evolution_lines = (
        [
            "",
            "## 遗传进化轨迹",
            "",
            _markdown_table(
                history,
                [
                    "generation",
                    "population",
                    "new_evaluations",
                    "total_evaluations",
                    "eligible",
                    "best_fitness",
                    "best_factor_id",
                ],
            ),
            "",
            "完整逐代记录见 [evolution_history.csv](evolution_history.csv)。",
        ]
        if has_history
        else []
    )
    equivalence_lines = (
        [
            f"遗传搜索仅在 selection 上按方向调整后的候选值和逐日横截面 Rank 指纹去重，本次移除 {equivalent_candidates_removed} 个等价表达式；`candidates.json` 仍保留全部已评价候选供审计。",
            "",
        ]
        if equivalence_enabled
        else []
    )
    equivalence_constraint_lines = (
        [
            "- 等价候选只依据 selection 方向值及逐日横截面 Rank 判断，每个等价类只保留节点最少者进入排行榜和后续 holdout。"
        ]
        if equivalence_enabled
        else []
    )
    leaderboard_attachment = (
        "- [leaderboard.csv](leaderboard.csv)：selection 等价去重后的候选及所有汇总指标。"
        if equivalence_enabled
        else "- [leaderboard.csv](leaderboard.csv)：全部成功候选及所有汇总指标。"
    )
    lines = [
        f"# {report_title}",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"> 本报告请求 {len(requested_codes)} 只证券，实际有 {evaluated_codes} 只进入目标评价区间，属于受限候选预算的真实行情 smoke 搜索。其统计功效有限，不应直接视为可交易结论。",
        "",
        "## 执行摘要",
        "",
        _markdown_table(overview, ["项目", "结果"]),
        "",
        f"本次运行耗时 {_format_cell(elapsed_seconds)} 秒。最优表达式为：",
        "",
        f"```text\n{best.canonical}\n```",
        "",
        "可把上面的字符串直接加入 `quant-factor-demo`：",
        "",
        "```powershell\n"
        f"{_reproduction_command(best.expression_str, context, metadata)}\n```",
        "",
        "![selection 排名图](selection_ranking.svg)",
        "",
        "## 最优入选候选的滚动 IC 稳定性",
        "",
        "![滚动 IC 与 Rank IC 稳定性图](rolling_ic_stability.svg)",
        "",
        f"图中使用 selection 排名最高且已完成 holdout 评价的候选 `{stability_factor_id or '—'}`。左列为 {IC_TREND_SHORT_WINDOW} 日趋势，用于观察近期拐点；右列为 {ROLLING_IC_WINDOW} 日趋势，用于观察长期稳定性。四个小面板分别按自身滚动值动态缩放纵轴，红色虚线标记 holdout 起点。因子方向只由 selection 锁定，图像用于观察 IC 和 Rank IC 的衰减、漂移与符号翻转，不参与候选重排。",
        "",
        "## 数据与日期切分",
        "",
        _markdown_table(split_table, ["period", "samples", "dates", "codes", "start", "end"]),
        "",
        f"原始分钟行数：{metadata.get('bar_rows', '—')}；日频行数：{len(context.daily)}；请求证券：{', '.join(map(str, requested_codes))}。",
        "",
        "切分按 `target_date` 执行。selection 用于排名和锁定方向；holdout 只报告运行前确定的 Top K，未对其他候选计算 holdout 指标属于预期行为。",
        *equivalence_lines,
        "## Selection 排名前 10",
        "",
        _markdown_table(leaderboard.head(10), top_columns),
        "",
        "完整榜单见 [leaderboard.csv](leaderboard.csv)。",
        *evolution_lines,
        "",
        "## 预先入选候选的 Holdout 表现",
        "",
        _markdown_table(disclosed, holdout_columns),
        "",
        f"在报告披露上限 Top {holdout_top_k} 内，共识别到 {len(completed_holdout_ids)} 个已由搜索阶段成功完成 holdout 评价的候选。{model_note}逐日 IC/Rank IC 见 [top_candidates_daily_ic.csv](top_candidates_daily_ic.csv)。报告只重建这些冻结候选的明细，不会扩大 holdout 范围；holdout 结果不用于重排候选。",
        "",
        "## 失败与覆盖率",
        "",
        _markdown_table(error_summary, ["stage", "count"]),
        "",
        "完整失败明细见 [errors.csv](errors.csv)。覆盖率低于搜索阈值或 selection 目标非有限值的候选会保留在完整榜单中，但 `eligible` 为 false。",
        "",
        "## 口径与防泄漏约束",
        "",
        "- 特征只使用特征日收盘时已可见的数据；滚动窗口以当日为右端点。",
        "- 目标为每只证券下一有效交易日开盘至收盘收益率。",
        "- 因子方向只依据 selection Rank IC 锁定，holdout 沿用同一方向。",
        *equivalence_constraint_lines,
        "- 横截面 IC 和 Rank IC 按目标交易日独立计算；报告中的 ICIR 未年化。",
        "- 本报告没有依据 holdout 结果增减候选或改变排名。",
        "",
        "## 可复现配置与附件",
        "",
        "- [run_metadata.json](run_metadata.json)：数据范围、证券、搜索参数、运行耗时和切分统计。",
        "- [candidates.json](candidates.json)：全部候选的规范表达式及可反序列化表达式树。",
        leaderboard_attachment,
        "- [errors.csv](errors.csv)：selection、holdout、model 各阶段的隔离错误。",
        "- [best_factor_values.csv](best_factor_values.csv)：已按 selection 方向调整的最优因子日频值。",
        "- [top_candidates_daily_ic.csv](top_candidates_daily_ic.csv)：预先入选 Top K 的逐日 selection/holdout IC。",
        "- [rolling_ic_stability.svg](rolling_ic_stability.svg)：最优入选候选的 20 日及 60 日 IC/Rank IC 趋势诊断。",
        *(
            ["- [evolution_history.csv](evolution_history.csv)：遗传搜索逐代种群和最优适应度。"]
            if has_history
            else []
        ),
        "",
        "复现命令：",
        "",
        "```powershell",
        "quant-grid-search",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
