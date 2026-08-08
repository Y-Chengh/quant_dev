"""执行小规模真实因子网格搜索，并输出可审计的完整报告。"""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from market_service.client import MarketDataClient

from factor_research.data import load_market_service
from factor_research.factor_search import (
    FactorGridSearch,
    FactorSearchResult,
    PipelineGrid,
    SearchContext,
    identity,
    op,
    prepare_search_context,
)
from factor_research.metrics import daily_cross_sectional_ic


DATABASE = Path(r"D:\量化\market.duckdb")
REPORT_ROOT = Path("logs/_search")
CODE_LIMIT = 10
DATA_START = "2024-01-01"
DATA_END = "2025-12-31"
SELECTION_START = "2024-06-01"
HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2025-12-31"
FIXED_FEATURES = ("return_1d", "volatility_5d")
HOLDOUT_TOP_K = 3


def resolve_report_output_dir(
    started_at: datetime,
    random_id: str | None = None,
) -> Path:
    """按日期、启动时间和随机运行 ID 生成独立报告目录。

    参数：
        started_at: 本次搜索开始时间；日期层和时间前缀均以该时间为准。
        random_id: 可选的运行 ID，主要供测试固定输出；缺省时生成 8 位十六进制 ID。

    返回：
        ``logs/_search/YYYY-MM-DD/YYYYMMDD_HHMMSS_<randomID>`` 格式的路径。
    """

    run_id = uuid.uuid4().hex[:8] if random_id is None else random_id
    if not run_id or not run_id.isalnum():
        raise ValueError("random_id 必须是非空字母数字字符串")
    date_folder = started_at.strftime("%Y-%m-%d")
    date_time = started_at.strftime("%Y%m%d_%H%M%S")
    return REPORT_ROOT / date_folder / f"{date_time}_{run_id}"


def build_search_space() -> PipelineGrid:
    """构造本次 smoke 搜索的确定性流水线网格。

    返回：
        包含 3 个数据源和 3 个算子阶段的搜索空间；去重前上界为 90 个候选。
    """

    return PipelineGrid(
        sources=["close", "volume", "return_1d"],
        stages=[
            [
                identity(),
                op("delta", periods=[1, 5]),
            ],
            [
                identity(),
                op("ts_stddev", window=[5, 10]),
                op("ts_argmax", window=[5, 10]),
            ],
            [
                identity(),
                op("cs_rank"),
            ],
        ],
    )


def _json_default(value: object) -> object:
    """把报告元数据中的常见非 JSON 类型转换为稳定文本或标量。

    参数：
        value: 待序列化的路径、日期、NumPy 标量或其他业务对象。

    返回：
        可由标准 ``json`` 模块编码的值。
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化类型 {type(value).__name__}")


def _format_cell(value: object) -> str:
    """把指标值格式化为紧凑且不会破坏 Markdown 表格的文本。

    参数：
        value: 指标、布尔值、日期、表达式或缺失值。

    返回：
        适合写入 Markdown 单元格的转义文本。
    """

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}" if math.isfinite(float(value)) else "—"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """生成不依赖 ``tabulate`` 的 Markdown 指标表。

    参数：
        frame: 报告中需要展示的数据表，可为空或缺少部分可选列。
        columns: 期望展示的列名及顺序；不存在的列会被跳过。

    返回：
        Markdown 表格文本；无可展示数据时返回明确提示。
    """

    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "无数据。"
    header = "| " + " | ".join(selected) + " |"
    divider = "| " + " | ".join("---" for _ in selected) + " |"
    rows = [
        "| " + " | ".join(_format_cell(row[column]) for column in selected) + " |"
        for _, row in frame.loc[:, selected].iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _period_summary(context: SearchContext, mask: np.ndarray) -> dict[str, object]:
    """汇总一个目标日期区间的样本数、证券数和日期边界。

    参数：
        context: 已准备好的只读搜索上下文，目标表按证券和目标日期对齐。
        mask: 与 ``context.targets`` 等长的布尔掩码，表示 selection 或 holdout。

    返回：
        可直接写入报告的区间统计字典。
    """

    period = context.targets.loc[mask]
    if period.empty:
        return {"samples": 0, "dates": 0, "codes": 0, "start": None, "end": None}
    return {
        "samples": len(period),
        "dates": period["target_date"].nunique(),
        "codes": period["code"].nunique(),
        "start": period["target_date"].min(),
        "end": period["target_date"].max(),
    }


def _build_top_daily_ic(
    result: FactorSearchResult,
    context: SearchContext,
    factor_ids: Sequence[str],
) -> pd.DataFrame:
    """重建搜索阶段已完成 holdout 评价候选的逐日横截面 IC 明细。

    因子方向沿用 selection 已锁定的 ``direction``；holdout 仅作为报告区间，
    不参与候选排序或方向选择。

    参数：
        result: 已完成 selection 排名和 Top K holdout 评价的搜索结果。
        context: 保存日频数据、目标和两个互斥日期掩码的搜索上下文。
        factor_ids: 搜索阶段已经成功完成 holdout 评价的冻结候选 ID，顺序与
            selection 排名一致；报告不得自行加入其他候选。

    返回：
        按候选排名、区间和目标日期排列的 IC/Rank IC 明细表。
    """

    frozen_ids = set(factor_ids)
    eligible = result.leaderboard.loc[
        result.leaderboard["eligible"]
        & result.leaderboard["factor_id"].isin(frozen_ids)
    ]
    rows: list[pd.DataFrame] = []
    for rank, (_, leaderboard_row) in enumerate(eligible.iterrows(), start=1):
        factor_id = str(leaderboard_row["factor_id"])
        daily = result.materialize(context, factor_id=factor_id, oriented=True)
        aligned = context.align_factor_values(daily[factor_id])
        candidate = result.get_candidate(factor_id)
        for period_name, mask in (
            ("selection", context.selection_mask),
            ("holdout", context.holdout_mask),
        ):
            if not mask.any():
                continue
            predictions = context.targets.loc[
                mask, ["target_date", "target_return"]
            ].copy()
            predictions["score"] = aligned[mask]
            detail = daily_cross_sectional_ic(predictions, "score")
            detail.insert(0, "canonical", candidate.canonical)
            detail.insert(0, "period", period_name)
            detail.insert(0, "factor_id", factor_id)
            detail.insert(0, "selection_rank", rank)
            rows.append(detail)
    columns = [
        "selection_rank",
        "factor_id",
        "period",
        "target_date",
        "samples",
        "ic",
        "rank_ic",
        "canonical",
    ]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def _write_objective_chart(
    leaderboard: pd.DataFrame,
    path: Path,
    objective: str,
    top_n: int = 20,
) -> None:
    """用纯 SVG 绘制前若干合格候选的 selection 排名指标条形图。

    参数：
        leaderboard: 已按 selection 目标降序排列的完整候选榜单。
        path: SVG 输出路径，父目录必须已存在。
        objective: 排名使用的 selection 指标列名。
        top_n: 图中最多展示的合格候选数量，缺省为 20。

    返回：
        无；函数把 UTF-8 SVG 文件写入 ``path``。
    """

    chart = leaderboard.loc[leaderboard["eligible"], ["factor_id", objective]].head(top_n)
    chart = chart.loc[np.isfinite(pd.to_numeric(chart[objective], errors="coerce"))]
    width = 920
    row_height = 28
    height = max(150, 95 + row_height * len(chart))
    label_width = 180
    plot_width = 650
    maximum = float(chart[objective].max()) if not chart.empty else 1.0
    maximum = maximum if maximum > 0 else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Microsoft YaHei,sans-serif;fill:#202124}.label{font-size:12px}.value{font-size:11px}.title{font-size:18px;font-weight:600}</style>',
        f'<text x="20" y="30" class="title">Top {len(chart)} selection candidates — {escape(objective)}</text>',
    ]
    for row_index, (_, row) in enumerate(chart.iterrows()):
        y = 60 + row_index * row_height
        value = float(row[objective])
        bar_width = max(1.0, value / maximum * plot_width)
        parts.extend(
            [
                f'<text x="20" y="{y + 14}" class="label">{escape(str(row["factor_id"]))}</text>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width:.2f}" height="18" rx="3" fill="#3b82f6"/>',
                f'<text x="{label_width + bar_width + 8:.2f}" y="{y + 14}" class="value">{value:.6f}</text>',
            ]
        )
    if chart.empty:
        parts.append('<text x="20" y="75" class="label">No eligible finite candidates</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grid_search_report(
    result: FactorSearchResult,
    context: SearchContext,
    output_dir: Path,
    metadata: Mapping[str, Any],
    *,
    holdout_top_k: int,
) -> Path:
    """输出网格搜索主报告及完整、可复核的明细附件。

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

    best = result.best_candidate
    best_daily = result.materialize(context, factor_id=best.factor_id, oriented=True)
    best_daily[["code", "trade_date", best.factor_id]].rename(
        columns={best.factor_id: "oriented_factor_value"}
    ).to_csv(best_values_path, index=False, encoding="utf-8-sig")
    candidate_payload = [
        {
            "factor_id": candidate.factor_id,
            "canonical": candidate.canonical,
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
    enriched_metadata = {
        **dict(metadata),
        "objective": result.objective,
        "generated_candidates": len(result.candidates),
        "leaderboard_rows": len(leaderboard),
        "eligible_candidates": int(leaderboard["eligible"].sum()),
        "failed_evaluations": len(errors),
        "successful_holdout_candidates": completed_holdout_ids,
        "best_factor_id": best.factor_id,
        "best_canonical": best.canonical,
        "selection": selection,
        "holdout": holdout,
    }
    metadata_path.write_text(
        json.dumps(enriched_metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    best_row = leaderboard.loc[
        leaderboard["factor_id"].eq(best.factor_id)
    ].iloc[0]
    overview = pd.DataFrame(
        [
            ("候选表达式", len(result.candidates)),
            ("成功进入排行榜", len(leaderboard)),
            ("合格候选", int(leaderboard["eligible"].sum())),
            ("失败评价", len(errors)),
            ("selection 排名目标", result.objective),
            ("最优因子", best.factor_id),
            ("最优目标值", best_row[result.objective]),
            ("报告 holdout 披露上限", holdout_top_k),
            ("成功完成 holdout 评价", len(completed_holdout_ids)),
        ],
        columns=["项目", "结果"],
    )
    split_table = pd.DataFrame(
        [
            {"period": "selection", **selection},
            {"period": "holdout", **holdout},
        ]
    )
    top_columns = [
        "factor_id",
        "canonical",
        "depth",
        "lookback",
        "eligible",
        "selection_coverage",
        "selection_oriented_ic",
        "selection_oriented_rank_ic",
        "selection_oriented_rank_icir",
        "direction",
    ]
    holdout_columns = [
        "factor_id",
        "selection_oriented_rank_ic",
        "holdout_coverage",
        "holdout_oriented_ic",
        "holdout_oriented_rank_ic",
        "holdout_oriented_rank_icir",
        "holdout_oriented_positive_rank_ic_ratio",
    ]
    disclosed = leaderboard.loc[
        leaderboard["factor_id"].isin(completed_holdout_ids)
    ]
    error_summary = (
        errors.groupby("stage", dropna=False).size().rename("count").reset_index()
        if not errors.empty
        else pd.DataFrame(columns=["stage", "count"])
    )
    elapsed_seconds = float(metadata.get("elapsed_seconds", float("nan")))
    requested_codes = list(metadata.get("codes", []))
    evaluated_codes = max(int(selection["codes"]), int(holdout["codes"]))
    lines = [
        "# 因子网格搜索报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"> 本报告请求 {len(requested_codes)} 只证券，实际有 {evaluated_codes} 只进入目标评价区间，属于小规模真实行情 smoke 搜索。其统计功效有限，不应直接视为可交易结论。",
        "",
        "## 执行摘要",
        "",
        _markdown_table(overview, ["项目", "结果"]),
        "",
        f"本次运行耗时 {_format_cell(elapsed_seconds)} 秒。最优表达式为：",
        "",
        f"```text\n{best.canonical}\n```",
        "",
        "![selection 排名图](selection_ranking.svg)",
        "",
        "## 数据与日期切分",
        "",
        _markdown_table(split_table, ["period", "samples", "dates", "codes", "start", "end"]),
        "",
        f"原始分钟行数：{metadata.get('bar_rows', '—')}；日频行数：{len(context.daily)}；请求证券：{', '.join(map(str, requested_codes))}。",
        "",
        "切分按 `target_date` 执行。selection 用于排名和锁定方向；holdout 只报告运行前确定的 Top K，未对其他候选计算 holdout 指标属于预期行为。",
        "",
        "## Selection 排名前 10",
        "",
        _markdown_table(leaderboard.head(10), top_columns),
        "",
        "完整榜单见 [leaderboard.csv](leaderboard.csv)。",
        "",
        "## 预先入选候选的 Holdout 表现",
        "",
        _markdown_table(disclosed, holdout_columns),
        "",
        f"在报告披露上限 Top {holdout_top_k} 内，共识别到 {len(completed_holdout_ids)} 个已由搜索阶段成功完成 holdout 评价的候选。逐日 IC/Rank IC 见 [top_candidates_daily_ic.csv](top_candidates_daily_ic.csv)。报告只重建这些冻结候选的明细，不会扩大 holdout 范围；holdout 结果不用于重排候选。",
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
        "- 横截面 IC 和 Rank IC 按目标交易日独立计算；报告中的 ICIR 未年化。",
        "- 本报告没有依据 holdout 结果增减候选或改变排名。",
        "",
        "## 可复现配置与附件",
        "",
        "- [run_metadata.json](run_metadata.json)：数据范围、证券、搜索参数、运行耗时和切分统计。",
        "- [candidates.json](candidates.json)：全部候选的规范表达式及可反序列化表达式树。",
        "- [leaderboard.csv](leaderboard.csv)：全部成功候选及所有汇总指标。",
        "- [errors.csv](errors.csv)：selection、holdout 各阶段的隔离错误。",
        "- [best_factor_values.csv](best_factor_values.csv)：已按 selection 方向调整的最优因子日频值。",
        "- [top_candidates_daily_ic.csv](top_candidates_daily_ic.csv)：预先入选 Top K 的逐日 selection/holdout IC。",
        "",
        "复现命令：",
        "",
        "```powershell",
        "python grid_search_smoke.py",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """加载真实分钟行情、执行网格搜索并生成带时间戳的完整报告目录。

    返回：
        无；报告写入 ``logs/_search/<date>/<date_time>_<randomID>/`` 并打印路径。
    """

    started_at = datetime.now().astimezone()
    timer = time.perf_counter()
    client = MarketDataClient(DATABASE)
    codes = client.search_symbols("", limit=CODE_LIMIT)
    bars = load_market_service(
        client,
        codes,
        start=DATA_START,
        end=DATA_END,
    )

    # 分钟聚合、固定因子和目标都只在这里计算一次。
    context = prepare_search_context(
        bars,
        fixed_features=FIXED_FEATURES,
        cache_dir=".factor_cache",
        selection_start=SELECTION_START,
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
    )
    space = build_search_space()
    search = FactorGridSearch(
        backend="sequential",
        max_candidates=500,
        max_depth=4,
        max_lookback=30,
        min_coverage=0.6,
    )
    result = search.run(context, space, holdout_top_k=HOLDOUT_TOP_K)
    elapsed_seconds = time.perf_counter() - timer

    report_path = write_grid_search_report(
        result,
        context,
        resolve_report_output_dir(started_at),
        {
            "started_at": started_at,
            "elapsed_seconds": elapsed_seconds,
            "database": DATABASE,
            "codes": codes,
            "code_limit": CODE_LIMIT,
            "data_start": DATA_START,
            "data_end": DATA_END,
            "selection_start": SELECTION_START,
            "holdout_start": HOLDOUT_START,
            "holdout_end": HOLDOUT_END,
            "fixed_features": FIXED_FEATURES,
            "bar_rows": len(bars),
            "search_space_estimated_size": space.estimate_size(),
            "search": {
                "backend": "sequential",
                "max_candidates": 500,
                "max_depth": 4,
                "max_lookback": 30,
                "min_coverage": 0.6,
                "holdout_top_k": HOLDOUT_TOP_K,
            },
        },
        holdout_top_k=HOLDOUT_TOP_K,
    )

    print("\n候选排行榜（前 10）：")
    print(result.leaderboard.head(10).to_string(index=False))
    print("\n最优表达式：")
    print(result.best_candidate.canonical)
    print(f"\n完整报告：{report_path.resolve()}")


if __name__ == "__main__":
    main()
