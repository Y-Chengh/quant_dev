"""执行真实行情遗传编程因子搜索，并输出可审计的完整报告。"""

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
    FactorGeneticSearch,
    FactorSearchResult,
    GeneticProgressEvent,
    GeneticSearchConfig,
    ModelCandidateEvaluator,
    PipelineGrid,
    SearchContext,
    identity,
    op,
    prepare_search_context,
)
from factor_research.metrics import daily_cross_sectional_ic
from factor_research.models.factor_passthrough import FactorPassthroughModelFactory
from factor_research.reporting import (
    IC_TREND_SHORT_MIN_PERIODS,
    IC_TREND_SHORT_WINDOW,
    render_ic_trend_svg,
)


DATABASE = Path(r"D:\量化\market.duckdb")
REPORT_ROOT = Path("logs/_search")
CODE_LIMIT = 10
DATA_START = "2024-01-01"
DATA_END = "2025-12-31"
SELECTION_START = "2024-06-01"
HOLDOUT_START = "2025-07-01"
HOLDOUT_END = "2025-12-31"
HOLDOUT_TOP_K = 3
SEARCH_N_JOBS = 4
SEARCH_BATCH_SIZE = 8
GENETIC_POPULATION_SIZE = 96
GENETIC_MAX_GENERATIONS = 8
GENETIC_MAX_EVALUATIONS = 500
GENETIC_RANDOM_SEED = 20260809
GENETIC_FREE_NODE_COUNT = 3
GENETIC_LENGTH_PENALTY = 0.001
ROLLING_IC_WINDOW = 60
ROLLING_IC_MIN_PERIODS = 20


def print_genetic_progress(event: GeneticProgressEvent) -> None:
    """在主进程按候选批次打印搜索阶段、成功数、预算、耗时和 ETA。

    参数：
        event: 后端完成一个候选批次后生成的只读遗传搜索进度快照。
    """

    stage_labels = {
        "selection": "进化筛选",
        "holdout": "Holdout",
        "model": "模型复验",
    }
    stage = stage_labels.get(event.stage, event.stage)
    generation = (
        f" 第 {event.generation}/{event.max_generations} 代"
        if event.generation is not None
        else ""
    )
    percentage = 100.0 * event.completed / event.total if event.total else 100.0
    eta = f"{event.eta_seconds:.1f}s" if math.isfinite(event.eta_seconds) else "--"
    print(
        f"[{stage}{generation}] "
        f"{event.completed}/{event.total} ({percentage:5.1f}%) | "
        f"成功 {event.successful} 失败 {event.failed} | "
        f"selection {event.selection_evaluations}/{event.max_evaluations} | "
        f"耗时 {event.elapsed_seconds:.1f}s ETA {eta}",
        flush=True,
    )


def build_model_evaluator(
    validation_start: str | pd.Timestamp = HOLDOUT_START,
) -> ModelCandidateEvaluator:
    """构造与主实验共用模型抽象的候选直出评价器。

    参数：
        validation_start: 模型验证区间首个目标日期；缺省使用搜索 holdout 起点。

    返回：
        使用 ``factor_passthrough`` 回归模型、并原样输出末列候选值的评价器。
    """

    return ModelCandidateEvaluator(
        model_factory=FactorPassthroughModelFactory(),
        validation_start=validation_start,
        training_mode="single",
        task="regression",
    )


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
    """保留原单输入网格构造入口，供既有调用方和报告测试继续复用。

    返回：
        包含 3 个数据源和 3 个算子阶段的旧网格空间；主函数已改用遗传搜索。
    """

    return PipelineGrid(
        sources=["close", "volume", "return_1d"],
        stages=[
            [identity(), op("delta", periods=[1, 5])],
            [
                identity(),
                op("ts_stddev", window=[5, 10]),
                op("ts_argmax", window=[5, 10]),
            ],
            [identity(), op("cs_rank")],
        ],
    )


def build_genetic_search_config() -> GeneticSearchConfig:
    """构造支持一元与多输入表达式递归组合的遗传搜索配置。

    返回：
        使用收盘价、成交量和一日收益率为终端，并允许相关性输入继续递归搜索的
        确定性遗传编程配置。
    """

    periods = (1, 5, 10, 20)
    windows = (5, 10, 20)
    return GeneticSearchConfig(
        sources=("close", "volume", "return_1d"),
        operator_parameters={
            # 逐元素算术与连续数值变换。
            "add": {},
            "subtract": {},
            "multiply": {},
            "divide": {},
            "negative": {},
            "absolute": {},
            "log": {},
            "sign": {},
            "power": {"exponent": (0.5, 2.0)},
            "signed_power": {"exponent": (0.5, 2.0)},
            # 单证券历史变换；所有位移均只引用当日或过去数据。
            "delay": {"periods": periods},
            "delta": {"periods": periods},
            "returns": {"periods": periods},
            "ts_sum": {"window": windows},
            "ts_mean": {"window": windows},
            "ts_min": {"window": windows},
            "ts_max": {"window": windows},
            "ts_stddev": {"window": windows, "ddof": (0, 1)},
            "ts_argmax": {"window": windows},
            "ts_argmin": {"window": windows},
            "ts_rank": {"window": windows},
            "ts_correlation": {"window": windows},
            "ts_covariance": {"window": windows},
            # 同一交易日内的横截面变换。
            "cs_rank": {},
            "cs_demean": {},
            "cs_zscore": {},
            "cs_scale": {},
            "cs_winsorize": {
                "lower": (0.01, 0.05),
                "upper": (0.95, 0.99),
            },
        },
        population_size=GENETIC_POPULATION_SIZE,
        max_generations=GENETIC_MAX_GENERATIONS,
        max_evaluations=GENETIC_MAX_EVALUATIONS,
        initial_max_depth=2,
        max_depth=4,
        max_nodes=12,
        max_lookback=30,
        min_coverage=0.6,
        target_coverage=0.9,
        # 一个数据源加一层有效变换不收费；继续包装必须提供足够的 Rank IC 增量。
        free_node_count=GENETIC_FREE_NODE_COUNT,
        length_penalty=GENETIC_LENGTH_PENALTY,
        random_seed=GENETIC_RANDOM_SEED,
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


def _format_cell(value: object, *, column: str | None = None) -> str:
    """把指标值格式化为紧凑且不会破坏 Markdown 表格的文本。

    参数：
        value: 指标、布尔值、日期、表达式或缺失值。
        column: 当前值所属的报告列名；以 ``_rank`` 结尾的名次列会移除无意义的
            小数尾零，缺省为空时沿用普通指标格式。

    返回：
        适合写入 Markdown 单元格的转义文本。
    """

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        if column is not None and column.endswith("_rank"):
            return f"{number:.6f}".rstrip("0").rstrip(".")
        return f"{number:.6f}"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _powershell_single_quoted(value: str) -> str:
    """把文本编码为不会展开美元符号或反引号的 PowerShell 单引号参数。

    参数：
        value: 要作为单个 PowerShell 命令行参数展示的因子表达式文本。

    返回：
        已包围单引号且把内部单引号加倍的 PowerShell 字面量。
    """

    return "'" + value.replace("'", "''") + "'"


def _reproduction_command(
    expression: str,
    context: SearchContext,
    metadata: Mapping[str, Any],
) -> str:
    """生成与搜索数据范围、证券池和直出模型一致的主实验命令。

    参数：
        expression: 要在主实验复验的候选因子规范表达式。
        context: 提供 holdout 验证起点的搜索上下文。
        metadata: 可选包含数据库、数据起止日期、证券列表和证券数量上限的元数据。

    返回：
        可直接粘贴到 PowerShell 的单行 ``run_factor_demo.py`` 命令。
    """

    parts = [
        "python",
        "run_factor_demo.py",
        "--model",
        "factor_passthrough",
        "--task",
        "regression",
        "--training-mode",
        "single",
        "--validation-start",
        _powershell_single_quoted(context.holdout_start.strftime("%Y-%m-%d")),
    ]
    for option, key in (
        ("--database", "database"),
        ("--start", "data_start"),
        ("--end", "data_end"),
    ):
        value = metadata.get(key)
        if value is not None:
            parts.extend([option, _powershell_single_quoted(str(value))])
    codes = list(metadata.get("codes", ()))
    if codes:
        parts.append("--codes")
        parts.extend(_powershell_single_quoted(str(code)) for code in codes)
    elif metadata.get("code_limit") is not None:
        parts.extend(["--symbol-limit", str(metadata["code_limit"])])
    parts.extend(
        [
            "--factors",
            "--factor-expressions",
            _powershell_single_quoted(expression),
        ]
    )
    return " ".join(parts)


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
        "| "
        + " | ".join(
            _format_cell(row[column], column=column) for column in selected
        )
        + " |"
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


def _write_rolling_ic_stability_chart(
    daily_ic: pd.DataFrame,
    path: Path,
    factor_id: str | None,
    *,
    window: int = ROLLING_IC_WINDOW,
    min_periods: int = ROLLING_IC_MIN_PERIODS,
) -> None:
    """调用共享渲染器绘制单个候选的双周期 IC/Rank IC 稳定性图。

    短期趋势固定使用项目通用的 20 日窗口，长期趋势由 ``window`` 指定。传入的
    逐日指标已经按 selection 锁定方向调整；holdout 起点只作为图中分界线，不参与
    方向确定或候选重排。

    参数：
        daily_ic: Top K 候选的逐日横截面指标，必须包含候选 ID、区间、目标日期、
            IC 和 Rank IC 列。
        path: SVG 输出路径，父目录必须已经存在。
        factor_id: 要绘图的候选 ID；为 ``None`` 或找不到对应明细时输出无数据占位图。
        window: 长期滚动均值包含的目标交易日行数，缺省为 60 日。
        min_periods: 长期均线出值所需的最少有限观测数，缺省为 20 日。

    返回：
        无；函数通过共享报告渲染器把 UTF-8 SVG 图像写入 ``path``。
    """

    required = {"factor_id", "period", "target_date", "ic", "rank_ic"}
    missing = required.difference(daily_ic.columns)
    if missing:
        raise ValueError(f"daily_ic 缺少列: {sorted(missing)}")
    selected = daily_ic.iloc[0:0].copy()
    if factor_id is not None:
        selected = daily_ic.loc[daily_ic["factor_id"].astype(str).eq(str(factor_id))].copy()
    holdout_dates = selected.loc[selected["period"].eq("holdout"), "target_date"]
    render_ic_trend_svg(
        selected,
        path,
        title=f"Rolling IC stability — {factor_id or 'no candidate'}",
        short_window=IC_TREND_SHORT_WINDOW,
        short_min_periods=IC_TREND_SHORT_MIN_PERIODS,
        long_window=window,
        long_min_periods=min_periods,
        boundary_date=(
            pd.to_datetime(holdout_dates, errors="coerce").min()
            if not holdout_dates.empty
            else None
        ),
    )


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
        "可把上面的字符串直接加入 `run_factor_demo.py`：",
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
        "python grid_search_smoke.py",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    """加载真实分钟行情、并行执行遗传搜索并生成带时间戳的完整报告目录。

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

    # 分钟聚合和目标都只在这里计算一次。
    context = prepare_search_context(
        bars,
        selection_start=SELECTION_START,
        holdout_start=HOLDOUT_START,
        holdout_end=HOLDOUT_END,
    )
    genetic_config = build_genetic_search_config()
    search = FactorGeneticSearch(
        genetic_config,
        backend="process",
        n_jobs=SEARCH_N_JOBS,
        batch_size=SEARCH_BATCH_SIZE,
    )
    result = search.run(
        context,
        holdout_top_k=HOLDOUT_TOP_K,
        model_evaluator=build_model_evaluator(),
        model_top_k=HOLDOUT_TOP_K,
        progress_callback=print_genetic_progress,
    )
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
            "bar_rows": len(bars),
            "search_algorithm": "genetic_programming",
            "search": {
                "backend": "process",
                "n_jobs": SEARCH_N_JOBS,
                "batch_size": SEARCH_BATCH_SIZE,
                "population_size": genetic_config.population_size,
                "max_generations": genetic_config.max_generations,
                "max_evaluations": genetic_config.max_evaluations,
                "max_nodes": genetic_config.max_nodes,
                "max_depth": genetic_config.max_depth,
                "max_lookback": genetic_config.max_lookback,
                "min_coverage": genetic_config.min_coverage,
                "free_node_count": genetic_config.free_node_count,
                "length_penalty": genetic_config.length_penalty,
                "random_seed": genetic_config.random_seed,
                "operator_parameters": genetic_config.operator_parameters,
                "holdout_top_k": HOLDOUT_TOP_K,
                "model": "factor_passthrough",
                "model_task": "regression",
                "model_top_k": HOLDOUT_TOP_K,
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
