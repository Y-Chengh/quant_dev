"""执行真实行情遗传编程因子搜索，并输出可审计的完整报告。"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.config import default_market_database
from quant.factor_research.data import load_market_service
from quant.factor_research.dataset import DEFAULT_LABEL_RETURN_THRESHOLD
from quant.factor_research.factor_search import (
    FactorGeneticSearch,
    GeneticProgressEvent,
    GeneticSearchConfig,
    ModelCandidateEvaluator,
    PipelineGrid,
    identity,
    op,
    prepare_search_context,
)
from quant.factor_research.models.factor_passthrough import FactorPassthroughModelFactory
from quant.factor_research.search_report import write_grid_search_report
from quant.market_data.client import MarketDataClient

DATABASE = default_market_database()
REPORT_ROOT = Path("logs/_search")
CODE_LIMIT = 500
DATA_START = "2018-01-01"
DATA_END = "2021-12-31"
SELECTION_START = "2018-01-01"
HOLDOUT_START = "2020-01-01"
HOLDOUT_END = "2021-12-31"
HOLDOUT_TOP_K = 20
SEARCH_N_JOBS = 8
SEARCH_BATCH_SIZE = 8
GENETIC_POPULATION_SIZE = 96
GENETIC_MAX_GENERATIONS = 8
GENETIC_MAX_EVALUATIONS = 500
GENETIC_RANDOM_SEED = 20260809
GENETIC_FREE_NODE_COUNT = 3
GENETIC_LENGTH_PENALTY = 0.001
LABEL_RETURN_THRESHOLD = DEFAULT_LABEL_RETURN_THRESHOLD


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
        sources=["close", "volume", "return_1d", "high", "low", "open"],
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
        使用收盘价、成交量、一日收益率及最高价、最低价、开盘价为终端，并允许
        相关性输入继续递归搜索的确定性遗传编程配置；数据源与网格空间保持一致。
    """

    periods = (1, 5, 10, 20)
    windows = (5, 10, 20)
    return GeneticSearchConfig(
        sources=("close", "volume", "return_1d", "high", "low", "open"),
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
        label_return_threshold=LABEL_RETURN_THRESHOLD,
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
            "label_return_threshold": LABEL_RETURN_THRESHOLD,
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
