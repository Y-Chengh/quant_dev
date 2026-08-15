from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import pandas as pd

from .data import validate_bars, validate_daily_bars
from .factor_dsl import DailyFactorFrame, ExpressionNode
from .factor_factories import FACTOR_FACTORIES, get_factor_factory
from .factor_factories.base import FactorFactory
from .factor_factories.capabilities import daily_capable_factor_names, intraday_factor_names
from .timing import log_elapsed

logger = logging.getLogger(__name__)
DEFAULT_FEATURES = sorted(FACTOR_FACTORIES)
KEY_COLUMNS = ["code", "trade_date"]


def available_factors() -> list[str]:
    """返回当前已注册正式因子的稳定排序名称列表。"""

    return sorted(FACTOR_FACTORIES)


def parse_factor_expressions(
    expressions: Sequence[str | ExpressionNode] | None,
) -> list[ExpressionNode]:
    """解析并校验运行时 DSL 因子表达式，保留调用方给出的顺序。

    参数：
        expressions: 搜索输出的规范字符串或已解析节点；为空时返回空列表。

    返回：
        已重新验证算子参数与因果性的不可变表达式节点列表。
    """

    parsed = [
        ExpressionNode.from_string(item) if isinstance(item, str) else item
        for item in (expressions or ())
    ]
    if not all(isinstance(node, ExpressionNode) for node in parsed):
        raise TypeError("运行时因子表达式必须是字符串或 ExpressionNode")
    factor_ids = [node.factor_id for node in parsed]
    if len(factor_ids) != len(set(factor_ids)):
        raise ValueError("运行时因子表达式不能包含重复项")
    return parsed


def _normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """统一代码和日期类型，避免 Parquet 类型差异干扰逻辑主键比较。

    参数：
        frame: 含证券代码和交易日键列的日频表或缓存表。
    """

    keys = frame[KEY_COLUMNS].copy().reset_index(drop=True)
    keys["code"] = keys["code"].astype(str)
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="raise").astype("datetime64[ns]")
    return keys


def _build_daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """按证券和交易日把已校验分钟行情聚合为排序后的日频 OHLCV。

    参数：
        bars: 已通过行情契约校验的分钟 OHLCV 表。
    """

    return (
        bars.groupby(KEY_COLUMNS, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values(["code", "trade_date"])
        .reset_index(drop=True)
    )


def aggregate_daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """校验分钟行情并只聚合基础日频 OHLCV，不计算任何注册因子。

    网格搜索在没有固定因子时使用该入口。原有 ``build_daily_features`` 仍走
    原来的内部聚合函数，因此既不改变默认因子集合，也不改变既有缓存指纹。

    参数：
        bars: 待校验并按证券与交易日聚合的分钟行情表。
    """

    validated = validate_bars(bars)
    return _build_daily_bars(validated).sort_values(
        ["trade_date", "code"]
    ).reset_index(drop=True)


def _input_fingerprint(bars: pd.DataFrame) -> str:
    """根据分钟行情列类型和全部值生成稳定的缓存输入指纹。

    参数：
        bars: 要标识版本的已校验分钟行情表。
    """

    columns = ["code", "trade_time", "open", "high", "low", "close", "volume"]
    digest = hashlib.sha256()
    digest.update("|".join(f"{column}:{bars[column].dtype}" for column in columns).encode())
    digest.update(pd.util.hash_pandas_object(bars[columns], index=False).values.tobytes())
    return digest.hexdigest()[:20]


def _implementation_fingerprint(factory: FactorFactory) -> str:
    """根据具体工厂、公共基类和日频聚合实现生成代码版本指纹。

    参数：
        factory: 要计算实现版本的具体正式因子工厂实例。
    """

    digest = hashlib.sha256()
    # Include the concrete factory, shared helpers, and base daily aggregation.
    module_path = Path(inspect.getfile(factory.__class__))
    base_path = Path(inspect.getfile(FactorFactory))
    digest.update(module_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(inspect.getsource(_build_daily_bars).encode())
    return digest.hexdigest()[:16]


def _expression_implementation_fingerprint(node: ExpressionNode) -> str:
    """根据表达式、DSL 实现及正式因子依赖生成代码版本指纹。

    参数：
        node: 要标识实现版本的已校验运行时表达式节点。

    返回：
        可用于表达式缓存文件名的稳定短指纹。
    """

    digest = hashlib.sha256(node.canonical.encode("utf-8"))
    dsl_directory = Path(inspect.getfile(DailyFactorFrame)).parent
    for module_path in sorted(dsl_directory.glob("*.py")):
        digest.update(module_path.name.encode("utf-8"))
        digest.update(module_path.read_bytes())
    digest.update(inspect.getsource(_build_daily_bars).encode("utf-8"))
    for dependency_name in sorted(node.columns.intersection(FACTOR_FACTORIES)):
        dependency = get_factor_factory(dependency_name)
        digest.update(dependency_name.encode("utf-8"))
        digest.update(_implementation_fingerprint(dependency).encode("ascii"))
    return digest.hexdigest()[:16]


class FactorCache:
    """按因子实现版本和行情输入版本安全读写独立的 Parquet 缓存。"""

    def __init__(self, root: str | Path, namespace: str = ""):
        """保存缓存根目录，具体因子目录在首次写入时创建。

        参数：
            root: 所有因子版本化 Parquet 缓存的根目录。
            namespace: 缓存命名空间，用于隔离不同数据源与不同复权口径的缓存，
                例如 ``qmt_daily/hfq``。缺省空串时 ``Path(root) / ""`` 仍等于
                ``root``，因此既有 5 分钟缓存的路径逐字节不变。
        """

        self.root = Path(root) / namespace if namespace else Path(root)

    def load(
        self,
        factory: FactorFactory,
        input_fingerprint: str,
        daily: pd.DataFrame,
        daily_mode: bool = False,
    ) -> pd.Series | None:
        """读取并严格校验缓存列、行数和日频主键，失效时返回 ``None``。

        参数：
            factory: 决定因子名和实现版本的工厂实例。
            input_fingerprint: 当前分钟行情内容的稳定短指纹。
            daily: 用于校验缓存行数和主键的当前日频表。
            daily_mode: 是否走日频数据源路径。为 ``True`` 时改用日频实现指纹，
                与分钟路径的缓存文件天然不会互相命中。
        """

        path = self._path(factory, input_fingerprint, daily_mode)
        return self._load_series(factory.name, path, daily)

    def load_expression(
        self,
        node: ExpressionNode,
        input_fingerprint: str,
        daily: pd.DataFrame,
        daily_mode: bool = False,
    ) -> pd.Series | None:
        """读取并校验一个运行时表达式因子的持久化缓存。

        参数：
            node: 决定稳定因子 ID 和表达式实现版本的节点。
            input_fingerprint: 当前分钟行情内容的稳定短指纹。
            daily: 用于校验缓存行数和主键的当前日频表。
            daily_mode: 是否走日频数据源路径；决定使用哪套实现指纹。

        返回：
            命中且校验通过的逐行因子值；缓存不存在或失效时返回 ``None``。
        """

        path = self._expression_path(node, input_fingerprint, daily_mode)
        return self._load_series(node.factor_id, path, daily)

    def _load_series(
        self,
        factor_name: str,
        path: Path,
        daily: pd.DataFrame,
    ) -> pd.Series | None:
        """按给定路径读取并严格校验一个因子缓存序列。

        参数：
            factor_name: 缓存值列使用的正式因子名或稳定表达式因子 ID。
            path: 已包含实现版本与行情指纹的 Parquet 缓存路径。
            daily: 用于校验缓存行数和主键的当前日频表。

        返回：
            命中且校验通过的逐行因子值；缓存不存在或失效时返回 ``None``。
        """

        if not path.exists():
            logger.debug("因子 %s 缓存不存在: %s", factor_name, path)
            return None
        try:
            cached = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("因子 %s 缓存读取失败，将重新计算: %s (%s)", factor_name, path, exc)
            return None
        expected_columns = [*KEY_COLUMNS, factor_name]
        if list(cached.columns) != expected_columns:
            logger.warning("因子 %s 缓存列校验失败，将重新计算: %s", factor_name, path)
            return None
        if len(cached) != len(daily):
            logger.warning(
                "因子 %s 缓存行数不匹配，将重新计算: cached=%d expected=%d",
                factor_name,
                len(cached),
                len(daily),
            )
            return None
        expected_keys = _normalized_keys(daily)
        cached_keys = _normalized_keys(cached)
        if not cached_keys.equals(expected_keys):
            different = (cached_keys != expected_keys).any(axis=1)
            first = int(different.idxmax()) if different.any() else -1
            logger.warning(
                "因子 %s 缓存日期或代码不匹配，将重新计算: %s first_difference=%d cached=%s expected=%s",
                factor_name,
                path,
                first,
                cached_keys.iloc[first].to_dict() if first >= 0 else None,
                expected_keys.iloc[first].to_dict() if first >= 0 else None,
            )
            return None
        logger.debug("因子 %s 缓存校验通过: %s", factor_name, path)
        return pd.Series(cached[factor_name].to_numpy(), index=daily.index, name=factor_name)

    def save(
        self,
        factory: FactorFactory,
        input_fingerprint: str,
        daily: pd.DataFrame,
        values: pd.Series,
        daily_mode: bool = False,
    ) -> Path:
        """先写临时 Parquet 再原子替换目标文件，并返回最终缓存路径。

        参数：
            factory: 决定因子名和实现版本的工厂实例。
            input_fingerprint: 当前分钟行情内容的稳定短指纹。
            daily: 要与缓存因子值共同写入的日频主键表。
            values: 与 ``daily`` 逐行对齐的因子值序列。
            daily_mode: 是否走日频数据源路径；决定使用哪套实现指纹。
        """

        path = self._path(factory, input_fingerprint, daily_mode)
        return self._save_series(factory.name, path, daily, values)

    def save_expression(
        self,
        node: ExpressionNode,
        input_fingerprint: str,
        daily: pd.DataFrame,
        values: pd.Series,
        daily_mode: bool = False,
    ) -> Path:
        """原子写入一个运行时表达式因子的持久化缓存。

        参数：
            node: 决定稳定因子 ID 和表达式实现版本的节点。
            input_fingerprint: 当前分钟行情内容的稳定短指纹。
            daily: 要与缓存因子值共同写入的日频主键表。
            values: 与 ``daily`` 逐行对齐的表达式因子值序列。
            daily_mode: 是否走日频数据源路径；决定使用哪套实现指纹。

        返回：
            最终写入的版本化 Parquet 缓存路径。
        """

        path = self._expression_path(node, input_fingerprint, daily_mode)
        return self._save_series(node.factor_id, path, daily, values)

    def _save_series(
        self,
        factor_name: str,
        path: Path,
        daily: pd.DataFrame,
        values: pd.Series,
    ) -> Path:
        """按给定路径原子写入一个因子缓存序列。

        参数：
            factor_name: 缓存值列使用的正式因子名或稳定表达式因子 ID。
            path: 已包含实现版本与行情指纹的 Parquet 缓存路径。
            daily: 要与缓存因子值共同写入的日频主键表。
            values: 与 ``daily`` 逐行对齐的因子值序列。

        返回：
            最终写入的版本化 Parquet 缓存路径。
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        frame = daily[KEY_COLUMNS].copy()
        frame[factor_name] = values.to_numpy()
        temporary = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        logger.debug("因子 %s 缓存已写入: %s", factor_name, path)
        return path

    def _path(
        self,
        factory: FactorFactory,
        input_fingerprint: str,
        daily_mode: bool = False,
    ) -> Path:
        """组合因子名称、实现指纹和输入指纹得到唯一缓存文件路径。

        参数：
            factory: 提供因子名和实现指纹的工厂实例。
            input_fingerprint: 用于区分行情版本的稳定短指纹。
            daily_mode: 是否使用日频实现指纹；缺省 ``False`` 保持分钟路径的
                历史缓存路径逐字节不变。
        """

        implementation = (
            _daily_implementation_fingerprint(factory)
            if daily_mode
            else _implementation_fingerprint(factory)
        )
        return self.root / factory.name / f"{implementation}-{input_fingerprint}.parquet"

    def _expression_path(
        self,
        node: ExpressionNode,
        input_fingerprint: str,
        daily_mode: bool = False,
    ) -> Path:
        """组合表达式 ID、实现指纹和输入指纹得到唯一缓存路径。

        参数：
            node: 提供稳定表达式因子 ID 和实现指纹的节点。
            input_fingerprint: 用于区分行情版本的稳定短指纹。
            daily_mode: 是否使用日频实现指纹；缺省 ``False`` 保持分钟路径的
                历史缓存路径逐字节不变。

        返回：
            该表达式因子当前实现和行情版本对应的 Parquet 路径。
        """

        implementation = (
            _daily_expression_implementation_fingerprint(node)
            if daily_mode
            else _expression_implementation_fingerprint(node)
        )
        return self.root / node.factor_id / f"{implementation}-{input_fingerprint}.parquet"


def _prepare_daily_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    """把外部日频行情整理成与分钟聚合结果同构的基础日线表。

    只保留 ``_build_daily_bars`` 会产出的那七列并按相同规则排序，让日频数据源
    与分钟数据源在因子工厂眼里完全一致。``amount``、``pre_close``、
    ``suspend_flag``、``adjust_factor`` 等额外列必须在这里丢掉，否则
    ``build_daily_features`` 的基础列守卫和每个表达式看到的列宇宙都会变。

    本函数刻意与 ``_build_daily_bars`` 分开实现：后者的源码文本参与因子缓存的
    实现指纹计算，改动它会作废全部既有缓存。

    参数：
        daily_bars: 已通过日频行情契约校验的表。

    返回：
        列为 ``code``、``trade_date``、``open``、``high``、``low``、``close``、
        ``volume``，并按证券与交易日升序排列的基础日线表。
    """

    columns = [*KEY_COLUMNS, "open", "high", "low", "close", "volume"]
    return (
        daily_bars.loc[:, columns]
        .sort_values(["code", "trade_date"])
        .reset_index(drop=True)
    )


def _daily_input_fingerprint(daily_bars: pd.DataFrame, namespace: str) -> str:
    """根据日频行情内容和数据源口径生成稳定的缓存输入指纹。

    指纹里必须拌进 ``namespace``：同一段日期在不复权与后复权两种口径下的取值
    不同，如果只哈希数值，切换 ``--adjust`` 后会命中上一次口径的缓存。

    参数：
        daily_bars: 已整理为基础七列的日频行情表。
        namespace: 数据源缓存命名空间，例如 ``qmt_daily/hfq``。

    返回：
        二十位十六进制短指纹。
    """

    columns = [*KEY_COLUMNS, "open", "high", "low", "close", "volume"]
    digest = hashlib.sha256()
    digest.update(namespace.encode())
    digest.update("|".join(f"{column}:{daily_bars[column].dtype}" for column in columns).encode())
    digest.update(
        pd.util.hash_pandas_object(daily_bars[columns], index=False).values.tobytes()
    )
    return digest.hexdigest()[:20]


def _daily_implementation_fingerprint(factory: FactorFactory) -> str:
    """为日频路径生成因子实现版本指纹。

    与分钟路径的 ``_implementation_fingerprint`` 分开实现，且哈希的是
    ``_prepare_daily_bars`` 的源码，因此两条路径的指纹天然不可能相同，
    即使同一个因子、同一份数据也会落在不同的缓存文件上。

    参数：
        factory: 要计算实现版本的具体正式因子工厂实例。

    返回：
        十六位十六进制短指纹。
    """

    digest = hashlib.sha256()
    module_path = Path(inspect.getfile(factory.__class__))
    base_path = Path(inspect.getfile(FactorFactory))
    digest.update(module_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(inspect.getsource(_prepare_daily_bars).encode())
    return digest.hexdigest()[:16]


def _daily_expression_implementation_fingerprint(node: ExpressionNode) -> str:
    """为日频路径生成表达式因子的实现版本指纹。

    与分钟路径的 ``_expression_implementation_fingerprint`` 分开实现，哈希的是
    ``_prepare_daily_bars`` 的源码与日频版的依赖因子指纹，因此两条路径的表达式
    缓存不可能互相命中。

    参数：
        node: 要标识实现版本的已校验运行时表达式节点。

    返回：
        十六位十六进制短指纹。
    """

    digest = hashlib.sha256(node.canonical.encode("utf-8"))
    dsl_directory = Path(inspect.getfile(DailyFactorFrame)).parent
    for module_path in sorted(dsl_directory.glob("*.py")):
        digest.update(module_path.name.encode("utf-8"))
        digest.update(module_path.read_bytes())
    digest.update(inspect.getsource(_prepare_daily_bars).encode("utf-8"))
    for dependency_name in sorted(node.columns.intersection(FACTOR_FACTORIES)):
        dependency = get_factor_factory(dependency_name)
        digest.update(dependency_name.encode("utf-8"))
        digest.update(_daily_implementation_fingerprint(dependency).encode("ascii"))
    return digest.hexdigest()[:16]


@log_elapsed(logger, "日频因子构建阶段")
def build_daily_features_from_daily(
    daily_bars: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    factor_expressions: Sequence[str | ExpressionNode] | None = None,
    cache_namespace: str = "",
    source_name: str = "daily",
) -> pd.DataFrame:
    """直接用日频行情构建因子，不需要分钟数据。

    依赖日内行情的因子无法在这条路径上计算：来自默认集合时会被自动剔除并告警，
    被显式点名时报错，绝不静默返回全 ``NaN``。

    参数：
        daily_bars: 已通过 ``validate_daily_bars`` 校验的日频行情表。
        feature_columns: 按顺序要计算的注册因子名；``None`` 表示使用该数据源
            支持的默认集合（即全部不依赖日内行情的因子）。
        cache_dir: 可选因子缓存根目录；为空时不读写缓存。
        factor_expressions: 运行时 DSL 因子表达式，按稳定因子 ID 追加为日频列。
        cache_namespace: 因子缓存命名空间，用于隔离数据源与复权口径。
        source_name: 数据源名，只用于生成可读的错误与告警信息。

    返回：
        按交易日与证券排序的日频因子表。
    """

    validated = validate_daily_bars(daily_bars)
    explicit = feature_columns is not None
    selected = list(feature_columns) if explicit else sorted(daily_capable_factor_names())
    expression_nodes = parse_factor_expressions(factor_expressions)
    if not selected and not expression_nodes:
        raise ValueError("至少需要选择一个因子")
    if len(selected) != len(set(selected)):
        raise ValueError("因子列表不能包含重复项")

    selected = _reject_or_drop_intraday_factors(selected, explicit, source_name)
    base_columns = {*KEY_COLUMNS, "open", "high", "low", "close", "volume"}
    referenced_columns = set().union(*(node.columns for node in expression_nodes)) if expression_nodes else set()
    unknown_columns = referenced_columns.difference(base_columns, FACTOR_FACTORIES)
    if unknown_columns:
        raise ValueError(f"运行时因子表达式引用未知列: {sorted(unknown_columns)}")
    referenced_intraday = sorted(referenced_columns.intersection(intraday_factor_names()))
    if referenced_intraday:
        raise ValueError(
            f"运行时因子表达式引用了需要分钟行情的因子 {referenced_intraday}；"
            f"数据源 {source_name!r} 只提供日频行情"
        )
    dependency_names = sorted(referenced_columns.intersection(FACTOR_FACTORIES) - set(selected))
    computed_names = [*selected, *dependency_names]
    factories = [get_factor_factory(name) for name in computed_names]

    daily = _prepare_daily_bars(validated)
    logger.info(
        "开始构建日频因子: registered=%d expressions=%d rows=%d symbols=%d cache=%s namespace=%s",
        len(factories),
        len(expression_nodes),
        len(daily),
        daily["code"].nunique(),
        cache_dir or "disabled",
        cache_namespace or "-",
    )
    cache = FactorCache(cache_dir, cache_namespace) if cache_dir is not None else None
    input_fingerprint = _daily_input_fingerprint(daily, cache_namespace) if cache is not None else ""

    for position, factory in enumerate(factories, start=1):
        values = (
            cache.load(factory, input_fingerprint, daily, daily_mode=True)
            if cache is not None
            else None
        )
        if values is None:
            logger.info("[%d/%d] 计算日频因子 %s", position, len(factories), factory.name)
            try:
                # 日频路径没有分钟行情；标错了 requires_intraday 的因子会在这里
                # 响亮地报错，而不是静默返回全 NaN。
                values = factory.compute(None, daily)
            except Exception:
                logger.exception(
                    "[%d/%d] 日频因子 %s 计算失败", position, len(factories), factory.name
                )
                raise
            if len(values) != len(daily):
                raise ValueError(f"因子 {factory.name} 返回了错误的行数")
            values = pd.Series(values.to_numpy(), index=daily.index, name=factory.name)
            if cache is not None:
                cache.save(factory, input_fingerprint, daily, values, daily_mode=True)
        else:
            logger.info("[%d/%d] 日频因子 %s 命中缓存", position, len(factories), factory.name)
        daily[factory.name] = values

    if expression_nodes:
        frame: DailyFactorFrame | None = None
        for position, node in enumerate(expression_nodes, start=1):
            values = (
                cache.load_expression(node, input_fingerprint, daily, daily_mode=True)
                if cache is not None
                else None
            )
            if values is None:
                logger.info(
                    "[%d/%d] 计算运行时表达式因子 %s: %s",
                    position,
                    len(expression_nodes),
                    node.factor_id,
                    node.canonical,
                )
                if frame is None:
                    frame = DailyFactorFrame(daily)
                values = frame.evaluate(node, name=node.factor_id)
                if cache is not None:
                    cache.save_expression(
                        node, input_fingerprint, daily, values, daily_mode=True
                    )
            else:
                logger.info(
                    "[%d/%d] 运行时表达式因子 %s 命中缓存",
                    position,
                    len(expression_nodes),
                    node.factor_id,
                )
            daily[node.factor_id] = values

    logger.info("全部日频因子构建完成: rows=%d", len(daily))
    return daily.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _reject_or_drop_intraday_factors(
    selected: list[str],
    explicit: bool,
    source_name: str,
) -> list[str]:
    """在日频路径上剔除或拒绝依赖分钟行情的因子。

    参数：
        selected: 本次选中的注册因子名列表。
        explicit: 因子集合是否由调用方显式指定。显式指定时不允许静默剔除，
            否则用户会拿到一份和自己要求不符、却毫无提示的结果。
        source_name: 数据源名，只用于生成可读的提示信息。

    返回：
        只含日频可算因子的新列表；显式点名了分钟因子时抛出 ``ValueError``。
    """

    intraday = intraday_factor_names()
    blocked = [name for name in selected if name in intraday]
    if not blocked:
        return list(selected)
    if explicit:
        raise ValueError(
            f"因子 {blocked} 需要分钟行情，数据源 {source_name!r} 只提供日频；"
            f"请改用 --data-source market_service，或从 --factors 中移除这些因子"
        )
    logger.warning(
        "数据源 %s 不提供分钟行情，已跳过 %d 个分钟因子: %s",
        source_name,
        len(blocked),
        ", ".join(blocked),
    )
    return [name for name in selected if name not in intraday]


@log_elapsed(logger, "因子构建阶段")
def build_daily_features(
    bars: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    factor_expressions: Sequence[str | ExpressionNode] | None = None,
) -> pd.DataFrame:
    """按运行时选择构建因子，并按实现和输入版本安全地复用缓存。

    参数：
        bars: 待校验、聚合并传给因子工厂的分钟行情表。
        feature_columns: 按顺序要计算的注册因子名；为 ``None`` 时使用默认集合，显式空序列非法。
        cache_dir: 可选因子缓存根目录；为空时不读写缓存。
        factor_expressions: 搜索输出的 DSL 字符串或节点；按稳定因子 ID 追加为日频列。
    """
    bars = validate_bars(bars)
    selected = list(feature_columns) if feature_columns is not None else DEFAULT_FEATURES.copy()
    expression_nodes = parse_factor_expressions(factor_expressions)
    if not selected and not expression_nodes:
        raise ValueError("至少需要选择一个因子")
    if len(selected) != len(set(selected)):
        raise ValueError("因子列表不能包含重复项")

    base_columns = {*KEY_COLUMNS, "open", "high", "low", "close", "volume"}
    referenced_columns = set().union(*(node.columns for node in expression_nodes))
    unknown_columns = referenced_columns.difference(base_columns, FACTOR_FACTORIES)
    if unknown_columns:
        raise ValueError(f"运行时因子表达式引用未知列: {sorted(unknown_columns)}")
    dependency_names = sorted(referenced_columns.intersection(FACTOR_FACTORIES) - set(selected))
    computed_names = [*selected, *dependency_names]
    factories = [get_factor_factory(name) for name in computed_names]
    logger.info(
        "开始构建日频因子: registered=%d expressions=%d bars=%d cache=%s",
        len(factories),
        len(expression_nodes),
        len(bars),
        cache_dir or "disabled",
    )
    daily = _build_daily_bars(bars)
    logger.info(
        "基础日线聚合完成: rows=%d symbols=%d",
        len(daily),
        daily["code"].nunique(),
    )
    cache = FactorCache(cache_dir) if cache_dir is not None else None
    input_fingerprint = _input_fingerprint(bars) if cache is not None else ""
    if cache is not None:
        logger.debug("输入行情指纹: %s", input_fingerprint)

    timings: list[tuple[str, str, float]] = []
    for position, factory in enumerate(factories, start=1):
        factor_started = perf_counter()
        logger.info("[%d/%d] 处理因子 %s", position, len(factories), factory.name)
        values = cache.load(factory, input_fingerprint, daily) if cache is not None else None
        if values is None:
            logger.info("[%d/%d] 计算因子 %s", position, len(factories), factory.name)
            try:
                values = factory.compute(bars, daily)
            except Exception:
                logger.exception("[%d/%d] 因子 %s 计算失败", position, len(factories), factory.name)
                raise
            if len(values) != len(daily):
                raise ValueError(f"因子 {factory.name} 返回了错误的行数")
            values = pd.Series(values.to_numpy(), index=daily.index, name=factory.name)
            if cache is not None:
                cache.save(factory, input_fingerprint, daily, values)
            mode = "computed"
            logger.info("[%d/%d] 因子 %s 计算完成", position, len(factories), factory.name)
        else:
            mode = "cached"
            logger.info("[%d/%d] 因子 %s 命中缓存", position, len(factories), factory.name)
        daily[factory.name] = values
        timings.append((factory.name, mode, perf_counter() - factor_started))

    if expression_nodes:
        frame: DailyFactorFrame | None = None
        for position, node in enumerate(expression_nodes, start=1):
            factor_started = perf_counter()
            logger.info(
                "[%d/%d] 处理运行时表达式因子 %s: %s",
                position,
                len(expression_nodes),
                node.factor_id,
                node.canonical,
            )
            values = (
                cache.load_expression(node, input_fingerprint, daily)
                if cache is not None
                else None
            )
            if values is None:
                logger.info(
                    "[%d/%d] 计算运行时表达式因子 %s",
                    position,
                    len(expression_nodes),
                    node.factor_id,
                )
                if frame is None:
                    frame = DailyFactorFrame(daily)
                values = frame.evaluate(node, name=node.factor_id)
                if cache is not None:
                    cache.save_expression(node, input_fingerprint, daily, values)
                mode = "computed"
                logger.info(
                    "[%d/%d] 运行时表达式因子 %s 计算完成",
                    position,
                    len(expression_nodes),
                    node.factor_id,
                )
            else:
                mode = "cached"
                logger.info(
                    "[%d/%d] 运行时表达式因子 %s 命中缓存",
                    position,
                    len(expression_nodes),
                    node.factor_id,
                )
            daily[node.factor_id] = values
            timings.append((node.factor_id, mode, perf_counter() - factor_started))

    slowest = sorted(timings, key=lambda item: item[2], reverse=True)
    logger.info(
        "因子耗时排行: %s",
        ", ".join(f"{name}={elapsed:.3f}s({mode})" for name, mode, elapsed in slowest),
    )
    logger.info(
        "全部因子构建完成: registered=%d expressions=%d rows=%d",
        len(factories),
        len(expression_nodes),
        len(daily),
    )
    return daily.sort_values(["trade_date", "code"]).reset_index(drop=True)
