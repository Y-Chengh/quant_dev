from __future__ import annotations

import hashlib
import inspect
import logging
from time import perf_counter
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import validate_bars
from .factor_factories import FACTOR_FACTORIES, get_factor_factory
from .factor_factories.base import FactorFactory
from .timing import log_elapsed


logger = logging.getLogger(__name__)
DEFAULT_FEATURES = sorted(FACTOR_FACTORIES)
KEY_COLUMNS = ["code", "trade_date"]


def available_factors() -> list[str]:
    return sorted(FACTOR_FACTORIES)


def _normalized_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Parquet/pandas dtype differences before comparing logical keys."""
    keys = frame[KEY_COLUMNS].copy().reset_index(drop=True)
    keys["code"] = keys["code"].astype(str)
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="raise").astype("datetime64[ns]")
    return keys


def _build_daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
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


def _input_fingerprint(bars: pd.DataFrame) -> str:
    columns = ["code", "trade_time", "open", "high", "low", "close", "volume"]
    digest = hashlib.sha256()
    digest.update("|".join(f"{column}:{bars[column].dtype}" for column in columns).encode())
    digest.update(pd.util.hash_pandas_object(bars[columns], index=False).values.tobytes())
    return digest.hexdigest()[:20]


def _implementation_fingerprint(factory: FactorFactory) -> str:
    digest = hashlib.sha256()
    # Include the concrete factory, shared helpers, and base daily aggregation.
    module_path = Path(inspect.getfile(factory.__class__))
    base_path = Path(inspect.getfile(FactorFactory))
    digest.update(module_path.read_bytes())
    digest.update(base_path.read_bytes())
    digest.update(inspect.getsource(_build_daily_bars).encode())
    return digest.hexdigest()[:16]


class FactorCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(
        self,
        factory: FactorFactory,
        input_fingerprint: str,
        daily: pd.DataFrame,
    ) -> pd.Series | None:
        path = self._path(factory, input_fingerprint)
        if not path.exists():
            logger.debug("因子 %s 缓存不存在: %s", factory.name, path)
            return None
        try:
            cached = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("因子 %s 缓存读取失败，将重新计算: %s (%s)", factory.name, path, exc)
            return None
        expected_columns = [*KEY_COLUMNS, factory.name]
        if list(cached.columns) != expected_columns:
            logger.warning("因子 %s 缓存列校验失败，将重新计算: %s", factory.name, path)
            return None
        if len(cached) != len(daily):
            logger.warning(
                "因子 %s 缓存行数不匹配，将重新计算: cached=%d expected=%d",
                factory.name,
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
                factory.name,
                path,
                first,
                cached_keys.iloc[first].to_dict() if first >= 0 else None,
                expected_keys.iloc[first].to_dict() if first >= 0 else None,
            )
            return None
        logger.debug("因子 %s 缓存校验通过: %s", factory.name, path)
        return pd.Series(cached[factory.name].to_numpy(), index=daily.index, name=factory.name)

    def save(
        self,
        factory: FactorFactory,
        input_fingerprint: str,
        daily: pd.DataFrame,
        values: pd.Series,
    ) -> Path:
        path = self._path(factory, input_fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = daily[KEY_COLUMNS].copy()
        frame[factory.name] = values.to_numpy()
        temporary = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        logger.debug("因子 %s 缓存已写入: %s", factory.name, path)
        return path

    def _path(self, factory: FactorFactory, input_fingerprint: str) -> Path:
        implementation = _implementation_fingerprint(factory)
        return self.root / factory.name / f"{implementation}-{input_fingerprint}.parquet"


@log_elapsed(logger, "因子构建阶段")
def build_daily_features(
    bars: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """按运行时选择构建因子，并按实现和输入版本安全地复用缓存。"""
    bars = validate_bars(bars)
    selected = list(feature_columns) if feature_columns is not None else DEFAULT_FEATURES.copy()
    if not selected:
        raise ValueError("至少需要选择一个因子")
    if len(selected) != len(set(selected)):
        raise ValueError("因子列表不能包含重复项")

    factories = [get_factor_factory(name) for name in selected]
    logger.info("开始构建日频因子: factors=%d bars=%d cache=%s", len(factories), len(bars), cache_dir or "disabled")
    stage_started = perf_counter()
    daily = _build_daily_bars(bars)
    logger.info(
        "基础日线聚合完成: rows=%d symbols=%d elapsed=%.3fs",
        len(daily), daily["code"].nunique(), perf_counter() - stage_started,
    )
    cache = FactorCache(cache_dir) if cache_dir is not None else None
    stage_started = perf_counter()
    input_fingerprint = _input_fingerprint(bars) if cache is not None else ""
    if cache is not None:
        logger.debug("输入行情指纹: %s elapsed=%.3fs", input_fingerprint, perf_counter() - stage_started)

    timings: list[tuple[str, str, float]] = []
    for position, factory in enumerate(factories, start=1):
        factor_started = perf_counter()
        logger.info("[%d/%d] 处理因子 %s", position, len(factories), factory.name)
        cache_started = perf_counter()
        values = cache.load(factory, input_fingerprint, daily) if cache is not None else None
        cache_elapsed = perf_counter() - cache_started
        if values is None:
            logger.info("[%d/%d] 计算因子 %s", position, len(factories), factory.name)
            compute_started = perf_counter()
            try:
                values = factory.compute(bars, daily)
            except Exception:
                logger.exception("[%d/%d] 因子 %s 计算失败", position, len(factories), factory.name)
                raise
            compute_elapsed = perf_counter() - compute_started
            if len(values) != len(daily):
                raise ValueError(f"因子 {factory.name} 返回了错误的行数")
            values = pd.Series(values.to_numpy(), index=daily.index, name=factory.name)
            if cache is not None:
                save_started = perf_counter()
                cache.save(factory, input_fingerprint, daily, values)
                logger.debug("因子 %s 缓存写入耗时 %.3fs", factory.name, perf_counter() - save_started)
            mode = "computed"
            logger.info("[%d/%d] 因子 %s 计算完成 elapsed=%.3fs", position, len(factories), factory.name, compute_elapsed)
        else:
            mode = "cached"
            logger.info("[%d/%d] 因子 %s 命中缓存 load=%.3fs", position, len(factories), factory.name, cache_elapsed)
        daily[factory.name] = values
        timings.append((factory.name, mode, perf_counter() - factor_started))

    slowest = sorted(timings, key=lambda item: item[2], reverse=True)
    logger.info(
        "因子耗时排行: %s",
        ", ".join(f"{name}={elapsed:.3f}s({mode})" for name, mode, elapsed in slowest),
    )
    logger.info("全部因子构建完成: factors=%d rows=%d", len(factories), len(daily))
    return daily.sort_values(["trade_date", "code"]).reset_index(drop=True)
