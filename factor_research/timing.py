from __future__ import annotations

import logging
from collections import defaultdict
from functools import wraps
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


class ElapsedRecorder:
    """Accumulate durations for several decorated operations."""

    def __init__(self) -> None:
        self._started = perf_counter()
        self._elapsed: defaultdict[str, float] = defaultdict(float)

    def track(self, operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorate(function: Callable[P, R]) -> Callable[P, R]:
            @wraps(function)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                started = perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    self._elapsed[operation] += perf_counter() - started

            return wrapper

        return decorate

    def elapsed(self, operation: str) -> float:
        return self._elapsed[operation]

    @property
    def total(self) -> float:
        return perf_counter() - self._started


def log_elapsed(
    logger: logging.Logger,
    label: str | None = None,
    level: int = logging.INFO,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log total function duration while preserving its signature and return value."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        operation = label or function.__qualname__

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            started = perf_counter()
            try:
                result = function(*args, **kwargs)
            except Exception:
                logger.exception("%s失败: elapsed=%.3fs", operation, perf_counter() - started)
                raise
            logger.log(level, "%s完成: elapsed=%.3fs", operation, perf_counter() - started)
            return result

        return wrapper

    return decorate
