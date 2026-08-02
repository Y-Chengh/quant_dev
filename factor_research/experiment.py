from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from time import perf_counter

import numpy as np
import pandas as pd

from .dataset import split_by_date
from .factors import DEFAULT_FEATURES
from .metrics import classification_metrics
from .tree import SimpleDecisionTreeClassifier
from .timing import log_elapsed


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    model: SimpleDecisionTreeClassifier
    feature_columns: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.Series


class DirectionExperiment:
    def __init__(
        self,
        validation_start: str | pd.Timestamp,
        feature_columns: list[str] | None = None,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        args: argparse.Namespace | None = None,
    ):
        self.validation_start = pd.Timestamp(validation_start)
        self.feature_columns = feature_columns or DEFAULT_FEATURES
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.args = args

    @log_elapsed(logger, "滚动训练验证")
    def run(self, dataset: pd.DataFrame) -> ExperimentResult:
        """Run expanding-window validation without using labels from the prediction date.

        Dates before validation_start form the initial history. For every target
        date T on or after it, a fresh model is fitted with target_date < T and
        then used to predict all symbols for T.
        """
        split = split_by_date(dataset, self.validation_start)
        validation_dates = pd.Index(split.validation["target_date"].drop_duplicates().sort_values())
        predictions, model, importance = self._walk_forward(dataset, validation_dates)

        return ExperimentResult(
            model=model,
            feature_columns=self.feature_columns,
            metrics=classification_metrics(predictions["label"], predictions["up_probability"]),
            predictions=predictions,
            feature_importance=pd.Series(importance, index=self.feature_columns).sort_values(ascending=False),
        )

    def _walk_forward(
        self,
        dataset: pd.DataFrame,
        prediction_dates: pd.Index,
    ) -> tuple[pd.DataFrame, SimpleDecisionTreeClassifier, np.ndarray]:
        predictions: list[pd.DataFrame] = []
        model: SimpleDecisionTreeClassifier | None = None
        importance = np.zeros(len(self.feature_columns), dtype=float)
        total_dates = len(prediction_dates)
        progress_interval = max(1, total_dates // 10)
        walk_started = perf_counter()
        preprocessing_elapsed = 0.0
        fitting_elapsed = 0.0
        prediction_elapsed = 0.0

        for position, target_date in enumerate(prediction_dates, start=1):
            logger.debug("滚动训练日期 [%d/%d]: %s", position, total_dates, pd.Timestamp(target_date).date())
            if position == 1 or position == total_dates or position % progress_interval == 0:
                logger.info(
                    "滚动验证进度 [%d/%d] %.1f%%",
                    position,
                    total_dates,
                    position / total_dates * 100,
                )
            # A label is available at T only after T closes, so training must end before T.
            train_frame = dataset.loc[dataset["target_date"] < target_date]
            predict_frame = dataset.loc[dataset["target_date"] == target_date]
            if train_frame.empty or predict_frame.empty:
                continue

            stage_started = perf_counter()
            medians = train_frame[self.feature_columns].median().fillna(0.0)
            model = SimpleDecisionTreeClassifier(self.max_depth, self.min_samples_leaf)
            train_matrix = self._matrix(train_frame, medians)
            predict_matrix = self._matrix(predict_frame, medians)
            preprocessing_elapsed += perf_counter() - stage_started
            stage_started = perf_counter()
            model.fit(
                train_matrix,
                train_frame["label"].to_numpy(dtype=int),
            )
            fitting_elapsed += perf_counter() - stage_started
            stage_started = perf_counter()
            probability = model.predict_proba(predict_matrix)[:, 1]
            prediction_elapsed += perf_counter() - stage_started
            daily = predict_frame[["feature_date", "target_date", "code", "label", "target_return"]].copy()
            daily["up_probability"] = probability
            daily["prediction"] = (probability >= 0.5).astype(int)
            daily["training_samples"] = len(train_frame)
            daily["training_end_date"] = train_frame["target_date"].max()
            predictions.append(daily)
            importance += model.feature_importances_

        if model is None or not predictions:
            raise ValueError("没有足够的数据执行滚动验证")
        total_elapsed = perf_counter() - walk_started
        logger.info(
            "滚动验证耗时汇总: dates=%d total=%.3fs preprocessing=%.3fs fit=%.3fs predict=%.3fs avg_per_date=%.3fs",
            total_dates,
            total_elapsed,
            preprocessing_elapsed,
            fitting_elapsed,
            prediction_elapsed,
            total_elapsed / total_dates,
        )
        importance /= len(predictions)
        total_importance = importance.sum()
        if total_importance > 0:
            importance /= total_importance
        return pd.concat(predictions, ignore_index=True), model, importance

    def _matrix(self, frame: pd.DataFrame, medians: pd.Series) -> np.ndarray:
        # Fit missing-value replacements on the currently available history only.
        return (
            frame[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
            .to_numpy(dtype=float)
        )
