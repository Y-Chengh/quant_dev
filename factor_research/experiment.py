from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from .dataset import REPORT_RETURN_COLUMNS, split_by_date
from .factors import DEFAULT_FEATURES
from .metrics import (
    classification_metrics,
    cross_sectional_ic_metrics,
    daily_accuracy_trend,
    daily_cross_sectional_ic,
    regression_metrics,
)
from .models.base import DirectionModel, DirectionModelFactory
from .models.simple_decision_tree import SimpleDecisionTreeModelFactory
from .timing import ElapsedRecorder, log_elapsed


logger = logging.getLogger(__name__)
TRAINING_MODES = ("rolling", "single")
PREDICTION_TASKS = ("classification", "regression")


@dataclass
class ExperimentResult:
    model: DirectionModel
    model_name: str
    feature_columns: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame
    feature_importance: pd.Series | None
    daily_accuracy_trend: pd.DataFrame
    task: str = "classification"
    daily_ic_trend: pd.DataFrame = field(default_factory=pd.DataFrame)


class DirectionExperiment:
    def __init__(
        self,
        validation_start: str | pd.Timestamp,
        feature_columns: list[str] | None = None,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        args: argparse.Namespace | None = None,
        model_factory: DirectionModelFactory | None = None,
        training_mode: str = "rolling",
        task: str = "classification",
    ):
        if training_mode not in TRAINING_MODES:
            raise ValueError(
                f"training_mode 必须是 {TRAINING_MODES} 之一，实际为 {training_mode!r}"
            )
        if task not in PREDICTION_TASKS:
            raise ValueError(
                f"task 必须是 {PREDICTION_TASKS} 之一，实际为 {task!r}"
            )
        self.validation_start = pd.Timestamp(validation_start)
        self.feature_columns = feature_columns or DEFAULT_FEATURES
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.args = args
        self.training_mode = training_mode
        self.task = task
        # 保留原有树参数作为默认配置；注入工厂后，实验流程不再关心具体算法。
        self.model_factory = (
            model_factory
            if model_factory is not None
            else SimpleDecisionTreeModelFactory(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            )
        )
        if task not in self.model_factory.supported_tasks:
            raise ValueError(
                f"模型 {self.model_factory.name!r} 不支持任务 {task!r}；"
                f"支持的任务为 {self.model_factory.supported_tasks}"
            )
        factory_task = getattr(self.model_factory, "task", None)
        if factory_task is not None and factory_task != task:
            raise ValueError(
                f"实验任务 {task!r} 与模型工厂任务 {factory_task!r} 不一致"
            )

    @log_elapsed(logger, "模型训练验证")
    def run(self, dataset: pd.DataFrame) -> ExperimentResult:
        """按配置执行扩展窗口滚动验证或固定训练集验证。

        ``validation_start`` 之前的日期构成固定训练集。滚动模式会在每个预测日
        使用 ``target_date < T`` 的全部样本重新训练；单次模式仅使用固定训练集
        拟合一次，并预测整个验证集。
        """
        dataset = self._filter_required_finite_features(dataset)
        split = split_by_date(dataset, self.validation_start)
        validation_dates = pd.Index(split.validation["target_date"].drop_duplicates().sort_values())
        if self.training_mode == "rolling":
            predictions, model, importance = self._walk_forward(
                dataset, validation_dates
            )
        else:
            predictions, model, importance = self._single_fit(
                split.train, split.validation
            )

        score_column = (
            "up_probability"
            if self.task == "classification"
            else "predicted_return"
        )
        metrics = (
            classification_metrics(
                predictions["label"],
                predictions["up_probability"],
                predictions["target_return"],
            )
            if self.task == "classification"
            else regression_metrics(
                predictions["target_return"],
                predictions["predicted_return"],
            )
        )
        metrics["pooled_ic"] = metrics.pop("ic")
        daily_ic = daily_cross_sectional_ic(predictions, score_column)
        metrics.update(cross_sectional_ic_metrics(daily_ic))
        return ExperimentResult(
            model=model,
            model_name=self.model_factory.name,
            feature_columns=self.feature_columns,
            metrics=metrics,
            predictions=predictions,
            feature_importance=(
                None
                if importance is None
                else pd.Series(importance, index=self.feature_columns).sort_values(
                    ascending=False
                )
            ),
            daily_accuracy_trend=daily_accuracy_trend(predictions),
            task=self.task,
            daily_ic_trend=daily_ic,
        )

    def _filter_required_finite_features(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """删除模型声明不能填充的非有限特征样本。

        普通模型仍由矩阵预处理统一填充缺失值。模型工厂可通过
        ``required_finite_feature_indices`` 声明必须保留原始有限值的列；例如
        因子直出模型声明末列后，搜索和主实验都会排除同一批缺失候选样本。

        参数：
            dataset: 含全部模型特征、目标和日期的待切分样本表。

        返回：
            保留声明特征均为有限数值的副本；没有声明时原样返回输入表。
        """

        indices = tuple(
            getattr(self.model_factory, "required_finite_feature_indices", ())
        )
        if not indices:
            return dataset
        feature_count = len(self.feature_columns)
        normalized: list[int] = []
        for index in indices:
            resolved = index if index >= 0 else feature_count + index
            if resolved < 0 or resolved >= feature_count:
                raise ValueError(
                    "模型必须有限的特征索引越界: "
                    f"index={index} feature_count={feature_count}"
                )
            normalized.append(resolved)
        columns = [self.feature_columns[index] for index in normalized]
        values = dataset.loc[:, columns].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        dropped = int((~finite).sum())
        if dropped:
            logger.info(
                "按模型有限值约束过滤样本: features=%s dropped=%d retained=%d",
                columns,
                dropped,
                int(finite.sum()),
            )
        return dataset.loc[finite].copy()

    def _target(self, frame: pd.DataFrame) -> np.ndarray:
        """按任务选择训练目标；两种目标都只属于对应的 target_date。"""
        if self.task == "classification":
            return frame["label"].to_numpy(dtype=int)
        return frame["target_return"].to_numpy(dtype=float)

    @staticmethod
    def _prediction_columns(frame: pd.DataFrame) -> list[str]:
        """返回预测结果需保留的基础列和可用报告上下文列。

        参数：
            frame: 待预测样本；可包含目标日及前一交易日收益口径的报告上下文列。

        返回：
            基础标识、标签、目标收益，以及输入中存在的报告收益上下文列名。
        """

        base_columns = [
            "feature_date",
            "target_date",
            "code",
            "label",
            "target_return",
        ]
        return [
            *base_columns,
            *(column for column in REPORT_RETURN_COLUMNS if column in frame.columns),
        ]

    def _add_model_predictions(
        self,
        predictions: pd.DataFrame,
        model: DirectionModel,
        matrix: np.ndarray,
        timings: ElapsedRecorder,
    ) -> None:
        if self.task == "classification":
            probability = timings.track("predict")(model.predict_proba)(matrix)[:, 1]
            predictions["up_probability"] = probability
            predictions["prediction"] = (probability >= 0.5).astype(int)
            return
        predicted_return = np.asarray(
            timings.track("predict")(model.predict)(matrix), dtype=float
        )
        if predicted_return.shape != (len(predictions),):
            raise ValueError(
                "回归模型预测形状不正确: "
                f"expected={(len(predictions),)} actual={predicted_return.shape}"
            )
        if not np.isfinite(predicted_return).all():
            raise ValueError("回归模型预测包含 NaN 或无穷值")
        predictions["predicted_return"] = predicted_return
        predictions["prediction"] = (predicted_return > 0).astype(int)

    def _single_fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, DirectionModel, np.ndarray | None]:
        """只使用验证起始日前的训练集拟合一次，并预测完整验证集。"""
        timings = ElapsedRecorder()
        nan_fill_value = -10000.0
        model = self.model_factory.create()
        train_matrix = timings.track("preprocessing")(self._matrix)(
            train_frame, nan_fill_value
        )
        validation_matrix = timings.track("preprocessing")(self._matrix)(
            validation_frame, nan_fill_value
        )
        timings.track("fit")(model.fit)(
            train_matrix,
            self._target(train_frame),
        )

        predictions = validation_frame[
            self._prediction_columns(validation_frame)
        ].copy()
        self._add_model_predictions(predictions, model, validation_matrix, timings)
        predictions["training_samples"] = len(train_frame)
        predictions["training_end_date"] = train_frame["target_date"].max()
        importance = getattr(model, "feature_importances_", None)
        if importance is not None:
            importance = np.asarray(importance, dtype=float)
            expected_shape = (len(self.feature_columns),)
            if importance.shape != expected_shape:
                raise ValueError(
                    "模型特征重要度形状不正确: "
                    f"expected={expected_shape} actual={importance.shape}"
                )
            if not np.isfinite(importance).all():
                raise ValueError("模型特征重要度包含 NaN 或无穷值")
            if importance.sum() > 0:
                importance = importance / importance.sum()
        logger.info(
            "单次训练验证耗时汇总: train_samples=%d validation_samples=%d "
            "total=%.3fs preprocessing=%.3fs fit=%.3fs predict=%.3fs",
            len(train_frame),
            len(validation_frame),
            timings.total,
            timings.elapsed("preprocessing"),
            timings.elapsed("fit"),
            timings.elapsed("predict"),
        )
        return predictions.reset_index(drop=True), model, importance

    def _walk_forward(
        self,
        dataset: pd.DataFrame,
        prediction_dates: pd.Index,
    ) -> tuple[pd.DataFrame, DirectionModel, np.ndarray | None]:
        predictions: list[pd.DataFrame] = []
        model: DirectionModel | None = None
        importance_sum: np.ndarray | None = None
        importance_count = 0
        total_dates = len(prediction_dates)
        progress_interval = max(1, total_dates // 10)
        timings = ElapsedRecorder()

        @timings.track("preprocessing")
        def preprocess(
            train_frame: pd.DataFrame,
            predict_frame: pd.DataFrame,
        ) -> tuple[DirectionModel, np.ndarray, np.ndarray]:
            # TODO: nan怎么处理要好好想想
            # nan_fill_value = train_frame[self.feature_columns].median().fillna(0.0)
            nan_fill_value = -10000.0

            current_model = self.model_factory.create()
            return (
                current_model,
                self._matrix(train_frame, nan_fill_value),
                self._matrix(predict_frame, nan_fill_value),
            )

        # if getattr(self.args, "debug", False):
        #     logger.debug("DEBUG: %s", prediction_dates)
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

            if logger.isEnabledFor(logging.DEBUG):
                # 监控na占比
                train_features = train_frame[self.feature_columns]
                predict_features = predict_frame[self.feature_columns]
                train_na_by_feature = train_features.isna().sum()
                predict_na_by_feature = predict_features.isna().sum()
                train_na_count = int(train_na_by_feature.sum())
                predict_na_count = int(predict_na_by_feature.sum())
                logger.debug(
                    "因子 NA 监控: target_date=%s train=%d/%d (%.2f%%) predict=%d/%d (%.2f%%) "
                    "train_by_feature=%s predict_by_feature=%s",
                    pd.Timestamp(target_date).date(),
                    train_na_count,
                    train_features.size,
                    train_na_count / train_features.size * 100,
                    predict_na_count,
                    predict_features.size,
                    predict_na_count / predict_features.size * 100,
                    train_na_by_feature[train_na_by_feature > 0].to_dict(),
                    predict_na_by_feature[predict_na_by_feature > 0].to_dict(),
                )

            model, train_matrix, predict_matrix = preprocess(train_frame, predict_frame)
            timings.track("fit")(model.fit)(
                train_matrix,
                self._target(train_frame),
            )
            daily = predict_frame[self._prediction_columns(predict_frame)].copy()
            self._add_model_predictions(daily, model, predict_matrix, timings)
            daily["training_samples"] = len(train_frame)
            daily["training_end_date"] = train_frame["target_date"].max()
            predictions.append(daily)
            model_importance = getattr(model, "feature_importances_", None)
            logger.debug("model.feature_importances_: %s", model_importance)
            if model_importance is not None:
                model_importance = np.asarray(model_importance, dtype=float)
                expected_shape = (len(self.feature_columns),)
                if model_importance.shape != expected_shape:
                    raise ValueError(
                        "模型特征重要度形状不正确: "
                        f"expected={expected_shape} actual={model_importance.shape}"
                    )
                if not np.isfinite(model_importance).all():
                    raise ValueError("模型特征重要度包含 NaN 或无穷值")
                if importance_sum is None:
                    importance_sum = np.zeros(expected_shape, dtype=float)
                importance_sum += model_importance
                importance_count += 1

        if model is None or not predictions:
            raise ValueError("没有足够的数据执行滚动验证")
        total_elapsed = timings.total
        logger.info(
            "滚动验证耗时汇总: dates=%d total=%.3fs preprocessing=%.3fs fit=%.3fs predict=%.3fs avg_per_date=%.3fs",
            total_dates,
            total_elapsed,
            timings.elapsed("preprocessing"),
            timings.elapsed("fit"),
            timings.elapsed("predict"),
            total_elapsed / total_dates,
        )
        importance = None
        if importance_sum is not None:
            importance = importance_sum / importance_count
            total_importance = importance.sum()
            if total_importance > 0:
                importance /= total_importance
        return pd.concat(predictions, ignore_index=True), model, importance

    def _matrix(
        self,
        frame: pd.DataFrame,
        medians: pd.Series | float,
    ) -> np.ndarray:
        # Fit missing-value replacements on the currently available history only.
        return (
            frame[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(medians)
            .to_numpy(dtype=float)
        )
