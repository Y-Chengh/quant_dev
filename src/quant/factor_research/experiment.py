from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dataset import (
    DEFAULT_LABEL_RETURN_THRESHOLD,
    REPORT_RETURN_COLUMNS,
    returns_exceed_label_threshold,
    split_by_date,
    validate_label_return_threshold,
)
from .factors import DEFAULT_FEATURES
from .metrics import (
    classification_metrics,
    cross_sectional_ic_metrics,
    daily_accuracy_trend,
    daily_cross_sectional_ic,
    regression_metrics,
)
from .models.base import (
    DirectionModel,
    DirectionModelFactory,
    FitProgressCallback,
)
from .models.simple_decision_tree import SimpleDecisionTreeModelFactory
from .progress import DEFAULT_MIN_INTERVAL, PROGRESS_MODES, ProgressBar
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
    label_return_threshold: float = DEFAULT_LABEL_RETURN_THRESHOLD


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
        progress: str = "never",
        label_return_threshold: float = DEFAULT_LABEL_RETURN_THRESHOLD,
    ):
        """保存训练验证配置，并校验训练方式、任务与模型工厂的兼容性。

        参数：
            validation_start: 验证集首个目标日期，该日期本身属于验证区间。
            feature_columns: 参与建模的特征列名；缺省使用 ``DEFAULT_FEATURES``。
            max_depth: 缺省决策树模型的最大深度；注入 ``model_factory`` 后不再生效。
            min_samples_leaf: 缺省决策树模型的叶子最小样本数；同上。
            args: 命令行参数命名空间，仅用于透传调试开关，不参与建模口径。
            model_factory: 模型工厂；缺省构造简单决策树工厂，注入后实验不关心具体算法。
            training_mode: ``rolling`` 为逐日扩展窗口重训，``single`` 为固定训练集只拟合一次。
            task: ``classification`` 为收益阈值二分类，``regression`` 为连续涨跌幅。
            progress: 训练验证进度条模式，取 :data:`PROGRESS_MODES` 之一。缺省 ``never``
                保持库层调用（含搜索的并行 worker）静默，由命令行显式开启为 ``auto``。
            label_return_threshold: 正类标签对应的最低目标收益率，单位为一；目标收益率
                严格大于该值才算达标，缺省 ``0.005`` 表示 0.5%。

        返回：
            无返回值；任一配置非法时直接抛出 ``ValueError``。
        """

        if training_mode not in TRAINING_MODES:
            raise ValueError(
                f"training_mode 必须是 {TRAINING_MODES} 之一，实际为 {training_mode!r}"
            )
        if task not in PREDICTION_TASKS:
            raise ValueError(
                f"task 必须是 {PREDICTION_TASKS} 之一，实际为 {task!r}"
            )
        if progress not in PROGRESS_MODES:
            raise ValueError(
                f"progress 必须是 {PROGRESS_MODES} 之一，实际为 {progress!r}"
            )
        self.validation_start = pd.Timestamp(validation_start)
        self.feature_columns = feature_columns or DEFAULT_FEATURES
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.args = args
        self.training_mode = training_mode
        self.task = task
        self.progress = progress
        self.label_return_threshold = validate_label_return_threshold(
            label_return_threshold
        )
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
        使用 ``target_end_date < T`` 的全部已实现收益样本重新训练；单次模式仅使用
        在验证起点前已实现收益的固定训练集拟合一次，并预测整个验证集。
        """
        if "target_end_date" not in dataset.columns:
            dataset = dataset.copy()
            dataset["target_end_date"] = dataset["target_date"]
        self._validate_label_consistency(dataset)
        dataset = self._filter_required_finite_features(dataset)
        split = split_by_date(dataset, self.validation_start)
        validation_dates = pd.Index(split.validation["target_date"].drop_duplicates().sort_values())
        if self.training_mode == "rolling":
            predictions, model, importance = self._walk_forward(
                dataset, validation_dates
            )
        else:
            predictions, model, importance = self._single_fit(
                split.train.loc[
                    split.train["target_end_date"] < self.validation_start
                ],
                split.validation,
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
            label_return_threshold=self.label_return_threshold,
        )

    def _validate_label_consistency(self, dataset: pd.DataFrame) -> None:
        """拒绝与本次收益阈值不一致或不属于二元集合的标签。

        参数：
            dataset: 含连续目标收益率及预先生成二分类标签的监督学习样本；标签必须
                等于目标收益率严格大于 ``label_return_threshold`` 的比较结果。

        返回：
            无返回值；发现非法标签或阈值口径不一致时抛出 ``ValueError``。
        """

        required = {"target_return", "label"}
        missing = required.difference(dataset.columns)
        if missing:
            raise ValueError(f"数据集缺少目标列: {sorted(missing)}")
        target_return = pd.to_numeric(dataset["target_return"], errors="coerce")
        finite_target = np.isfinite(target_return.to_numpy(dtype=float))
        if not finite_target.all():
            raise ValueError(
                "数据集 target_return 必须全部是有限数值: "
                f"invalid_rows={int((~finite_target).sum())}"
            )
        label = pd.to_numeric(dataset["label"], errors="coerce")
        expected = returns_exceed_label_threshold(
            target_return,
            self.label_return_threshold,
        ).astype(int)
        inconsistent = label.isna() | ~label.isin((0, 1)) | label.ne(expected)
        if inconsistent.any():
            raise ValueError(
                "数据集 label 与 label_return_threshold 不一致: "
                f"threshold={self.label_return_threshold} "
                f"mismatched_rows={int(inconsistent.sum())}"
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

    def _create_progress(
        self,
        total: int,
        description: str,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> ProgressBar:
        """按显示模式和当前日志等级创建本阶段的进度条。

        参数：
            total: 本阶段的总步数，例如滚动模式的验证日数量。
            description: 进度条行首的阶段名称，例如 ``滚动验证``。
            min_interval: 两次重绘的最小间隔秒数；步数很少的阶段传 0 以保证每一步
                都立即显示，逐日循环则用默认节流避免刷屏。

        返回：
            已解析好是否输出的进度条。DEBUG 等级下强制关闭：该等级会逐日写调试
            日志，日志与进度帧写同一个流，会不断打断同一行的进度条。
        """

        return ProgressBar(
            total,
            description,
            mode=self._progress_mode(),
            min_interval=min_interval,
        )

    def _progress_mode(self) -> str:
        """返回本次运行实际生效的进度显示模式。

        返回：
            :data:`PROGRESS_MODES` 之一。DEBUG 等级下一律返回 ``never``：该等级会
            逐日写调试日志，日志与进度帧写同一个流，会不断打断同一行的进度条。
        """

        return "never" if logger.isEnabledFor(logging.DEBUG) else self.progress

    def _install_fit_progress(
        self,
        model: DirectionModel,
        callback: FitProgressCallback | None,
    ) -> int:
        """给支持逐轮上报的模型安装训练进度回调。

        参数：
            model: 本次训练使用的模型实例。
            callback: 按轮接收训练进度的回调；传 ``None`` 表示卸载。

        返回：
            模型声明的训练总轮数；模型未实现该可选能力、卸载回调，或声明的轮数
            不是正整数时返回 ``0``，此时整段训练在进度条上只占一步。该能力只用于
            显示，因此声明值异常时降级而不是让训练失败。
        """

        setter = getattr(model, "set_fit_progress", None)
        if not callable(setter):
            return 0
        units = setter(callback)
        if units is None:
            return 0
        try:
            rounds = int(units)
        except (TypeError, ValueError, OverflowError):
            # 进度显示不值得让一次训练失败，声明值不可用时按不支持处理。
            return 0
        return rounds if rounds >= 1 else 0

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
            "target_end_date",
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
        predictions["prediction"] = returns_exceed_label_threshold(
            predicted_return,
            self.label_return_threshold,
        ).astype(int)

    def _single_fit(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, DirectionModel, np.ndarray | None]:
        """只使用验证起始日前的训练集拟合一次，并预测完整验证集。"""
        timings = ElapsedRecorder()
        nan_fill_value = -10000.0
        model = self.model_factory.create()
        progress: ProgressBar | None = None
        fit_units = 0

        def report_fit(completed: int, total: int) -> None:
            """把模型上报的已完成训练轮数映射为进度条步数。

            参数：
                completed: 模型已完成的迭代轮数，从 1 起计。
                total: 模型声明的本次训练总轮数，只用于行尾文本。

            返回：
                无返回值。该闭包只在 ``model.fit`` 执行期间被调用，此时进度条与
                训练刻度都已就绪；训练段固定占用进度条的第 2 步起共 ``fit_units`` 步。
            """

            if progress is None:
                return
            progress.advance_to(
                1 + min(int(completed), fit_units),
                detail=f"训练 {int(completed)}/{int(total)} 轮",
            )

        # 训练是单次模式里唯一不可细分的长阶段：能逐轮上报的模型按轮推进，
        # 其余模型整段训练只占一步，退回预处理、训练、预测三阶段的粒度。
        # 完全关闭进度时连回调都不装，让训练调用与不带该能力时逐字节一致；
        # auto 模式下即使最终判定为非终端，回调也只是空转，开销可忽略。
        reports_progress = self._progress_mode() != "never"
        fit_units = (
            self._install_fit_progress(model, report_fit) if reports_progress else 0
        )
        # 逐轮推进时帧数很多，用默认节流；只有三步时关掉节流，让每步立即可见。
        with self._create_progress(
            2 + max(fit_units, 1),
            "单次训练验证",
            min_interval=DEFAULT_MIN_INTERVAL if fit_units > 1 else 0.0,
        ) as bar:
            progress = bar
            try:
                train_matrix = timings.track("preprocessing")(self._matrix)(
                    train_frame, nan_fill_value
                )
                validation_matrix = timings.track("preprocessing")(self._matrix)(
                    validation_frame, nan_fill_value
                )
                bar.advance(detail="预处理")
                # 预处理是整表操作，单步耗时与单轮提升差一个数量级，
                # 用它外推剩余时间会得到离谱的结果。
                bar.rebase_eta()
                timings.track("fit")(model.fit)(
                    train_matrix,
                    self._target(train_frame),
                )
            finally:
                # 结果里会带回该模型，不能让它长期持有进度条闭包；没装过就不去动它。
                # 卸载只是清理，失败不得掩盖训练本身抛出的异常。
                if reports_progress:
                    try:
                        self._install_fit_progress(model, None)
                    except Exception:
                        logger.debug("卸载训练进度回调失败", exc_info=True)
            # 模型可能提前收敛或走了常数目标分支而没报满轮数，这里补齐训练段。
            bar.advance_to(1 + max(fit_units, 1), detail="训练")

            predictions = validation_frame[
                self._prediction_columns(validation_frame)
            ].copy()
            self._add_model_predictions(predictions, model, validation_matrix, timings)
            bar.advance(detail="预测")
        predictions["training_samples"] = len(train_frame)
        predictions["training_end_date"] = train_frame["target_end_date"].max()
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
        with self._create_progress(total_dates, "滚动验证") as progress:
            for position, target_date in enumerate(prediction_dates, start=1):
                date_text = str(pd.Timestamp(target_date).date())
                logger.debug("滚动训练日期 [%d/%d]: %s", position, total_dates, date_text)
                milestone = (
                    position == 1
                    or position == total_dates
                    or position % progress_interval == 0
                )
                # 里程碑日志照常写，日志文件里仍留有 10% 一档的进度记录；
                # 进度条与日志写同一个流，先让出整行再写，写完自动重画。
                # 日志等级高于 INFO 时这条记录会被丢弃，此时不必让行，
                # 否则终端上的进度条会毫无理由地闪一下。
                if milestone and logger.isEnabledFor(logging.INFO):
                    with progress.paused():
                        logger.info(
                            "滚动验证进度 [%d/%d] %.1f%%",
                            position,
                            total_dates,
                            position / total_dates * 100,
                        )
                # A label is available at T only after T closes, so training must end before T.
                train_frame = dataset.loc[dataset["target_end_date"] < target_date]
                predict_frame = dataset.loc[dataset["target_date"] == target_date]
                if train_frame.empty or predict_frame.empty:
                    # 跳过的日期同样占用一个进度步，否则末帧到不了 100%。
                    progress.advance(detail=date_text)
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
                daily["training_end_date"] = train_frame["target_end_date"].max()
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
                progress.advance(detail=date_text)

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
