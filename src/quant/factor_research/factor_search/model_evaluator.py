"""因子搜索与现有 DirectionExperiment 之间的可选模型评价适配器。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant.factor_research.dataset import build_direction_dataset
from quant.factor_research.experiment import DirectionExperiment
from quant.factor_research.models.base import DirectionModelFactory

from .context import SearchContext
from .space import FactorCandidate


@dataclass(frozen=True)
class ModelCandidateEvaluator:
    """把候选追加到固定因子后，复用现有 DirectionExperiment 做 Top K 验证。"""

    model_factory: DirectionModelFactory
    validation_start: str | pd.Timestamp | None = None
    training_mode: str = "rolling"
    task: str = "classification"

    def evaluate(
        self,
        candidate: FactorCandidate,
        values: pd.Series,
        context: SearchContext,
    ) -> dict[str, float]:
        """把当前候选与固定因子合并，运行既有实验并返回带前缀的模型指标。

        参数：
            candidate: 要追加到固定因子集的 Top K 候选。
            values: 与 ``context.daily`` 逐行对齐的候选因子值。
            context: 提供日频数据、固定因子和验证日期边界的上下文。
        """

        validation_start = (
            pd.Timestamp(self.validation_start)
            if self.validation_start is not None
            else context.holdout_start
        )
        if validation_start is None:
            raise ValueError(
                "ModelCandidateEvaluator 需要 validation_start 或 SearchContext.holdout_start"
            )
        if candidate.factor_id in context.daily.columns:
            raise ValueError(f"候选列 {candidate.factor_id!r} 已存在，拒绝静默覆盖")
        daily = context.daily.copy()
        daily[candidate.factor_id] = values.to_numpy(dtype=float)
        features = [*context.fixed_features, candidate.factor_id]
        dataset = build_direction_dataset(
            daily,
            feature_columns=features,
            label_return_threshold=context.label_return_threshold,
        )
        if context.holdout_end is not None:
            # 保留全部训练历史，但禁止验证结果越过 SearchContext 声明的最终日期。
            dataset = dataset.loc[
                dataset["target_date"] <= context.holdout_end
            ].copy()
        experiment = DirectionExperiment(
            validation_start,
            feature_columns=features,
            model_factory=self.model_factory,
            training_mode=self.training_mode,
            task=self.task,
            label_return_threshold=context.label_return_threshold,
        )
        result = experiment.run(dataset)
        return {f"model_{name}": float(value) for name, value in result.metrics.items()}
