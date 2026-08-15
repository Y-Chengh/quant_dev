"""评估报告使用的指标中文名与 IC 趋势窗口常量。"""

from __future__ import annotations

METRIC_LABELS = {
    "samples": "样本数",
    "positive_rate": "实际上涨比例",
    "accuracy": "准确率",
    "balanced_accuracy": "平衡准确率",
    "auc": "ROC AUC",
    "ic": "IC",
    "rank_ic": "Rank IC",
    "icir": "ICIR",
    "ic_win_rate": "IC 胜率",
    "pooled_ic": "全样本 Pearson IC",
    "ic_dates": "有效 IC 交易日数",
    "rank_ic_dates": "有效 Rank IC 交易日数",
    "brier_score": "Brier 分数",
    "log_loss": "Log Loss",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "direction_accuracy": "方向准确率",
}

IC_TREND_SHORT_WINDOW = 20
IC_TREND_SHORT_MIN_PERIODS = 5
IC_TREND_LONG_WINDOW = 60
IC_TREND_LONG_MIN_PERIODS = 20
