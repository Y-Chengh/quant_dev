from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    prediction = (probability >= 0.5).astype(int)
    accuracy = float(np.mean(prediction == y))
    recalls = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    # 对相同概率使用平均秩，避免树模型大量并列预测时AUC产生偏差。
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2
        start = end
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = sorted_ranks
    positive, negative = y == 1, y == 0
    auc = (
        float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))
        if positive.any() and negative.any()
        else float("nan")
    )
    return {
        "samples": float(len(y)),
        "positive_rate": float(y.mean()),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "auc": auc,
        "brier_score": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
    }
