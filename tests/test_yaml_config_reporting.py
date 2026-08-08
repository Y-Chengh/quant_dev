from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from factor_research.experiment import ExperimentResult
from factor_research.reporting import write_evaluation_report
import run_factor_demo


def _result() -> ExperimentResult:
    target_date = pd.Timestamp("2025-06-02")
    return ExperimentResult(
        model=None,  # type: ignore[arg-type]
        model_name="test_model",
        feature_columns=["return_1d"],
        metrics={"samples": 1.0, "accuracy": 1.0},
        predictions=pd.DataFrame(
            {
                "target_date": [target_date],
                "label": [1],
                "up_probability": [0.75],
            }
        ),
        feature_importance=pd.Series({"return_1d": 1.0}),
        daily_accuracy_trend=pd.DataFrame(
            {
                "target_date": [target_date],
                "samples": [1],
                "accuracy": [1.0],
                "accuracy_change": [np.nan],
            }
        ),
    )


class YamlConfigReportingTest(unittest.TestCase):
    def _render(self, yaml_config: str | None) -> str:
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "evaluation.md"
            chart_path = Path(temp_dir) / "accuracy.svg"
            write_evaluation_report(
                _result(),
                report_path,
                chart_path,
                "test-run",
                {"config": "experiment.yaml"},
                yaml_config=yaml_config,
            )
            return report_path.read_text(encoding="utf-8")

    def test_yaml_snapshot_is_between_arguments_and_daily_summary(self):
        yaml_config = "model: simple_decision_tree\nmax_depth: 2\n"

        report = self._render(yaml_config)

        self.assertLess(report.index("## 运行参数"), report.index("## YAML 配置"))
        self.assertLess(report.index("## YAML 配置"), report.index("## 每日预估汇总"))
        self.assertIn(yaml_config.strip(), report)

    def test_yaml_snapshot_uses_a_longer_fence_when_content_has_backticks(self):
        yaml_config = 'description: "contains ``` markdown fence"\n'

        report = self._render(yaml_config)

        self.assertIn(f"````yaml\n{yaml_config.strip()}\n````", report)

    def test_empty_yaml_snapshot_is_preserved_as_an_empty_code_block(self):
        report = self._render("")

        self.assertIn("## YAML 配置\n\n```yaml\n\n```", report)

    def test_missing_yaml_config_is_recorded(self):
        report = self._render(None)

        self.assertIn("## YAML 配置\n\n未使用 YAML 配置文件。", report)

    def test_regression_result_is_reported_without_probabilities(self):
        target_date = pd.Timestamp("2025-06-02")
        result = ExperimentResult(
            model=None,  # type: ignore[arg-type]
            model_name="test_regressor",
            feature_columns=["return_1d"],
            metrics={"samples": 1.0, "rmse": 0.001},
            predictions=pd.DataFrame(
                {
                    "target_date": [target_date],
                    "label": [1],
                    "target_return": [0.01],
                    "predicted_return": [0.009],
                    "prediction": [1],
                }
            ),
            feature_importance=None,
            daily_accuracy_trend=pd.DataFrame(
                {
                    "target_date": [target_date],
                    "samples": [1],
                    "accuracy": [1.0],
                    "accuracy_change": [np.nan],
                }
            ),
            task="regression",
        )
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "evaluation.md"
            chart_path = Path(temp_dir) / "accuracy.svg"
            write_evaluation_report(
                result, report_path, chart_path, "regression-run", {}
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("# 涨跌幅预测评估报告", report)
        self.assertIn("| RMSE | 0.001000 |", report)

    def test_main_reads_and_passes_the_original_yaml_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "experiment.yaml"
            yaml_config = "model: simple_decision_tree\n# preserve this comment\n"
            config_path.write_text(yaml_config, encoding="utf-8")
            args = argparse.Namespace(
                config=config_path,
                debug=False,
                log_level="ERROR",
                log_dir=root / "logs",
                database=root / "market.duckdb",
                start="2024-01-01",
                end="2026-01-01",
                validation_start="2025-06-01",
                codes=["000001.SZ"],
                symbol_limit=1,
                training_mode="rolling",
                task="classification",
                factors=["return_1d"],
                factor_cache_dir=root / "cache",
                no_factor_cache=True,
                model="simple_decision_tree",
            )
            experiment = SimpleNamespace(run=lambda dataset: _result())
            client = SimpleNamespace(get_metadata=lambda: {})

            with ExitStack() as stack:
                stack.enter_context(patch.object(run_factor_demo, "parse_args", return_value=args))
                stack.enter_context(
                    patch.object(
                        run_factor_demo,
                        "resolve_window",
                        return_value=(datetime(2024, 1, 1), datetime(2026, 1, 1)),
                    )
                )
                stack.enter_context(
                    patch("market_service.client.MarketDataClient", return_value=client)
                )
                stack.enter_context(
                    patch.object(
                        run_factor_demo,
                        "load_market_service",
                        return_value=pd.DataFrame({"close": [1.0]}),
                    )
                )
                stack.enter_context(
                    patch.object(
                        run_factor_demo,
                        "build_daily_features",
                        return_value=pd.DataFrame({"return_1d": [0.1]}),
                    )
                )
                stack.enter_context(
                    patch.object(
                        run_factor_demo,
                        "build_direction_dataset",
                        return_value=pd.DataFrame({"return_1d": [0.1]}),
                    )
                )
                stack.enter_context(
                    patch.object(run_factor_demo, "DirectionExperiment", return_value=experiment)
                )
                stack.enter_context(patch.object(run_factor_demo, "model_factory_from_args"))
                report_writer = stack.enter_context(
                    patch.object(run_factor_demo, "write_evaluation_report")
                )
                stack.enter_context(
                    patch.object(
                        run_factor_demo,
                        "RotatingFileHandler",
                        return_value=logging.NullHandler(),
                    )
                )
                stack.enter_context(patch.object(run_factor_demo.logging, "basicConfig"))

                run_factor_demo.main()

        self.assertEqual(report_writer.call_args.kwargs["yaml_config"], yaml_config)


if __name__ == "__main__":
    unittest.main()
