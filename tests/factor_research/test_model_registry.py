from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import quant.factor_research.models as models_package
from quant.factor_research.models.base import DirectionModelFactory
from quant.factor_research.models.registry import (
    MODEL_FACTORY_TYPES,
    available_models,
    model_factory_from_args,
    register_model_factory,
)
from quant.factor_research.models.simple_decision_tree import (
    SimpleDecisionTreeClassifier,
    SimpleDecisionTreeModelFactory,
)
from quant.cli.factor_demo import _config_defaults, parse_args


class ModelRegistryTest(unittest.TestCase):
    def setUp(self):
        self.original_factories = MODEL_FACTORY_TYPES.copy()

    def tearDown(self):
        MODEL_FACTORY_TYPES.clear()
        MODEL_FACTORY_TYPES.update(self.original_factories)

    def test_model_is_selected_and_configured_from_args(self):
        argv = [
            "quant-factor-demo",
            "--model",
            "simple_decision_tree",
            "--max-depth",
            "7",
            "--min-samples-leaf",
            "11",
            "--max-thresholds",
            "19",
        ]
        args = parse_args(argv[1:])

        factory = model_factory_from_args(args)
        self.assertIsInstance(factory, SimpleDecisionTreeModelFactory)
        self.assertEqual(args.model, "simple_decision_tree")
        self.assertEqual(factory.max_depth, 7)
        self.assertEqual(factory.min_samples_leaf, 11)
        self.assertEqual(factory.max_thresholds, 19)

    def test_default_model_is_registered(self):
        self.assertIn("simple_decision_tree", available_models())
        args = parse_args([])
        self.assertEqual(args.model, "simple_decision_tree")
        self.assertEqual(args.training_mode, "rolling")
        self.assertEqual(args.task, "classification")

    def test_task_supports_cli_and_yaml(self):
        args = parse_args(["--model", "lightgbm", "--task", "regression"])
        self.assertEqual(args.task, "regression")
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                "model: lightgbm\n"
                "task: regression\n"
                "objective: huber\n"
                "objective_alpha: 0.02\n",
                encoding="utf-8",
            )
            args = parse_args(["--config", str(config_path)])
        factory = model_factory_from_args(args)
        self.assertEqual(factory.task, "regression")
        self.assertEqual(factory.objective, "huber")
        self.assertEqual(factory.objective_alpha, 0.02)

        unsupported = parse_args(
            ["--model", "simple_decision_tree", "--task", "regression"]
        )
        with self.assertRaisesRegex(ValueError, "不支持任务"):
            model_factory_from_args(unsupported)

    def test_training_mode_supports_cli_and_yaml(self):
        self.assertEqual(
            parse_args(["--training-mode", "single"]).training_mode,
            "single",
        )
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text("training_mode: single\n", encoding="utf-8")
            args = parse_args(["--config", str(config_path)])
        self.assertEqual(args.training_mode, "single")

        with self.assertRaises(SystemExit):
            parse_args(["--training-mode", "unknown"])

    def test_factor_expressions_support_cli_and_yaml(self):
        """搜索表达式应能从命令行或 YAML 字符串列表原样传入。"""

        expression = "cs_rank(delta(column(close),periods=5))"
        args = parse_args(["--factor-expressions", expression])
        self.assertEqual(args.factor_expressions, [expression])
        only_expression = parse_args(
            ["--factors", "--factor-expressions", expression]
        )
        self.assertEqual(only_expression.factors, [])

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                f'factors: []\nfactor_expressions:\n  - "{expression}"\n',
                encoding="utf-8",
            )
            args = parse_args(["--config", str(config_path)])
        self.assertEqual(args.factor_expressions, [expression])
        self.assertEqual(args.factors, [])

        with self.assertRaises(SystemExit):
            parse_args(["--factor-expressions", "open('secret.txt').read()"])

    def test_yaml_config_selects_model_and_converts_values(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                """
model: gradient_boosting_tree
database: data/market.duckdb
codes:
  - 000001.SZ
  - 600000.SH
factors:
  - return_1d
  - return_5d
no_factor_cache: true
n_estimators: 25
learning_rate: 0.2
""".strip(),
                encoding="utf-8",
            )

            args = parse_args(["--config", str(config_path)])

        self.assertEqual(args.model, "gradient_boosting_tree")
        self.assertEqual(args.database, Path("data/market.duckdb"))
        self.assertEqual(args.codes, ["000001.SZ", "600000.SH"])
        self.assertEqual(args.factors, ["return_1d", "return_5d"])
        self.assertTrue(args.no_factor_cache)
        self.assertEqual(args.n_estimators, 25)
        self.assertEqual(args.learning_rate, 0.2)

    def test_command_line_overrides_yaml_config(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                "model: simple_decision_tree\nmax_depth: 4\nlog_level: WARNING\n",
                encoding="utf-8",
            )

            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--max-depth",
                    "9",
                    "--log-level",
                    "DEBUG",
                ]
            )

        self.assertEqual(args.max_depth, 9)
        self.assertEqual(args.log_level, "DEBUG")

    def test_command_line_model_ignores_previous_model_only_options(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                """
model: gradient_boosting_tree
n_estimators: 25
learning_rate: 0.2
max_depth: 4
""".strip(),
                encoding="utf-8",
            )

            args = parse_args(
                ["--config", str(config_path), "--model", "simple_decision_tree"]
            )

        self.assertEqual(args.model, "simple_decision_tree")
        self.assertEqual(args.max_depth, 4)
        self.assertFalse(hasattr(args, "n_estimators"))
        self.assertFalse(hasattr(args, "learning_rate"))

    def test_yaml_config_rejects_unknown_and_invalid_values(self):
        invalid_configs = [
            "unknown_option: 1\n",
            "log-level: DEBUG\n",
            "model: unknown_model\n",
            "model: simple_decision_tree\nfactors: not-a-list\n",
            "model: simple_decision_tree\nfactors: []\n",
            "model: simple_decision_tree\nfactors: [unknown_factor]\n",
            "model: simple_decision_tree\nlog_level: VERBOSE\n",
            "model: simple_decision_tree\ntraining_mode: unknown\n",
            'model: simple_decision_tree\ndebug: "true"\n',
            "model: simple_decision_tree\nsymbol_limit: true\n",
            "- not-a-mapping\n",
            "[invalid yaml\n",
        ]
        for content in invalid_configs:
            with self.subTest(content=content), TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "invalid.yaml"
                config_path.write_text(content, encoding="utf-8")
                with self.assertRaises(SystemExit):
                    parse_args(["--config", str(config_path)])

    def test_yaml_config_validates_fixed_length_nargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pair", nargs=2, type=int)

        with self.assertRaises(SystemExit):
            _config_defaults(parser, {"pair": [1]})
        self.assertEqual(_config_defaults(parser, {"pair": [1, 2]}), {"pair": [1, 2]})

    def test_yaml_config_converts_argument_type_error_to_parser_error(self):
        def positive_integer(value: str) -> int:
            converted = int(value)
            if converted <= 0:
                raise argparse.ArgumentTypeError("必须是正整数")
            return converted

        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=positive_integer)

        with self.assertRaises(SystemExit):
            _config_defaults(parser, {"count": -1})
        self.assertEqual(_config_defaults(parser, {"count": 2}), {"count": 2})

    def test_selected_model_arguments_are_isolated(self):
        @register_model_factory
        class AlternativeTreeFactory(DirectionModelFactory):
            name = "test_alternative_tree"

            def __init__(self, depth: int = 9):
                self.depth = depth

            @classmethod
            def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
                # 与默认模型使用同一个参数名；两阶段解析应避免 argparse 冲突。
                parser.add_argument("--max-depth", type=int, default=9)

            @classmethod
            def from_args(cls, args: argparse.Namespace):
                return cls(depth=args.max_depth)

            def create(self):
                return SimpleDecisionTreeClassifier(max_depth=self.depth)

        argv = [
            "quant-factor-demo",
            "--model",
            "test_alternative_tree",
            "--max-depth",
            "13",
        ]
        args = parse_args(argv[1:])

        factory = model_factory_from_args(args)
        self.assertIsInstance(factory, AlternativeTreeFactory)
        self.assertEqual(factory.depth, 13)

    def test_factory_without_cli_configuration_keeps_injection_compatibility(self):
        class InjectedFactory(DirectionModelFactory):
            name = "injected"

            def create(self):
                return SimpleDecisionTreeClassifier()

        factory = InjectedFactory()
        self.assertIsInstance(factory.create(), SimpleDecisionTreeClassifier)
        self.assertIsInstance(
            InjectedFactory.from_args(argparse.Namespace()), InjectedFactory
        )

    def test_duplicate_and_unknown_model_names_are_rejected(self):
        with self.assertRaises(ValueError):
            register_model_factory(SimpleDecisionTreeModelFactory)
        with self.assertRaises(ValueError):
            model_factory_from_args(argparse.Namespace(model="unknown"))

    def test_new_model_module_is_discovered_without_main_program_changes(self):
        module_name = "auto_discovery_test_model"
        qualified_name = f"quant.factor_research.models.{module_name}"
        model_name = "auto_discovered_test_model"
        source = f'''\
from quant.factor_research.models import DirectionModelFactory, register_model_factory


@register_model_factory
class AutoDiscoveredTestModelFactory(DirectionModelFactory):
    name = "{model_name}"

    def create(self):
        raise NotImplementedError("The discovery test does not train this model")
'''

        with TemporaryDirectory() as module_dir:
            Path(module_dir, f"{module_name}.py").write_text(source, encoding="utf-8")
            models_package.__path__.append(module_dir)
            try:
                self.assertNotIn(model_name, MODEL_FACTORY_TYPES)
                self.assertNotIn(qualified_name, sys.modules)

                # 首次查询时惰性扫描包路径，新模块无需加入 models/__init__.py。
                self.assertIn(model_name, available_models())
                self.assertIn(qualified_name, sys.modules)
                factory = model_factory_from_args(argparse.Namespace(model=model_name))
                self.assertEqual(factory.name, model_name)
                self.assertEqual(factory.__class__.__module__, qualified_name)
            finally:
                models_package.__path__.remove(module_dir)
                sys.modules.pop(qualified_name, None)


if __name__ == "__main__":
    unittest.main()
