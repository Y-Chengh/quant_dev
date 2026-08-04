from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import factor_research.models as models_package
from factor_research.models.base import DirectionModelFactory
from factor_research.models.registry import (
    MODEL_FACTORY_TYPES,
    available_models,
    model_factory_from_args,
    register_model_factory,
)
from factor_research.models.simple_decision_tree import (
    SimpleDecisionTreeClassifier,
    SimpleDecisionTreeModelFactory,
)
from run_factor_demo import parse_args


class ModelRegistryTest(unittest.TestCase):
    def setUp(self):
        self.original_factories = MODEL_FACTORY_TYPES.copy()

    def tearDown(self):
        MODEL_FACTORY_TYPES.clear()
        MODEL_FACTORY_TYPES.update(self.original_factories)

    def test_model_is_selected_and_configured_from_args(self):
        argv = [
            "run_factor_demo.py",
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
            "run_factor_demo.py",
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
        qualified_name = f"factor_research.models.{module_name}"
        model_name = "auto_discovered_test_model"
        source = f'''\
from factor_research.models import DirectionModelFactory, register_model_factory


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
