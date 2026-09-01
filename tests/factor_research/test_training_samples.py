from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quant.cli.factor_demo import parse_args
from quant.factor_research.dataset import (
    TRAINING_SAMPLE_ELIGIBLE_COLUMN,
    TRAINING_SAMPLE_FILTER_REASON_COLUMN,
    build_direction_dataset,
)
from quant.factor_research.experiment import DirectionExperiment
from quant.factor_research.training_samples import mark_training_sample_eligibility


class TrainingSampleExecutionFilterTest(unittest.TestCase):
    """验证训练样本按目标买卖时点排除不可成交记录。"""

    @staticmethod
    def _dataset() -> pd.DataFrame:
        """构造同一持有期内分别涨停、跌停和正常的三个样本。"""

        return pd.DataFrame(
            {
                "feature_date": pd.to_datetime(["2025-01-01"] * 3),
                "target_date": pd.to_datetime(["2025-01-02"] * 3),
                "target_end_date": pd.to_datetime(["2025-01-03"] * 3),
                "code": ["UP", "DOWN", "NORMAL"],
                "target_return": [0.01, -0.01, 0.02],
                "label": [1, 0, 1],
            }
        )

    @staticmethod
    def _context() -> pd.DataFrame:
        """构造买入日涨停、卖出日跌停及正常价格上下文。"""

        return pd.DataFrame(
            {
                "execution_date": pd.to_datetime(
                    [
                        "2025-01-02",
                        "2025-01-03",
                        "2025-01-02",
                        "2025-01-03",
                        "2025-01-02",
                        "2025-01-03",
                    ]
                ),
                "code": ["UP", "UP", "DOWN", "DOWN", "NORMAL", "NORMAL"],
                "execution_open": [11.0, 11.2, 10.5, 10.6, 10.2, 10.3],
                "execution_close": [11.0, 11.2, 10.6, 9.54, 10.2, 10.3],
                "execution_pre_close": [10.0, 11.0, 10.0, 10.6, 10.0, 10.2],
                "execution_board": ["main"] * 6,
                "execution_is_st": [False] * 6,
                "execution_sessions_since_listing": [100] * 6,
                "execution_has_corporate_action": [False] * 6,
                "execution_suspended": [False] * 6,
            }
        )

    def test_marks_limit_up_entry_and_limit_down_exit_ineligible(self) -> None:
        """目标日涨停买入和结束日跌停卖出样本都不得进入训练。"""

        marked, stats = mark_training_sample_eligibility(
            self._dataset(),
            self._context(),
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
        )

        self.assertEqual(
            marked[TRAINING_SAMPLE_ELIGIBLE_COLUMN].tolist(),
            [False, False, True],
        )
        self.assertEqual(
            marked[TRAINING_SAMPLE_FILTER_REASON_COLUMN].tolist(),
            ["buy:limit_up_buy", "sell:limit_down_sell", ""],
        )
        blocked = stats.set_index("filter")["blocked_count"].to_dict()
        self.assertEqual(blocked, {"limit_up_buy": 1, "limit_down_sell": 1})

    def test_generator_applies_filters_to_both_trade_sides(self) -> None:
        """一次性过滤器迭代器不得在买入侧消费后漏掉卖出侧规则。"""

        marked, _ = mark_training_sample_eligibility(
            self._dataset(),
            self._context(),
            (name for name in ("limit_up_buy", "limit_down_sell")),
            entry_timing="open",
            exit_timing="close",
        )

        self.assertEqual(
            marked[TRAINING_SAMPLE_ELIGIBLE_COLUMN].tolist(),
            [False, False, True],
        )

    def test_duplicate_dataset_index_keeps_positional_alignment(self) -> None:
        """重复外部索引不得让买卖判定串到其它训练样本。"""

        dataset = self._dataset()
        dataset.index = [7, 7, 7]
        marked, _ = mark_training_sample_eligibility(
            dataset,
            self._context(),
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
        )

        self.assertEqual(
            marked[TRAINING_SAMPLE_FILTER_REASON_COLUMN].tolist(),
            ["buy:limit_up_buy", "sell:limit_down_sell", ""],
        )

    def test_cutoff_is_exclusive_and_can_leave_no_candidates(self) -> None:
        """结束日等于边界的样本不参与本次检查，空候选应稳定返回。"""

        dataset = self._dataset().iloc[[2]].copy()
        marked, stats = mark_training_sample_eligibility(
            dataset,
            None,
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
            training_cutoff=dataset.iloc[0]["target_end_date"],
            missing_policy="error",
        )

        self.assertTrue(marked.iloc[0][TRAINING_SAMPLE_ELIGIBLE_COLUMN])
        self.assertTrue(stats.empty)

    def test_inday_uses_open_for_entry_and_close_for_exit(self) -> None:
        """日内目标同一天应分别检查开盘买入与收盘卖出。"""

        dataset = self._dataset().iloc[[0]].copy()
        dataset["target_end_date"] = dataset["target_date"]
        context = self._context().iloc[[0]].copy()
        context["execution_open"] = 10.5
        context["execution_close"] = 9.0

        marked, _ = mark_training_sample_eligibility(
            dataset,
            context,
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
        )

        self.assertFalse(marked.loc[marked.index[0], TRAINING_SAMPLE_ELIGIBLE_COLUMN])
        self.assertEqual(
            marked.loc[marked.index[0], TRAINING_SAMPLE_FILTER_REASON_COLUMN],
            "sell:limit_down_sell",
        )

    def test_unknown_context_is_allowed_by_default(self) -> None:
        """缺省策略只排除已确认不可成交样本，不误删无法判定记录。"""

        marked, stats = mark_training_sample_eligibility(
            self._dataset().iloc[[2]],
            self._context().iloc[0:0],
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
        )

        self.assertTrue(marked.iloc[0][TRAINING_SAMPLE_ELIGIBLE_COLUMN])
        self.assertEqual(stats["unknown_count"].tolist(), [1, 1])

    def test_cutoff_does_not_evaluate_terminal_validation_samples(self) -> None:
        """从不进入拟合的末端验证样本不应触发严格缺失上下文策略。"""

        dataset = pd.DataFrame(
            {
                "target_date": pd.to_datetime(["2025-01-02", "2025-01-04"]),
                "target_end_date": pd.to_datetime(["2025-01-02", "2025-01-04"]),
                "code": ["TRAIN", "VALIDATION"],
            }
        )
        context = self._context().iloc[[4]].copy()
        context["execution_date"] = pd.Timestamp("2025-01-02")
        context["code"] = "TRAIN"

        marked, stats = mark_training_sample_eligibility(
            dataset,
            context,
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="open",
            exit_timing="close",
            training_cutoff="2025-01-03",
            missing_policy="error",
        )

        self.assertEqual(marked[TRAINING_SAMPLE_ELIGIBLE_COLUMN].tolist(), [True, True])
        self.assertEqual(stats["candidate_count"].tolist(), [1, 1])

    def test_suspended_daily_row_is_filtered_at_planned_trade_date(self) -> None:
        """日线保留停牌日时，目标构建不得跳过计划买卖日的停牌状态。"""

        daily = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=4, freq="D"),
                "code": ["A"] * 4,
                "open": [10.0, 10.1, 10.1, 10.2],
                "close": [10.0, 10.1, 10.1, 10.2],
                "test_feature": [0.0, 1.0, 2.0, 3.0],
            }
        )
        dataset = build_direction_dataset(
            daily,
            ["test_feature"],
            target="close",
        )
        context = pd.DataFrame(
            {
                "execution_date": pd.date_range("2025-01-02", periods=3, freq="D"),
                "code": ["A"] * 3,
                "execution_open": [10.1, 10.1, 10.2],
                "execution_close": [10.1, 10.1, 10.2],
                "execution_pre_close": [10.0, 10.1, 10.1],
                "execution_board": ["main"] * 3,
                "execution_is_st": [False] * 3,
                "execution_sessions_since_listing": [100] * 3,
                "execution_has_corporate_action": [False] * 3,
                "execution_suspended": [False, True, False],
            }
        )

        marked, _ = mark_training_sample_eligibility(
            dataset,
            context,
            ["limit_up_buy", "limit_down_sell"],
            entry_timing="close",
            exit_timing="close",
        )

        self.assertEqual(
            marked[TRAINING_SAMPLE_FILTER_REASON_COLUMN].tolist(),
            ["sell:suspended_sell", "buy:suspended_buy"],
        )

    def test_yaml_config_and_cli_parse_training_filter_options(self) -> None:
        """训练过滤器名称和缺失策略应支持 YAML 及命令行覆盖。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "training_sample_filters: [limit_down_sell]\n"
                "training_sample_filter_missing_policy: exclude\n",
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(path),
                    "--training-sample-filter-missing-policy",
                    "error",
                ]
            )

        self.assertEqual(args.training_sample_filters, ["limit_down_sell"])
        self.assertEqual(args.training_sample_filter_missing_policy, "error")


class TrainingSampleExperimentTest(unittest.TestCase):
    """验证训练资格只作用于拟合历史而不缩小验证预测截面。"""

    @staticmethod
    def _dataset() -> pd.DataFrame:
        """构造五个目标日、每日两个证券的可滚动训练数据集。"""

        rows: list[dict[str, object]] = []
        dates = pd.date_range("2025-01-02", periods=5, freq="D")
        for date_position, target_date in enumerate(dates):
            for code_position, code in enumerate(("A", "B")):
                positive = (date_position + code_position) % 2 == 0
                rows.append(
                    {
                        "feature_date": target_date - pd.Timedelta(days=1),
                        "target_date": target_date,
                        "target_end_date": target_date,
                        "code": code,
                        "test_feature": float(date_position + code_position),
                        "target_return": 0.01 if positive else -0.01,
                        "label": 1 if positive else 0,
                        TRAINING_SAMPLE_ELIGIBLE_COLUMN: True,
                    }
                )
        dataset = pd.DataFrame(rows)
        dataset.loc[
            (dataset["target_date"] == pd.Timestamp("2025-01-02"))
            & dataset["code"].eq("A"),
            TRAINING_SAMPLE_ELIGIBLE_COLUMN,
        ] = False
        dataset.loc[
            (dataset["target_date"] == pd.Timestamp("2025-01-04"))
            & dataset["code"].eq("A"),
            TRAINING_SAMPLE_ELIGIBLE_COLUMN,
        ] = False
        return dataset

    def test_single_fit_excludes_ineligible_history_but_predicts_all_validation(self) -> None:
        """单次训练应过滤历史样本并完整预测验证截面。"""

        result = DirectionExperiment(
            "2025-01-04",
            feature_columns=["test_feature"],
            max_depth=2,
            min_samples_leaf=1,
            training_mode="single",
        ).run(self._dataset())

        self.assertEqual(len(result.predictions), 6)
        self.assertTrue((result.predictions["training_samples"] == 3).all())

    def test_rolling_fit_keeps_filtered_validation_row_out_of_later_history(self) -> None:
        """验证期被标记样本在后续日期进入历史窗口时仍应被排除。"""

        result = DirectionExperiment(
            "2025-01-04",
            feature_columns=["test_feature"],
            max_depth=2,
            min_samples_leaf=1,
            training_mode="rolling",
        ).run(self._dataset())

        samples = result.predictions.groupby("target_date")["training_samples"].first()
        self.assertEqual(samples.loc[pd.Timestamp("2025-01-04")], 3)
        self.assertEqual(samples.loc[pd.Timestamp("2025-01-05")], 4)
        first_validation = result.predictions.loc[
            result.predictions["target_date"] == pd.Timestamp("2025-01-04")
        ]
        self.assertEqual(set(first_validation["code"]), {"A", "B"})

    def test_rolling_fit_fails_instead_of_skipping_date_with_no_history(self) -> None:
        """首个验证日没有合格历史时必须明确失败，不能静默缩短验证区间。"""

        dataset = self._dataset()
        dataset.loc[
            dataset["target_date"] < pd.Timestamp("2025-01-04"),
            TRAINING_SAMPLE_ELIGIBLE_COLUMN,
        ] = False

        with self.assertRaisesRegex(ValueError, "target_date=2025-01-04"):
            DirectionExperiment(
                "2025-01-04",
                feature_columns=["test_feature"],
                max_depth=2,
                min_samples_leaf=1,
                training_mode="rolling",
            ).run(dataset)


if __name__ == "__main__":
    unittest.main()
