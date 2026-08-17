from __future__ import annotations

import io
import logging
import os
import unittest
from contextlib import redirect_stderr
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant.factor_research import experiment as experiment_module
from quant.factor_research.experiment import DirectionExperiment, ExperimentResult
from quant.factor_research.models.base import (
    DirectionModel,
    DirectionModelFactory,
    FitProgressCallback,
)
from quant.factor_research.progress import (
    PROGRESS_MODES,
    ProgressBar,
    display_width,
    format_duration,
    truncate_to_width,
)


class _TerminalStream(io.StringIO):
    """伪装成交互式终端的内存流，用于验证 ``auto`` 模式的分支。"""

    def isatty(self) -> bool:
        """始终声明自己是终端。"""

        return True


def _build_dataset(periods: int = 6) -> pd.DataFrame:
    """构造两只证券、每日一条样本的最小训练验证数据集。

    参数：
        periods: 目标交易日数量；验证起点固定取第 4 个日期。

    返回：
        含特征、标签、目标收益及日期列的样本表，可直接交给实验运行。
    """

    target_dates = pd.date_range("2024-03-01", periods=periods, freq="D")
    rows = []
    for date_position, target_date in enumerate(target_dates):
        for code_position, code in enumerate(["000001.SZ", "600000.SH"]):
            label = (date_position + code_position) % 2
            rows.append(
                {
                    "feature_date": target_date - pd.Timedelta(days=1),
                    "target_date": target_date,
                    "code": code,
                    "factor_a": float(date_position),
                    "factor_b": float(code_position),
                    "label": label,
                    "target_return": 0.01 if label else -0.01,
                }
            )
    return pd.DataFrame(rows)


_USE_ROUNDS = object()
"""桩模型的哨兵：表示 ``set_fit_progress`` 按实际轮数声明。

不能用 ``None`` 当哨兵：``None`` 本身是「无法预知轮数」这一合法声明值，
需要作为独立用例传入。
"""


class _RoundReportingModel(DirectionModel):
    """按固定轮数上报训练进度的最小分类模型，用于验证进度刻度。"""

    def __init__(
        self,
        rounds: int,
        reported_rounds: int,
        declared_rounds: object = _USE_ROUNDS,
    ) -> None:
        """记录声明轮数与实际上报轮数。

        参数：
            rounds: 训练时传给回调的总轮数。
            reported_rounds: 训练时实际上报的轮数；小于 ``rounds`` 用于模拟提前收敛。
            declared_rounds: ``set_fit_progress`` 的返回值；缺省按 ``rounds`` 声明，
                传入 ``None`` 或非法值用于验证调用方对异常声明的降级处理。
        """

        self.rounds = rounds
        self.reported_rounds = reported_rounds
        self.declared_rounds = (
            rounds if declared_rounds is _USE_ROUNDS else declared_rounds
        )
        self.installed_callback: FitProgressCallback | None = None
        # 只看最终值无法区分「从未安装」与「装了又卸」，因此记录每一次装卸。
        self.progress_calls: list[FitProgressCallback | None] = []
        self._positive_rate = 0.0

    def set_fit_progress(self, callback: FitProgressCallback | None) -> int | None:
        """注册或卸载训练进度回调，并声明总轮数。

        参数：
            callback: 训练中按轮调用的回调；``None`` 表示卸载。

        返回：
            注册时返回声明的总轮数，卸载时返回 ``None``。
        """

        self.installed_callback = callback
        self.progress_calls.append(callback)
        return None if callback is None else self.declared_rounds

    def fit(self, X: np.ndarray, y: np.ndarray) -> _RoundReportingModel:
        """逐轮上报进度，并记住训练集里的多数类。

        参数：
            X: 二维特征矩阵，本模型不使用其取值。
            y: 0/1 标签数组，用于决定常数预测概率。

        返回：
            模型自身。
        """

        for completed in range(1, self.reported_rounds + 1):
            if self.installed_callback is not None:
                self.installed_callback(completed, self.rounds)
        self._positive_rate = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回与训练集正类比例相同的常数概率。

        参数：
            X: 待预测的二维特征矩阵，只用于确定输出行数。

        返回：
            形状为（样本数, 2）的下跌、上涨概率。
        """

        positive = np.full(len(np.asarray(X)), self._positive_rate, dtype=float)
        return np.column_stack([1.0 - positive, positive])


class _RoundReportingModelFactory(DirectionModelFactory):
    """创建 :class:`_RoundReportingModel` 的测试工厂。"""

    name = "round_reporting_stub"

    def __init__(
        self,
        rounds: int,
        reported_rounds: int | None = None,
        declared_rounds: object = _USE_ROUNDS,
    ) -> None:
        """保存声明轮数、实际上报轮数与对外声明值。

        参数：
            rounds: 传给进度回调的训练总轮数。
            reported_rounds: 实际上报的轮数；缺省与 ``rounds`` 相同。
            declared_rounds: ``set_fit_progress`` 的返回值；缺省按 ``rounds`` 声明。
        """

        self.rounds = rounds
        self.reported_rounds = rounds if reported_rounds is None else reported_rounds
        self.declared_rounds = declared_rounds

    def create(self) -> _RoundReportingModel:
        """创建无历史状态的新模型实例。"""

        return _RoundReportingModel(
            self.rounds, self.reported_rounds, self.declared_rounds
        )


def _pin_terminal_columns(test: unittest.TestCase, columns: int = 200) -> None:
    """把终端列数固定为给定值，直到用例结束。

    参数：
        test: 需要固定列数的用例实例；清理动作注册在它身上。
        columns: 期望的终端列数；缺省取足够宽的值，使进度帧不触发截断。

    返回：
        无返回值。进度帧会按终端列数截断，不固定列数时断言会随运行窗口宽度而变。
    """

    patcher = patch.dict(os.environ, {"COLUMNS": str(columns)})
    patcher.start()
    test.addCleanup(patcher.stop)


class ProgressBarTest(unittest.TestCase):
    """验证进度条的渲染内容、节流、覆盖清屏和启停判定。"""

    def setUp(self) -> None:
        """固定终端列数，使渲染断言与实际运行窗口无关。"""

        _pin_terminal_columns(self)

    def test_auto_mode_stays_silent_on_non_terminal_stream(self) -> None:
        """重定向到文件等非终端目标时不得写入任何进度帧。"""

        stream = io.StringIO()
        bar = ProgressBar(5, "滚动验证", mode="auto", stream=stream)

        self.assertFalse(bar.enabled)
        with bar:
            bar.advance()
            bar.advance(3)
        bar.close()

        self.assertEqual(stream.getvalue(), "")
        # 关闭状态下仍照常累计步数，调用方可以复用该计数。
        self.assertEqual(bar.completed, 4)

    def test_auto_mode_renders_on_terminal_stream(self) -> None:
        """输出目标是终端时 ``auto`` 模式应自动开启。"""

        stream = _TerminalStream()
        with ProgressBar(2, "滚动验证", mode="auto", stream=stream) as bar:
            self.assertTrue(bar.enabled)
            bar.advance()

        self.assertIn("滚动验证", stream.getvalue())

    def test_rendering_reports_ratio_counts_and_final_newline(self) -> None:
        """进度帧应含阶段名、比例、计数与明细，末帧补齐 100% 和换行。"""

        stream = io.StringIO()
        with ProgressBar(
            4, "滚动验证", mode="always", stream=stream, min_interval=0.0, bar_width=4
        ) as bar:
            bar.advance(detail="2024-03-04")
            bar.advance(3, detail="2024-03-07")

        frames = stream.getvalue().split("\r")
        # 首帧由 start() 渲染，其后每次 advance 各一帧。
        self.assertEqual(frames[0], "")
        self.assertEqual(len(frames), 4)
        self.assertIn("滚动验证 [----]   0.0% 0/4", frames[1])
        self.assertIn("[#---]  25.0% 1/4", frames[2])
        self.assertIn("2024-03-04", frames[2])
        self.assertIn("[####] 100.0% 4/4", frames[3])
        self.assertIn("2024-03-07", frames[3])
        self.assertTrue(stream.getvalue().endswith("\n"))
        # 一步都没完成时无法外推剩余时间。
        self.assertIn("预计剩余 --", frames[1])
        self.assertNotIn("预计剩余 --", frames[3])

    def test_min_interval_throttles_middle_frames_only(self) -> None:
        """节流应跳过中间帧，但首帧和末帧必须渲染。"""

        stream = io.StringIO()
        bar = ProgressBar(
            3, "滚动验证", mode="always", stream=stream, min_interval=3600.0
        )
        bar.start()
        bar.advance()
        bar.advance()
        bar.advance()
        bar.close()

        frames = [frame for frame in stream.getvalue().split("\r") if frame]
        self.assertEqual(len(frames), 2)
        self.assertIn("0/3", frames[0])
        self.assertIn("100.0% 3/3", frames[1])

    def test_shorter_frame_clears_previous_tail(self) -> None:
        """新帧更短时必须用空格覆盖上一帧残留，避免终端出现拼接文本。"""

        stream = io.StringIO()
        bar = ProgressBar(
            2, "滚动验证", mode="always", stream=stream, min_interval=0.0, bar_width=4
        )
        bar.advance(detail="非常长的定位信息" * 2)
        bar.advance(detail="短")
        bar.close()

        frames = [frame for frame in stream.getvalue().split("\r") if frame]
        self.assertEqual(len(frames), 2)
        first_width = display_width(frames[0])
        # 末帧含补齐空格，其显示宽度不得小于上一帧，否则残留字符会留在行尾。
        self.assertGreaterEqual(display_width(frames[1].rstrip("\n")), first_width)
        self.assertTrue(frames[1].rstrip("\n").endswith(" "))

    def test_paused_clears_the_line_and_redraws_the_same_frame(self) -> None:
        """暂停时应先用空格擦掉整行，退出时按最近一次内容重画同一帧。"""

        stream = io.StringIO()
        bar = ProgressBar(
            4, "滚动验证", mode="always", stream=stream, min_interval=0.0, bar_width=4
        )
        bar.advance(detail="2024-03-04")
        rendered = stream.getvalue().split("\r")[-1]
        stream.seek(0)
        stream.truncate()

        with bar.paused():
            paused_output = stream.getvalue()
            stream.write("[日志占位]\n")

        cleared = "\r" + " " * display_width(rendered) + "\r"
        self.assertEqual(paused_output, cleared)
        # 擦除后光标回到行首，日志整行输出，随后重画同一进度状态。
        # 两帧的耗时文本由实时时钟生成，因此只断言结构与进度内容，不做逐字节比较。
        self.assertTrue(stream.getvalue().startswith(cleared + "[日志占位]\n\r"))
        redrawn = stream.getvalue().split("\r")[-1]
        self.assertIn("[#---]  25.0% 1/4", redrawn)
        self.assertTrue(redrawn.endswith("2024-03-04"))

    def test_clear_then_close_does_not_emit_a_stray_newline(self) -> None:
        """擦行后行上已无内容，直接结束不应再补一个空行。"""

        stream = io.StringIO()
        bar = ProgressBar(2, "滚动验证", mode="always", stream=stream, min_interval=0.0)
        bar.start()
        bar.clear()
        bar.close()

        self.assertFalse(stream.getvalue().endswith("\n"))

    def test_line_degrades_to_fit_terminal_width(self) -> None:
        """进度行必须逐级降级到终端宽度内，否则折行后回车无法覆盖整行。"""

        def render(columns: str, detail: str) -> str:
            """在指定终端列数下画一帧并返回该帧文本。

            参数：
                columns: 伪造的终端列数字符串。
                detail: 本帧的定位信息。

            返回：
                最后一个回车之后的帧文本。
            """

            stream = io.StringIO()
            with patch.dict(os.environ, {"COLUMNS": columns}):
                bar = ProgressBar(
                    1000, "滚动验证", mode="always", stream=stream, min_interval=0.0
                )
                bar.advance(detail=detail)
            return stream.getvalue().split("\r")[-1].rstrip("\n")

        wide = render("200", "2024-03-05")
        without_detail = render("80", "2024-03-06")
        medium = render("64", "2024-03-04")
        narrow = render("30", "2024-03-03")

        # 宽终端完整显示，说明降级只在必要时发生。
        self.assertIn("2024-03-05", wide)
        self.assertIn("[--", wide)
        # 第一级：只丢定位信息，图形条与计数都保留。
        self.assertLessEqual(display_width(without_detail), 79)
        self.assertNotIn("2024-03-06", without_detail)
        self.assertIn("[--", without_detail)
        self.assertIn("1/1000", without_detail)
        # 第二级：再丢图形条，比例和计数必须保留。
        self.assertLessEqual(display_width(medium), 63)
        self.assertNotIn("2024-03-04", medium)
        self.assertNotIn("[--", medium)
        self.assertIn("1/1000", medium)
        # 第三级：只能硬截断，但绝不允许超出可用列数。
        self.assertLessEqual(display_width(narrow), 29)
        self.assertTrue(narrow.startswith("滚动验证 "))

    def test_narrowing_the_terminal_does_not_overflow_the_new_width(self) -> None:
        """终端在两帧之间被缩窄时，补齐与擦行都不得超出新的列数。"""

        stream = io.StringIO()
        bar = ProgressBar(
            1000, "滚动验证", mode="always", stream=stream, min_interval=0.0
        )
        with patch.dict(os.environ, {"COLUMNS": "200"}):
            bar.advance(detail="2024-03-05")
        stream.seek(0)
        stream.truncate()
        with patch.dict(os.environ, {"COLUMNS": "50"}):
            bar.advance(detail="2024-03-06")
            narrowed = stream.getvalue().split("\r")[-1]
            stream.seek(0)
            stream.truncate()
        self.assertLessEqual(display_width(narrowed), 49)

    def test_clearing_after_the_terminal_narrows_stays_within_the_new_width(self) -> None:
        """缩窄后未重绘就擦行时，擦除长度必须按新列数而不是旧帧宽度。"""

        stream = io.StringIO()
        bar = ProgressBar(
            1000, "滚动验证", mode="always", stream=stream, min_interval=0.0
        )
        with patch.dict(os.environ, {"COLUMNS": "200"}):
            bar.advance(detail="2024-03-05")
        stream.seek(0)
        stream.truncate()
        with patch.dict(os.environ, {"COLUMNS": "50"}):
            bar.clear()

        self.assertLessEqual(display_width(stream.getvalue().strip("\r")), 49)

    def test_truncate_to_width_never_splits_a_full_width_character(self) -> None:
        """按列数截断时全角字符要么整体保留，要么整体丢弃。"""

        self.assertEqual(truncate_to_width("滚动验证", 5), "滚动")
        self.assertEqual(truncate_to_width("滚动验证", 6), "滚动验")
        self.assertEqual(truncate_to_width("ab滚", 3), "ab")
        self.assertEqual(truncate_to_width("ab滚", 4), "ab滚")
        self.assertEqual(truncate_to_width("abc", 0), "")
        self.assertEqual(truncate_to_width("abc", 10), "abc")

    def test_paused_is_inert_when_bar_is_disabled(self) -> None:
        """进度条关闭时暂停上下文不得写入任何字节。"""

        stream = io.StringIO()
        bar = ProgressBar(4, "滚动验证", mode="never", stream=stream)
        bar.clear()
        with bar.paused():
            pass
        bar.clear()

        self.assertEqual(stream.getvalue(), "")

    def test_advance_to_moves_forward_only_and_clamps_to_total(self) -> None:
        """按绝对位置推进时只前进不回退，且不得越过总步数。"""

        stream = io.StringIO()
        bar = ProgressBar(
            10, "单次训练验证", mode="always", stream=stream, min_interval=0.0
        )
        bar.advance_to(4, detail="训练 4/8 轮")
        self.assertEqual(bar.completed, 4)
        bar.advance_to(2, detail="回退")
        self.assertEqual(bar.completed, 4)
        bar.advance_to(99, detail="补齐")
        self.assertEqual(bar.completed, 10)
        bar.close()

        frames = [frame for frame in stream.getvalue().split("\r") if frame]
        # 回退的一次不产生新帧，末帧强制渲染并显示 100%。
        self.assertEqual(len(frames), 2)
        self.assertIn("40.0% 4/10", frames[0])
        self.assertIn("训练 4/8 轮", frames[0])
        self.assertIn("100.0% 10/10", frames[1])
        self.assertNotIn("回退", stream.getvalue())

    def test_rebase_eta_restarts_the_remaining_time_estimate(self) -> None:
        """重设基准后，剩余时间只按新阶段的速度估计，已用时间不受影响。"""

        stream = io.StringIO()
        bar = ProgressBar(
            10, "单次训练验证", mode="always", stream=stream, min_interval=0.0
        )
        bar.advance(detail="预处理")
        before = stream.getvalue().split("\r")[-1]
        bar.rebase_eta()
        with bar.paused():
            pass
        after = stream.getvalue().split("\r")[-1]
        bar.advance(detail="训练 1/8 轮")
        resumed = stream.getvalue().split("\r")[-1]
        bar.close()

        # 重设基准前，1/10 已经可以按全程平均耗时外推剩余时间。
        self.assertIn("10.0% 1/10", before)
        self.assertNotIn("预计剩余 --", before)
        # 重设基准后仍是 1/10，但基准之后还没完成任何一步，无从外推。
        self.assertIn("10.0% 1/10", after)
        self.assertIn("预计剩余 --", after)
        # 新阶段完成一步后恢复外推，且只按新阶段的耗时计算。
        self.assertIn("20.0% 2/10", resumed)
        self.assertNotIn("预计剩余 --", resumed)

    def test_zero_total_disables_bar(self) -> None:
        """没有可展示步数时进度条直接关闭，避免除零和空帧。"""

        stream = _TerminalStream()
        with ProgressBar(0, "滚动验证", mode="always", stream=stream) as bar:
            self.assertFalse(bar.enabled)
            bar.advance()

        self.assertEqual(stream.getvalue(), "")

    def test_closed_bar_stops_writing_and_repeat_close_is_safe(self) -> None:
        """关闭后不再输出进度帧，重复关闭不得追加多余换行。"""

        stream = io.StringIO()
        bar = ProgressBar(2, "滚动验证", mode="always", stream=stream, min_interval=0.0)
        bar.start()
        bar.close()
        bar.close()
        bar.advance()

        self.assertEqual(stream.getvalue().count("\n"), 1)
        self.assertEqual(len(stream.getvalue().split("\r")), 2)

    def test_invalid_arguments_are_rejected(self) -> None:
        """非法模式、节流间隔、条宽和步长必须报错而不是静默降级。"""

        with self.assertRaises(ValueError):
            ProgressBar(1, "滚动验证", mode="on")
        with self.assertRaises(ValueError):
            ProgressBar(1, "滚动验证", min_interval=-1.0)
        with self.assertRaises(ValueError):
            ProgressBar(1, "滚动验证", bar_width=0)
        with self.assertRaises(ValueError):
            ProgressBar(1, "滚动验证", mode="always", stream=io.StringIO()).advance(-1)

    def test_format_duration_covers_all_scales(self) -> None:
        """时长文本应覆盖秒、分秒、时分及不可估计四种情况。"""

        self.assertEqual(format_duration(12.34), "12.3s")
        self.assertEqual(format_duration(201.0), "3m21s")
        self.assertEqual(format_duration(3900.0), "1h05m")
        self.assertEqual(format_duration(float("nan")), "--")
        self.assertEqual(format_duration(-1.0), "--")

    def test_display_width_counts_full_width_characters_twice(self) -> None:
        """东亚全角字符按两列计，ASCII 按一列计。"""

        self.assertEqual(display_width("abc"), 3)
        self.assertEqual(display_width("滚动验证"), 8)
        self.assertEqual(display_width("滚动 abc"), 8)


class ExperimentProgressTest(unittest.TestCase):
    """验证实验按配置开关进度条，且不改变原有训练验证结果。"""

    def setUp(self) -> None:
        """准备最小数据集与验证起点，并固定终端列数。"""

        _pin_terminal_columns(self)
        self.dataset = _build_dataset()
        self.validation_start = self.dataset["target_date"].drop_duplicates().iloc[3]

    def _run(self, stream: io.StringIO, **kwargs: Any) -> ExperimentResult:
        """在重定向的标准错误上运行实验并返回结果。

        参数：
            stream: 接收进度帧的内存流；进度条在实验内部构造时读取该流。
            **kwargs: 透传给 ``DirectionExperiment`` 的关键字参数，例如
                ``progress`` 显示模式与 ``training_mode`` 训练方式。

        返回：
            实验结果对象，供调用方核对进度条未改变预测行为。
        """

        with redirect_stderr(stream):
            return DirectionExperiment(
                validation_start=self.validation_start,
                feature_columns=["factor_a", "factor_b"],
                max_depth=2,
                min_samples_leaf=1,
                **kwargs,
            ).run(self.dataset)

    def test_default_run_writes_no_progress_frames(self) -> None:
        """库层默认不显示进度条，保持既有调用方（含搜索 worker）静默。"""

        stream = io.StringIO()
        result = self._run(stream)

        self.assertEqual(stream.getvalue(), "")
        self.assertGreater(len(result.predictions), 0)

    def test_rolling_mode_renders_progress_to_stderr(self) -> None:
        """滚动模式开启后应逐日刷新进度，并以 100% 收尾。"""

        stream = io.StringIO()
        result = self._run(stream, progress="always")
        output = stream.getvalue()

        self.assertIn("滚动验证", output)
        validation_dates = result.predictions["target_date"].nunique()
        self.assertIn(f"100.0% {validation_dates}/{validation_dates}", output)
        self.assertTrue(output.endswith("\n"))

    def test_single_mode_reports_three_stages(self) -> None:
        """单次模式按预处理、训练、预测三个阶段推进进度。"""

        stream = io.StringIO()
        self._run(stream, progress="always", training_mode="single")
        output = stream.getvalue()

        self.assertIn("单次训练验证", output)
        for stage in ("预处理", "训练", "预测"):
            self.assertIn(stage, output)
        self.assertIn("100.0% 3/3", output)

    def test_progress_keeps_milestone_logs_and_predictions_unchanged(self) -> None:
        """开启进度条不得改变里程碑日志与预测结果，只影响终端呈现。"""

        with self.assertLogs("quant.factor_research.experiment", level="INFO") as logs:
            baseline = self._run(io.StringIO())
        baseline_logs = logs.output

        stream = io.StringIO()
        with self.assertLogs("quant.factor_research.experiment", level="INFO") as logs:
            with_progress = self._run(stream, progress="always")

        self.assertEqual(len(logs.output), len(baseline_logs))
        self.assertTrue(any("滚动验证进度" in message for message in logs.output))
        self.assertTrue(any("滚动验证耗时汇总" in message for message in logs.output))
        pd.testing.assert_frame_equal(baseline.predictions, with_progress.predictions)
        # 写里程碑日志前先擦掉进度行，避免日志接在半行进度帧后面。
        self.assertIn("\r   ", stream.getvalue())

    def test_single_mode_follows_model_training_rounds(self) -> None:
        """模型声明逐轮上报时，训练段应按轮细分而不是整段一步。"""

        stream = io.StringIO()
        # 逐轮推进在生产环境按 0.2 秒节流；桩模型瞬间跑完，关掉节流才能观察到中间帧。
        with patch.object(experiment_module, "DEFAULT_MIN_INTERVAL", 0.0):
            self._run(
                stream,
                progress="always",
                training_mode="single",
                model_factory=_RoundReportingModelFactory(rounds=8),
            )
        output = stream.getvalue()

        # 总步数为 预处理 1 步 + 训练 8 轮 + 预测 1 步。
        self.assertIn("0.0% 0/10", output)
        self.assertIn("训练 3/8 轮", output)
        self.assertIn("100.0% 10/10", output)
        self.assertIn("预测", output)

    def test_single_mode_fills_the_training_segment_when_rounds_fall_short(self) -> None:
        """模型提前收敛、上报轮数不足时，训练段仍要补齐，末帧必须到 100%。"""

        stream = io.StringIO()
        with patch.object(experiment_module, "DEFAULT_MIN_INTERVAL", 0.0):
            self._run(
                stream,
                progress="always",
                training_mode="single",
                model_factory=_RoundReportingModelFactory(rounds=8, reported_rounds=3),
            )
        output = stream.getvalue()

        self.assertIn("训练 3/8 轮", output)
        self.assertIn("100.0% 10/10", output)

    def test_fit_progress_callback_is_uninstalled_after_training(self) -> None:
        """训练结束后必须卸载回调，避免结果里的模型长期持有进度条。"""

        factory = _RoundReportingModelFactory(rounds=4)
        result = self._run(
            io.StringIO(), progress="always", training_mode="single", model_factory=factory
        )

        self.assertIsNone(result.model.installed_callback)
        # 先装后卸，且只装一次。
        self.assertEqual(len(result.model.progress_calls), 2)
        self.assertIsNotNone(result.model.progress_calls[0])
        self.assertIsNone(result.model.progress_calls[1])

    def test_round_reporting_model_stays_silent_when_progress_is_off(self) -> None:
        """搜索并行 worker 的默认路径：即使模型支持逐轮上报也不得有任何输出。"""

        stream = io.StringIO()
        factory = _RoundReportingModelFactory(rounds=8)
        result = self._run(stream, training_mode="single", model_factory=factory)

        self.assertEqual(stream.getvalue(), "")
        # 关闭进度时根本不安装回调，训练调用与不带该能力时完全一致；
        # 只断言最终值为 None 无效，装了又卸同样是 None。
        self.assertEqual(result.model.progress_calls, [])
        self.assertGreater(len(result.predictions), 0)

    def test_debug_level_forces_progress_off(self) -> None:
        """DEBUG 等级逐日写调试日志，会打断进度行，因此强制关闭进度条。"""

        experiment_logger = logging.getLogger("quant.factor_research.experiment")
        previous_level = experiment_logger.level
        experiment_logger.setLevel(logging.DEBUG)
        try:
            stream = io.StringIO()
            self._run(stream, progress="always")
        finally:
            experiment_logger.setLevel(previous_level)

        self.assertNotIn("滚动验证 [", stream.getvalue())

    def test_illegal_declared_rounds_fall_back_to_three_steps(self) -> None:
        """模型声明的轮数不可用时降级为三阶段，绝不能让训练失败。"""

        for declared in (float("inf"), 0, -5, "many", None):
            with self.subTest(declared=declared):
                stream = io.StringIO()
                result = self._run(
                    stream,
                    progress="always",
                    training_mode="single",
                    model_factory=_RoundReportingModelFactory(
                        rounds=8, declared_rounds=declared
                    ),
                )

                self.assertIn("100.0% 3/3", stream.getvalue())
                self.assertGreater(len(result.predictions), 0)

    def test_debug_level_also_skips_the_training_callback(self) -> None:
        """DEBUG 等级关闭进度条时，同样不应给模型安装逐轮回调。"""

        experiment_logger = logging.getLogger("quant.factor_research.experiment")
        previous_level = experiment_logger.level
        experiment_logger.setLevel(logging.DEBUG)
        try:
            result = self._run(
                io.StringIO(),
                progress="always",
                training_mode="single",
                model_factory=_RoundReportingModelFactory(rounds=8),
            )
        finally:
            experiment_logger.setLevel(previous_level)

        self.assertEqual(result.model.progress_calls, [])

    def test_progress_mode_is_validated(self) -> None:
        """非法进度模式必须在构造实验时报错。"""

        with self.assertRaises(ValueError):
            DirectionExperiment(self.validation_start, progress="verbose")
        self.assertEqual(PROGRESS_MODES, ("auto", "always", "never"))


if __name__ == "__main__":
    unittest.main()
