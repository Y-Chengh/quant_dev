# -*- coding: utf-8 -*-
"""自检各阶段与长循环的进度日志。

自检要读遍全部日分区并逐日校验行级数据，全量运行可达数十分钟。没有进度输出时
终端在整个过程中完全静默，无法区分“正在扫描”和“卡住”。本模块统一负责阶段起止、
耗时和百分比进度的输出，保证问题日志与进度日志格式一致。
"""

from __future__ import annotations

import time

from .base import _CheckerState


class _ProgressLoggingMixin(_CheckerState):
    """输出自检阶段起止耗时与长循环的百分比进度。"""

    def _begin_phase(self, name: str) -> None:
        """记录一个自检阶段开始并重置该阶段计时器。

        参数：
            name: 阶段名称，如 ``发现日线分区``、``逐日扫描``；同时用于阶段完成日志。

        返回：
            无返回值；阶段计时器供 ``_end_phase`` 与 ``_log_step_progress`` 共用。
        """

        self._phase_name = name
        self._phase_started_at = time.monotonic()
        self.logger.info("[自检] 阶段开始 %s", name)

    def _end_phase(self, detail: str = "") -> None:
        """记录当前阶段结束、耗时与统计明细。

        参数：
            detail: 该阶段的结果摘要，如“分区 1200 个”；缺省只输出阶段名与耗时。

        返回：
            无返回值。
        """

        elapsed = time.monotonic() - self._phase_started_at
        if detail:
            self.logger.info(
                "[自检] 阶段完成 %s 耗时 %.1fs %s", self._phase_name, elapsed, detail
            )
        else:
            self.logger.info("[自检] 阶段完成 %s 耗时 %.1fs", self._phase_name, elapsed)

    def _log_step_progress(
        self,
        stage: str,
        index: int,
        total: int,
        detail: str = "",
        pending: bool = False,
    ) -> None:
        """按固定步长输出长循环的进度、已用时间和预计剩余时间。

        参数：
            stage: 循环名称，如 ``分区校验``、``逐日扫描``、``除权事件``。
            index: 当前元素在本轮循环中的零基下标。
            total: 本轮循环的元素总数；小于 1 时按 1 处理。
            detail: 当前元素的定位信息，如 ``date=20260810``；缺省不输出。
            pending: 调用点位于循环体开头、当前元素尚未处理时传 ``True``，剩余元素
                因此比在循环体末尾调用时多一个；带 ``continue`` 分支的循环只能在
                开头打点，故需要显式区分，否则末条进度会少算一个元素的时间。

        返回：
            无返回值；只有首个、最后一个以及每前进一个步长的元素写 INFO，其余
            降级为 DEBUG。步长取 ``progress_every``，为 0 时按总量的 5% 自动选择，
            避免上万条逐项日志把真正的问题淹没。
        """

        total = max(int(total), 1)
        current = min(int(index) + 1, total)
        # 一个阶段里可能先做一段与循环无关的准备工作（如先解析生命周期再读除权
        # 事件），因此循环耗时按本循环自己的起点计时，避免首条进度把准备时间当成
        # 单个元素的耗时、外推出荒谬的剩余时间。
        if stage != self._step_stage or current <= 1:
            self._step_stage = stage
            self._step_started_at = time.monotonic()
        configured = int(self.config.progress_every)
        step = configured if configured > 0 else max(total // 20, 1)
        milestone = current == 1 or current == total or current % step == 0
        write_log = self.logger.info if milestone else self.logger.debug
        elapsed = time.monotonic() - self._step_started_at
        # 计时起点是首条进度，因此不论打点位置，此刻都恰好完成了 current-1 个元素；
        # 用它们的平均耗时线性外推剩余时间，各元素工作量相近，足以判断还要等几分钟。
        completed = current - 1
        outstanding = total - current + 1 if pending else total - current
        remaining_text = (
            "未知"
            if completed <= 0
            else "{0:.1f}s".format(elapsed / completed * outstanding)
        )
        write_log(
            "[自检] %s %d/%d (%.1f%%) 已用 %.1fs 预计剩余 %s%s",
            stage,
            current,
            total,
            current * 100.0 / total,
            elapsed,
            remaining_text,
            " " + detail if detail else "",
        )
