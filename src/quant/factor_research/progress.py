"""长循环阶段的单行终端进度条。

模型训练验证在滚动模式下要对每个验证日重新训练一次，全量运行常常需要几分钟到
几十分钟。此前只在完成 10% 时写一行日志，粒度粗且不给剩余时间，两条日志之间
终端长时间静默，无法区分"仍在训练"和"已经卡住"。本模块提供不依赖三方库的单行
进度条：在终端里用回车覆盖同一行，持续显示完成比例、已用时间和线性外推的预计
剩余时间；在重定向、CI 等非终端环境下自动关闭，避免把成百上千个进度帧写进日志
文件或报告。

进度条默认写 ``sys.stderr``，与命令行的控制台日志同一个流，因此二者不会互相错位
到不同目标；同时意味着 ``2>`` 重定向会一起带走日志和进度帧，此时 ``auto`` 模式
检测到目标不是终端会自动退化为完全不输出。
"""

from __future__ import annotations

import math
import shutil
import sys
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from types import TracebackType
from typing import TextIO

PROGRESS_MODES = ("auto", "always", "never")
"""进度条显示模式：``auto`` 仅在交互式终端显示，``always`` 强制显示，``never`` 关闭。

取值刻意避开 ``on``/``off``：YAML 1.1 会把这两个裸词解析成布尔值，写进 YAML
配置时会变成 ``True``/``False`` 而被参数校验拒绝。
"""

DEFAULT_BAR_WIDTH = 24
"""进度条方括号内的字符数；固定宽度保证填充增长时进度条本身不左右抖动。

整行是否折行由终端列数决定，与该常量无关，因此渲染时还会按当前列数截断。
"""

DEFAULT_MIN_INTERVAL = 0.2
"""两次重绘之间的最小间隔秒数；首帧和末帧不受该节流限制。"""

FALLBACK_TERMINAL_COLUMNS = 80
"""取不到终端列数时假定的宽度；按最保守的经典终端宽度处理。"""

_FILLED_CHAR = "#"
_EMPTY_CHAR = "-"
# 只用 ASCII 画进度条：Windows 终端常用 GBK 代码页，方块类字符要么无法编码
# （如 U+2591），要么按东亚宽度渲染成两列，会让进度条随填充增长左右抖动。


def display_width(text: str) -> int:
    """估算文本在等宽终端中占用的列数。

    参数：
        text: 待测量的单行文本；东亚全角字符按两列计，其余按一列计。

    返回：
        文本占用的终端列数，用于计算覆盖上一帧所需补齐的空格数。
    """

    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def truncate_to_width(text: str, limit: int) -> str:
    """按显示列数截断文本，保证结果不超过给定列宽。

    参数：
        text: 待截断的单行文本。
        limit: 允许占用的最大列数；小于等于零时返回空串。

    返回：
        显示宽度不超过 ``limit`` 的前缀；全角字符不会被截成半个字符宽。
    """

    if limit <= 0:
        return ""
    width = 0
    for index, char in enumerate(text):
        char_width = 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if width + char_width > limit:
            return text[:index]
        width += char_width
    return text


def format_duration(seconds: float) -> str:
    """把秒数格式化为终端友好的时长文本。

    参数：
        seconds: 时长秒数；NaN、无穷或负数表示时长不可估计。

    返回：
        不可估计时返回 ``--``；不足 1 分钟返回 ``12.3s``；不足 1 小时返回
        ``3m21s``；否则返回 ``1h05m``。
    """

    if not math.isfinite(seconds) or seconds < 0.0:
        return "--"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h{remaining_minutes:02d}m"


class ProgressBar:
    """在同一行滚动显示长循环的完成比例、已用时间与预计剩余时间。

    典型用法是配合 ``with`` 语句：进入时渲染 0% 初始帧并把计时起点对齐到该帧，
    每完成一个元素调用一次 :meth:`advance`，退出时补一个换行，让后续日志从新行
    开始。进度条被关闭（非终端、``never`` 模式或总步数为零）时，全部方法都不会向
    流写入任何字节，调用方无需再做分支判断。
    """

    def __init__(
        self,
        total: int,
        description: str,
        mode: str = "auto",
        stream: TextIO | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        bar_width: int = DEFAULT_BAR_WIDTH,
    ) -> None:
        """保存本次进度条的总量、显示模式和输出目标，并解析是否实际输出。

        参数：
            total: 本次循环的总步数，例如滚动验证的验证日数量；小于等于零表示
                没有可展示的进度，进度条直接关闭。
            description: 进度条行首的阶段名称，例如 ``滚动验证``、``单次训练验证``。
            mode: :data:`PROGRESS_MODES` 之一。``auto`` 只在输出流是交互式终端时
                显示，``always`` 无条件显示（供测试和强制输出使用），``never`` 关闭。
            stream: 进度帧的输出目标；缺省在构造时取 ``sys.stderr``，与命令行的
                控制台日志同流。
            min_interval: 两次重绘之间的最小间隔秒数，避免高频循环把终端刷爆；
                首帧与末帧强制重绘，不受该值影响。
            bar_width: 方括号内的字符数，必须为正整数。

        返回：
            无返回值；是否输出在构造时确定，之后不再随流状态变化。
        """

        if mode not in PROGRESS_MODES:
            raise ValueError(
                f"progress 模式必须是 {PROGRESS_MODES} 之一，实际为 {mode!r}"
            )
        if not math.isfinite(min_interval) or min_interval < 0.0:
            raise ValueError(
                f"min_interval 必须是非负有限秒数，实际为 {min_interval!r}"
            )
        if int(bar_width) < 1:
            raise ValueError(f"bar_width 必须是正整数，实际为 {bar_width!r}")
        self.total = max(int(total), 0)
        self.description = description
        self.min_interval = float(min_interval)
        self.bar_width = int(bar_width)
        self._stream = sys.stderr if stream is None else stream
        self._enabled = self._resolve_enabled(mode)
        self._completed = 0
        self._started_at = perf_counter()
        # 首帧必须绕过节流，因此把上次重绘时间初始化为负无穷。
        self._last_render_at = float("-inf")
        self._last_line_width = 0
        self._last_detail = ""
        self._rendered = False
        self._closed = False

    @property
    def enabled(self) -> bool:
        """返回进度条是否会真正输出进度帧。

        返回：
            ``True`` 表示后续调用会向流写入进度帧。启停在构造时一次性解析，
            之后不再变化；调用方可据此为关闭场景准备替代输出。
        """

        return self._enabled

    @property
    def completed(self) -> int:
        """返回已完成步数；即使进度条关闭也照常累计，便于调用方复用计数。"""

        return self._completed

    def _resolve_enabled(self, mode: str) -> bool:
        """按显示模式、总步数和流的可交互性判断是否输出进度帧。

        参数：
            mode: 已校验过取值的显示模式。

        返回：
            ``never`` 模式、总步数为零、流不存在或已关闭时为 ``False``；``always``
            模式为 ``True``；``auto`` 模式取决于流是否为终端。
        """

        if mode == "never" or self.total <= 0:
            return False
        stream = self._stream
        if stream is None or getattr(stream, "closed", False):
            return False
        if mode == "always":
            return True
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty):
            return False
        try:
            return bool(isatty())
        except ValueError:
            # 流在构造与查询之间被关闭时 isatty 会抛错，等同于不可交互。
            return False

    def start(self) -> None:
        """渲染 0% 初始帧，并把计时起点对齐到该帧。

        返回：
            无返回值。计时起点在这里重置，因此构造与真正进入循环之间的准备
            耗时不会被摊进单步平均耗时，避免首帧外推出离谱的剩余时间。
        """

        if not self._enabled or self._closed:
            return
        self._started_at = perf_counter()
        # 初始帧不带定位信息，重置记录，避免 paused() 重画出上一轮的旧信息。
        self._last_detail = ""
        self._render(force=True)

    def advance(self, step: int = 1, detail: str = "") -> None:
        """累计已完成步数并按节流重绘进度帧。

        参数：
            step: 本次完成的步数，必须为非负整数；累计值封顶到 ``total``。
            detail: 追加在进度帧末尾的定位信息，例如当前验证日期；为空则不追加。

        返回：
            无返回值。达到总步数时强制重绘，保证末帧一定显示 100%。
        """

        if int(step) < 0:
            raise ValueError(f"step 必须是非负整数，实际为 {step!r}")
        self._completed = min(self._completed + int(step), self.total)
        if not self._enabled or self._closed:
            return
        # 记录在渲染之前：本次可能被节流跳过，但 paused() 重画时应使用最新的
        # 定位信息，而不是上一个真正落笔的帧所带的旧信息。
        self._last_detail = detail
        self._render(force=self._completed >= self.total, detail=detail)

    def clear(self) -> None:
        """擦掉当前进度行，把光标留在行首。

        返回：
            无返回值；进度条未开启、已关闭或还没画过任何帧时不做任何事。调用后
            调用方可以在同一个流上写完整的一行文本，不会接在半行进度帧后面。
        """

        if not self._enabled or self._closed or not self._rendered:
            return
        # 上一帧画完后终端可能被缩窄，擦除长度同样要受当前列数约束，否则擦行本身
        # 就会折行，在屏幕上留下多余的空白行。
        erased = min(self._last_line_width, self._terminal_columns() - 1)
        self._stream.write("\r" + " " * erased + "\r")
        self._stream.flush()
        self._last_line_width = 0
        # 行上已无内容，因此复位渲染标记：此后直接 close() 不应再补一个空行。
        self._rendered = False

    @contextmanager
    def paused(self) -> Iterator[None]:
        """在上下文内让出终端行，退出时按最近一次内容重画进度帧。

        返回：
            上下文管理器，没有产出值。进度条与日志写同一个流，日志必须从行首开始
            输出；退出时强制重画，保证日志滚过之后进度条仍然停在最后一行。
        """

        # clear() 会复位渲染标记，因此先记下进入前是否已经有进度行需要恢复。
        rendered = self._rendered
        self.clear()
        try:
            yield
        finally:
            if rendered and self._enabled and not self._closed:
                self._render(force=True, detail=self._last_detail)

    def close(self) -> None:
        """结束进度条：已经画过进度帧时补一个换行并停止后续输出。

        返回：
            无返回值；重复调用安全，关闭后的 :meth:`advance` 仍累计计数但不再输出。
        """

        if self._closed:
            return
        self._closed = True
        if not self._rendered:
            return
        self._stream.write("\n")
        self._stream.flush()

    def _terminal_columns(self) -> int:
        """返回当前终端列数，取不到时按经典宽度处理。

        返回：
            终端列数，至少 20 列。``COLUMNS`` 环境变量优先，其次向标准输出查询；
            重定向或无控制台时回落到 :data:`FALLBACK_TERMINAL_COLUMNS`。
        """

        try:
            columns = shutil.get_terminal_size(
                fallback=(FALLBACK_TERMINAL_COLUMNS, 24)
            ).columns
        except (OSError, ValueError):
            columns = FALLBACK_TERMINAL_COLUMNS
        return max(int(columns), 20)

    def _render(self, force: bool, detail: str = "") -> None:
        """把当前进度渲染为一行文本并覆盖写回流。

        参数：
            force: 为 ``True`` 时忽略 ``min_interval`` 节流，用于首帧和末帧。
            detail: 追加在行尾的定位信息；为空则不追加。

        返回：
            无返回值。写入前用回车把光标移回行首，并按上一帧的显示宽度补齐空格，
            确保较短的新帧不会残留上一帧的尾部字符。整行还会按当前终端列数截断：
            一旦折行，回车只能回到最后一个视觉行的行首，进度条会逐行堆积，
            :meth:`clear` 的擦行也会跨行乱写。
        """

        now = perf_counter()
        if not force and now - self._last_render_at < self.min_interval:
            return
        elapsed = now - self._started_at
        ratio = self._completed / self.total if self.total else 1.0
        filled = (
            self.bar_width
            if self._completed >= self.total
            else int(ratio * self.bar_width)
        )
        # 线性外推：已完成步的平均耗时乘以剩余步数；一步都没完成时无从估计。
        remaining = (
            elapsed / self._completed * (self.total - self._completed)
            if self._completed > 0
            else float("nan")
        )
        head = f"{self.description} "
        gauge = f"[{_FILLED_CHAR * filled}{_EMPTY_CHAR * (self.bar_width - filled)}] "
        body = (
            f"{ratio * 100.0:5.1f}% {self._completed}/{self.total} "
            f"已用 {format_duration(elapsed)} 预计剩余 {format_duration(remaining)}"
        )
        # 留出最后一列不写，避免部分终端在写满整行时立刻自动换行。
        # 一旦折行，回车只回到最后一个视觉行的行首，进度条会逐行往下堆积。
        limit = self._terminal_columns() - 1
        line = head + gauge + body
        if detail and display_width(line) + 1 + display_width(detail) <= limit:
            line = f"{line} {detail}"
        elif display_width(line) > limit:
            # 窄终端按信息量逐级降级：先丢定位信息，再丢图形条，最后才硬截断。
            line = head + body
            if display_width(line) > limit:
                line = truncate_to_width(line, limit)
        width = display_width(line)
        # 补齐只为覆盖上一帧的残留，长度同样受当前列数约束：终端在两帧之间被缩窄
        # 时，按旧宽度补空格会把本帧顶出可用列宽而折行。
        padding = max(min(self._last_line_width, limit) - width, 0)
        self._stream.write("\r" + line + " " * padding)
        self._stream.flush()
        # 只记本帧文本宽度：补齐区已经是空格，下一帧无需再覆盖一次。
        self._last_line_width = width
        self._last_render_at = now
        self._rendered = True

    def __enter__(self) -> ProgressBar:
        """进入上下文时渲染初始帧。

        返回：
            进度条自身，便于 ``with ... as progress`` 后直接调用 :meth:`advance`。
        """

        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出上下文时结束进度条，异常照常向外传播。

        参数：
            exc_type: 触发退出的异常类型；正常退出时为 ``None``。
            exc_value: 触发退出的异常实例；正常退出时为 ``None``。
            traceback: 触发退出的异常回溯；正常退出时为 ``None``。

        返回：
            无返回值，即不吞掉循环体内抛出的异常。
        """

        self.close()
