"""由候选表达式生成可直接粘贴执行的主实验复现命令。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..factor_search import SearchContext
from .formatting import _powershell_single_quoted


def _reproduction_command(
    expression: str,
    context: SearchContext,
    metadata: Mapping[str, Any],
) -> str:
    """生成与搜索数据范围、证券池和直出模型一致的主实验命令。

    参数：
        expression: 要在主实验复验的候选因子规范表达式。
        context: 提供 holdout 验证起点的搜索上下文。
        metadata: 可选包含数据库、数据起止日期、证券列表和证券数量上限的元数据。

    返回：
        可直接粘贴到 PowerShell 的单行主实验命令，形如
        ``python -m quant.cli.factor_demo ...``，需在仓库根目录执行。
    """

    parts = [
        "python",
        "-m",
        "quant.cli.factor_demo",
        "--model",
        "factor_passthrough",
        "--task",
        "regression",
        "--training-mode",
        "single",
        "--validation-start",
        _powershell_single_quoted(context.holdout_start.strftime("%Y-%m-%d")),
    ]
    for option, key in (
        ("--database", "database"),
        ("--start", "data_start"),
        ("--end", "data_end"),
    ):
        value = metadata.get(key)
        if value is not None:
            parts.extend([option, _powershell_single_quoted(str(value))])
    codes = list(metadata.get("codes", ()))
    if codes:
        parts.append("--codes")
        parts.extend(_powershell_single_quoted(str(code)) for code in codes)
    elif metadata.get("code_limit") is not None:
        parts.extend(["--symbol-limit", str(metadata["code_limit"])])
    parts.extend(
        [
            "--factors",
            "--factor-expressions",
            _powershell_single_quoted(expression),
        ]
    )
    return " ".join(parts)
