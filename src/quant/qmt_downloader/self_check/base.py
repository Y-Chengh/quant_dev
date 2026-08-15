# -*- coding: utf-8 -*-
"""自检各 mixin 共享的实例状态契约。

``QmtDataSelfChecker`` 的方法按职责拆到多个 mixin，它们都读写同一批实例属性。
这些属性由 ``QmtDataSelfChecker.__init__`` 一次性建立，本类只声明类型不做赋值，
使每个 mixin 的依赖显式可见，也让类型检查器能解析跨模块的属性引用。

本类不可单独实例化使用；它只作为 mixin 的公共基类存在。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AuditIssue, SelfCheckConfig


class _CheckerState:
    """声明自检器在各 mixin 之间共享的实例属性。"""

    #: 数据根目录、审计日期范围、交易日历与异常阈值配置。
    config: SelfCheckConfig
    #: 已解析为绝对路径的输出根目录。
    root: Path
    #: 实际读取数据的根目录；staging 模式下指向作业目录而非输出根目录。
    data_root: Path
    #: 数据来源标记，取值为 ``partitions`` 或 ``staging``。
    source_mode: str
    #: 本次审计累计的全部问题记录，按发现顺序追加。
    issues: list[AuditIssue]
    #: 各业务数据集名称到其分区根目录的映射。
    partition_paths: dict[str, Path]
    #: 首个已完成分区的证券池范围，作为后续分区的一致性基准；无基准时为 None。
    reference_scope: dict[str, Any] | None
    #: 已登记过生命周期问题的 ``(证券代码, 问题码)``，用于抑制重复告警。
    _lifecycle_issue_keys: set[tuple[str, str]]
    #: staging 模式下按日期记录的缺口证券列表。
    staging_daily_gap_dates: list[tuple[str, list[str]]]
