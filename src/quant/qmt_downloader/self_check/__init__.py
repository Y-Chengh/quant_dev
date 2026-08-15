# -*- coding: utf-8 -*-
"""对大 QMT 按日落盘行情执行全样本内部质量审计。

原先的单文件 ``self_check.py``（2494 行，其中单个类占 2050 行）按职责拆为：

- ``models``：问题记录、审计结果与配置数据类。
- ``base``：各 mixin 共享的实例状态契约。
- ``utils``：与自检器状态无关的解析、校验与文件摘要工具。
- ``writers``：报告文件的原子写入与摘要 Markdown 渲染。
- ``discovery``：分区发现、日线根目录定位与分区元数据读取。
- ``loading``：生命周期、交易日历、证券池与除权事件装载。
- ``scanning``：按交易日扫描分区并读取日线与 staging 数据。
- ``validation``：行级字段校验、数值异常检测与连续性校验。
- ``summary``：摘要汇总与报告落盘。
- ``checker``：编排层，建立共享状态并按固定顺序驱动各 mixin。

本包只在外部 Python 的 ``quant-qmt-self-check`` 中使用，不在大 QMT 的导入链上，
因此可以使用 ``from __future__ import annotations`` 等新版本语法。

公开接口与拆分前完全一致，调用方无需改动导入语句。
"""

from __future__ import annotations

from .checker import QmtDataSelfChecker, run_full_sample_self_check
from .models import ISSUE_COLUMNS, AuditIssue, AuditResult, SelfCheckConfig

__all__ = [
    "ISSUE_COLUMNS",
    "AuditIssue",
    "AuditResult",
    "QmtDataSelfChecker",
    "SelfCheckConfig",
    "run_full_sample_self_check",
]
