# -*- coding: utf-8 -*-
"""运行器各 mixin 共享的实例状态契约。

``QmtDailyDownloader`` 的方法按职责拆到多个 mixin，它们都读写同一批实例属性。
这些属性由 ``QmtDailyDownloader.__init__`` 一次性建立，本类只作为公共基类存在，
用文档集中说明每项状态的含义，使各 mixin 的依赖显式可见。

本模块运行在大 QMT 内置 Python（早于 3.7）中，因此不使用变量注解，也不写
``from __future__`` 导入；属性说明一律以文档字符串给出。
"""


class _RunnerState(object):
    u"""声明运行器在各 mixin 之间共享的实例属性。

    属性：
        config: 已完成校验的 ``DownloaderConfig``，提供日期范围、数据集选择、
            批大小、并发度和各类开关。
        gateway: 封装大 QMT 内置接口的 ``QmtGateway``，是唯一的数据来源。
        store: 管理日分区和 staging 文件的 ``DailyPartitionStore``。
        checkpoints: 仅保存批次状态的 ``CheckpointStore``，不保存业务数据。
        logger: 同时输出到 QMT 窗口和滚动文件的日志对象。
        issues: 本次运行累计的 ``IssueCollector``，按发现顺序追加问题记录。
        line_corrections: 行级价格修正明细，每项含原值与修正值，最终写入报告。
        symbols: 本次任务实际使用的证券代码列表，已按配置解析和排序。
        batches: 证券池按 ``batch_size`` 切分后的批次列表。
        job_key: 由日期范围、模式、数据集和证券池派生的任务键，用于断点匹配。
        partition_scope: 各数据集分区的证券池范围，用于跳过前的一致性核验。
        started_at: ``run`` 入口的时间戳（``time.time()`` 秒）；未开始时为 ``None``。
    """
