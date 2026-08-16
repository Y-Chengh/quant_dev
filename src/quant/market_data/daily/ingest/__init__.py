"""把大 QMT 落盘的日线 CSV 增量转换为日线库的月度 Parquet 与目录表。

各子模块职责：

- ``source``：只读扫描 QMT 落盘目录，解析源根目录、枚举分区、读取完成标记。
- ``state``：``ingest_state`` 与 ``dataset_metadata`` 的读写。
- ``detect``：四级轻量增量检查，产出 ``SourceDelta``。
- ``filter_reasons``：入库过滤原因的取值、说明文案与汇总渲染。
- ``shards``：``bars_1d`` 月度 Parquet 的重建与原子替换，含上市日过滤。
- ``aux_tables``：证券生命周期、交易日历与除权送转三张辅助表的刷新。
- ``catalog``：``bars_1d`` 视图、月度库存、证券库存与同步运行记录。
- ``locking``：``.sync.lock`` 文件锁与 DuckDB 写锁冲突的降级处理。
- ``sync``：同步主流程的编排。
- ``cli_support``：同步相关命令行参数的注册与装配。

对外只暴露编排层与命令行支撑；分区扫描、分片重写等实现细节由使用方从对应
子模块直接导入。
"""

from __future__ import annotations

from .cli_support import add_sync_arguments, sync_from_args
from .locking import DailyStoreLockedError
from .sync import SYNC_MODES, DailySyncConfig, SyncReport, sync_daily_store

__all__ = [
    "SYNC_MODES",
    "DailyStoreLockedError",
    "DailySyncConfig",
    "SyncReport",
    "add_sync_arguments",
    "sync_daily_store",
    "sync_from_args",
]
