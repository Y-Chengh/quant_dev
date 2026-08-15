# -*- coding: utf-8 -*-
"""编排批量回溯、日增量、断点恢复和按天分区落盘。

原先的单文件 ``runner.py``（1664 行，其中单个类占 1470 行）按职责拆为：

- ``base``：各 mixin 共享的实例状态契约。
- ``helpers``：生命周期窗口、分批、任务键与问题判定等无状态工具。
- ``instruments``：上市退市信息收集与证券池推导。
- ``issues``：问题级别降级、日线问题过滤与无下载收尾。
- ``progress``：批次与日期进度日志、并行落盘与问题明细输出。
- ``kline``：日线收集、缺口探测与定向补下载、按日分区落盘。
- ``finance_steps``：原始财务抓取、日级快照生成与财务分区落盘。
- ``corporate_actions``：除权除息事件收集与按除权日分区落盘。
- ``partitions``：断点复用判定、通用分区写入与整日水位标记。
- ``downloader``：编排层，建立共享状态并按固定顺序驱动各 mixin。

本包位于大 QMT 内置 Python（早于 3.7）的导入链上，因此包内所有模块都不得写
``from __future__`` 导入或变量注解，类一律显式继承 ``object``。该约束由
``test_qmt_import_chain_avoids_future_annotations`` 守护。

公开接口与拆分前完全一致，调用方无需改动导入语句。
"""

from .downloader import QmtDailyDownloader

__all__ = ["QmtDailyDownloader"]
