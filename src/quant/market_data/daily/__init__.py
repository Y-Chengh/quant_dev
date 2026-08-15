"""QMT 日线库：存储结构、增量入库与只读查询。

日线库与现有 5 分钟库完全独立，落在自己的 ``qmt_daily.duckdb`` 与
``bars_1d/year=*/month=*/bars.parquet`` 上，互不影响。库中一律保存**原始不复权**
价格与原始除权送转记录，复权在读取层按需计算。

各子模块职责：

- ``models``：复权口径枚举与查询条件数据类。
- ``schema``：列契约、建表 DDL、生命周期哨兵归一化与板块/风险警示推导。
- ``database``：只读 DuckDB 仓储层，全部 SQL 集中于此。
- ``adjust``：由除权记录累乘出前复权与后复权系数。
- ``universe``：无幸存者偏差的证券池筛选。
- ``client``：对外唯一入口。
- ``ingest``：把 QMT 落盘 CSV 增量转换为本库格式的子包。

对外只暴露客户端、查询条件与复权口径；仓储层与各计算函数由使用方从对应子模块
直接导入。
"""

from __future__ import annotations

from .client import DailyMarketClient
from .models import AdjustMode, DailyKlineQuery

__all__ = ["AdjustMode", "DailyKlineQuery", "DailyMarketClient"]
