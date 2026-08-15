"""支持多输入表达式、长度惩罚和并行评价的遗传编程因子搜索。

原先的单文件 ``genetic.py``（1510 行）按职责拆为：

- ``config``：默认算子参数与搜索配置校验。
- ``events``：搜索结果、进度事件与候选出身记录。
- ``session``：把执行后端包装为可上报阶段进度与 ETA 的会话。
- ``trees``：表达式树的路径枚举、子树定位与替换。
- ``generator``：受配置约束的表达式随机生成、交叉与变异。
- ``search``：适应度评价、去重、选择与逐代进化的主流程。

本模块保持与拆分前完全一致的公开接口，``factor_search`` 包的再导出无需改动。
包内私有实现请从对应子模块直接导入，不在此处再导出。
"""

from __future__ import annotations

from .config import GeneticSearchConfig
from .events import GeneticProgressCallback, GeneticProgressEvent, GeneticSearchResult
from .search import FactorGeneticSearch

__all__ = [
    "FactorGeneticSearch",
    "GeneticProgressCallback",
    "GeneticProgressEvent",
    "GeneticSearchConfig",
    "GeneticSearchResult",
]
