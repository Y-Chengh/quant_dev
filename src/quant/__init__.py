"""量化研究与数据工程的统一命名空间包。

本包只声明版本号，不导入任何子包。子包之间的依赖差异很大：
``qmt_downloader`` 需要在大 QMT 内置 Python 中运行，
``factor_research`` 依赖 scikit-learn 与 lightgbm，
``market_data`` 依赖 duckdb 与 fastapi。
在包入口做即时导入会让轻量场景被迫加载全部三方依赖，因此这里保持空实现。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
