"""项目级路径与环境变量的统一解析入口。

任何模块都不应再内联书写 ``D:\\量化\\market.duckdb`` 之类的绝对路径，
一律通过本包获取，确保换机器或换盘符时只需改动一处。
"""

from __future__ import annotations

from .paths import (
    MARKET_DATABASE_ENV,
    PROJECT_ROOT_ENV,
    configs_dir,
    default_factor_config_path,
    default_market_database,
    default_qmt_config_path,
    project_root,
)

__all__ = [
    "MARKET_DATABASE_ENV",
    "PROJECT_ROOT_ENV",
    "configs_dir",
    "default_factor_config_path",
    "default_market_database",
    "default_qmt_config_path",
    "project_root",
]
