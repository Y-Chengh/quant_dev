"""解析项目根目录、配置目录和行情库路径。

优先级统一为：显式传参 > 环境变量 > 内置回退值。
所有函数都在调用时读取环境变量，因此测试可以用 ``monkeypatch`` 或
``unittest.mock.patch.dict`` 临时改写而无需重新导入模块。
"""

from __future__ import annotations

import os
from pathlib import Path

#: 覆盖行情库路径的环境变量名。
MARKET_DATABASE_ENV = "MARKET_DB_PATH"

#: 覆盖项目根目录的环境变量名；非可编辑安装或部署到别处时使用。
PROJECT_ROOT_ENV = "QUANT_PROJECT_ROOT"

#: 覆盖 QMT 日线库路径的环境变量名。
QMT_DAILY_DATABASE_ENV = "QMT_DAILY_DB_PATH"

#: 覆盖大 QMT 下载器落盘根目录的环境变量名。
QMT_OUTPUT_ROOT_ENV = "QMT_OUTPUT_ROOT"

#: 未设置 ``MARKET_DB_PATH`` 时使用的行情库回退路径。
FALLBACK_MARKET_DATABASE = Path(r"D:\量化\market.duckdb")

#: 未设置 ``QMT_DAILY_DB_PATH`` 时使用的日线库回退路径。
FALLBACK_QMT_DAILY_DATABASE = Path(r"D:\量化\qmt_daily\qmt_daily.duckdb")

#: 未设置 ``QMT_OUTPUT_ROOT`` 时使用的下载器落盘根目录回退路径。
FALLBACK_QMT_OUTPUT_ROOT = Path(r"D:\qmt_kline_test1")

# 本文件位于 <项目根>/src/quant/config/paths.py，向上四级即项目根目录。
_REPO_ROOT_FROM_SOURCE = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    """返回项目根目录，即包含 ``configs/``、``docs/`` 和 ``src/`` 的目录。

    返回：
        绝对路径。设置了 ``QUANT_PROJECT_ROOT`` 时以其为准，
        否则按本文件在 src layout 中的位置向上推算。
    """
    override = os.getenv(PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT_FROM_SOURCE


def configs_dir() -> Path:
    """返回配置样例根目录 ``<项目根>/configs``。

    返回：
        绝对路径；本函数不校验目录是否存在。
    """
    return project_root() / "configs"


def default_market_database() -> Path:
    """返回默认行情库文件路径。

    返回：
        环境变量 ``MARKET_DB_PATH`` 指向的路径；未设置时返回
        ``D:\\量化\\market.duckdb``。本函数不校验文件是否存在。
    """
    value = os.getenv(MARKET_DATABASE_ENV)
    if value:
        return Path(value)
    return FALLBACK_MARKET_DATABASE


def default_qmt_daily_database() -> Path:
    """返回默认 QMT 日线库文件路径。

    返回：
        环境变量 ``QMT_DAILY_DB_PATH`` 指向的路径；未设置或为空字符串时返回
        ``D:\\量化\\qmt_daily\\qmt_daily.duckdb``。本函数不校验文件是否存在。
    """
    value = os.getenv(QMT_DAILY_DATABASE_ENV)
    if value:
        return Path(value)
    return FALLBACK_QMT_DAILY_DATABASE


def default_qmt_daily_root() -> Path:
    """返回默认 QMT 日线库的数据根目录。

    该目录同时存放 ``qmt_daily.duckdb``、``bars_1d`` 月度 Parquet 分片、
    ``reports`` 审计报告和 ``.sync.lock`` 同步锁。

    返回：
        ``default_qmt_daily_database()`` 所在目录；本函数不创建也不校验该目录。
    """
    return default_qmt_daily_database().parent


def default_qmt_output_root() -> Path:
    """返回默认的大 QMT 下载器落盘根目录。

    该目录下是下载器写出的 ``kline_1d``、``instrument_info``、
    ``trading_calendar``、``corporate_actions`` 等按日 CSV 分区。

    返回：
        环境变量 ``QMT_OUTPUT_ROOT`` 指向的路径；未设置或为空字符串时返回
        ``D:\\qmt_kline_test1``。本函数不校验目录是否存在。
    """
    value = os.getenv(QMT_OUTPUT_ROOT_ENV)
    if value:
        return Path(value)
    return FALLBACK_QMT_OUTPUT_ROOT


def default_factor_config_path(name: str) -> Path:
    """拼接因子研究配置样例的完整路径。

    参数：
        name: ``configs/factor_research/`` 下的文件名，例如 ``example.yaml``；
            必须是纯文件名，不接受目录分隔符。

    返回：
        绝对路径；本函数不校验文件是否存在。
    """
    return configs_dir() / "factor_research" / name


def default_qmt_config_path(name: str) -> Path:
    """拼接大 QMT 下载器配置样例的完整路径。

    参数：
        name: ``configs/qmt_downloader/`` 下的文件名，例如
            ``kline_only.backfill.json``；必须是纯文件名，不接受目录分隔符。

    返回：
        绝对路径；本函数不校验文件是否存在。
    """
    return configs_dir() / "qmt_downloader" / name
