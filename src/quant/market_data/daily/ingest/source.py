"""大 QMT 落盘目录的只读扫描：根目录解析、数据集描述与分区枚举。

本模块刻意不使用 ``qmt_downloader.storage.DailyPartitionStore``：那个类在构造时
会创建 ``staging/`` 与 ``reports/`` 目录，而入库侧对源目录只应该读、不应该写。
分区完整性校验仍然复用下载器公开的 ``validate_partition_directory``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from quant.config import (
    QMT_OUTPUT_ROOT_ENV,
    default_qmt_config_path,
    default_qmt_output_root,
)
from quant.qmt_downloader.config import strip_jsonc
from quant.qmt_downloader.gateway import (
    CORPORATE_ACTION_COLUMNS,
    INSTRUMENT_INFO_COLUMNS,
    KLINE_COLUMNS,
)

#: 交易日历快照的列。刻意在这里本地声明而不从 ``qmt_downloader.runner`` 导入：
#: ``runner`` 是下载器的内部 mixin 包，不在 AGENTS.md 允许的跨包依赖清单内，
#: 而且日历落表功能仍在开发中，直接依赖它会让本模块跟着它一起动。
TRADING_CALENDAR_COLUMNS = ("trade_date", "calendar_symbol")

#: 数据集名，与下载器写出的目录名一致。
DATASET_KLINE = "kline_1d"
DATASET_INSTRUMENTS = "instrument_info"
DATASET_CALENDAR = "trading_calendar"
DATASET_ACTIONS = "corporate_actions"


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    """描述一个 QMT 数据集在磁盘上的组织方式。

    参数：
        name: 数据集名，同时是源根目录下的子目录名。
        partition_name: 分区字段名，例如 ``date``、``ex_date`` 或 ``snapshot``。
        columns: 该数据集 ``data.csv`` 的规范列顺序，来自下载器的列常量。
        date_partitioned: 是否按交易日分区。``False`` 表示只有一个整体快照分区。
    """

    name: str
    partition_name: str
    columns: tuple[str, ...]
    date_partitioned: bool


#: 入库关心的四个数据集；财务数据本次不接。
DATASET_LAYOUTS: tuple[DatasetLayout, ...] = (
    DatasetLayout(DATASET_INSTRUMENTS, "snapshot", tuple(INSTRUMENT_INFO_COLUMNS), False),
    DatasetLayout(DATASET_CALENDAR, "snapshot", tuple(TRADING_CALENDAR_COLUMNS), False),
    DatasetLayout(DATASET_KLINE, "date", tuple(KLINE_COLUMNS), True),
    DatasetLayout(DATASET_ACTIONS, "ex_date", tuple(CORPORATE_ACTION_COLUMNS), True),
)

DATASET_BY_NAME: dict[str, DatasetLayout] = {layout.name: layout for layout in DATASET_LAYOUTS}


@dataclass(frozen=True, slots=True)
class SourcePartition:
    """一个源分区的定位信息与文件指纹。

    参数：
        dataset: 所属数据集名。
        partition_key: 分区目录名，例如 ``date=20240102`` 或 ``snapshot=latest``。
        partition_value: 分区值，即 ``partition_key`` 等号右侧部分。
        directory: 分区目录的绝对路径。
        marker_mtime_ns: ``_SUCCESS.json`` 的修改时间纳秒数；标记缺失时为 ``-1``。
        marker_size: ``_SUCCESS.json`` 的字节数；标记缺失时为 ``-1``。
        data_mtime_ns: ``data.csv`` 的修改时间纳秒数；文件缺失时为 ``-1``。
        data_size: ``data.csv`` 的字节数；文件缺失时为 ``-1``。
        content_sha256: 完成标记里记录的 ``data.csv`` 字节摘要；未读取标记时为空串。
        identity_sha256: 完成标记里记录的业务主键摘要；不存在时为空串。
        source_rows: 完成标记里记录的行数；未读取标记时为 ``-1``。
        completed_at: 完成标记里记录的落盘时间文本；未读取标记时为空串。
    """

    dataset: str
    partition_key: str
    partition_value: str
    directory: Path
    marker_mtime_ns: int
    marker_size: int
    data_mtime_ns: int
    data_size: int
    content_sha256: str = ""
    identity_sha256: str = ""
    source_rows: int = -1
    completed_at: str = ""

    @property
    def data_path(self) -> Path:
        """返回该分区 ``data.csv`` 的绝对路径。"""
        return self.directory / "data.csv"

    @property
    def marker_path(self) -> Path:
        """返回该分区 ``_SUCCESS.json`` 的绝对路径。"""
        return self.directory / "_SUCCESS.json"

    def month(self) -> tuple[int, int] | None:
        """返回该分区所属的年月，供按月重写 Parquet 使用。

        返回：
            ``(年, 月)`` 二元组；非日期分区或分区值不是八位日期时返回 ``None``。
        """
        value = self.partition_value
        if len(value) != 8 or not value.isdigit():
            return None
        return int(value[:4]), int(value[4:6])


def resolve_source_root(
    explicit: Path | str | None = None,
    config_path: Path | str | None = None,
) -> Path:
    """解析大 QMT 落盘根目录。

    优先级：显式传参 > ``QMT_OUTPUT_ROOT`` 环境变量 > 下载器 JSONC 配置里的
    ``output_root`` > 内置回退值。这与 ``quant.cli.qmt_self_check.load_config_defaults``
    读取的是同一份配置、同一套 JSONC 解析，两者必须保持一致。

    参数：
        explicit: 命令行显式指定的根目录；``None`` 表示继续按后续优先级解析。
        config_path: 下载器 JSONC 配置路径；``None`` 时使用
            ``configs/qmt_downloader/incremental.example.json``。配置不存在或
            解析失败时静默跳过这一级，不抛异常。

    返回：
        绝对或相对的根目录路径；本函数不校验目录是否存在。
    """
    if explicit is not None and str(explicit).strip():
        return Path(explicit)
    if os.getenv(QMT_OUTPUT_ROOT_ENV):
        return default_qmt_output_root()
    configured = _read_configured_output_root(config_path)
    if configured is not None:
        return configured
    return default_qmt_output_root()


def _read_configured_output_root(config_path: Path | str | None) -> Path | None:
    """从下载器 JSONC 配置中读取 ``output_root``。

    参数：
        config_path: 配置文件路径；``None`` 时使用下载器的增量样例配置。

    返回：
        配置中的根目录；文件缺失、解析失败或没有该键时返回 ``None``。
    """
    path = Path(config_path) if config_path is not None else default_qmt_config_path(
        "incremental.example.json"
    )
    try:
        text = path.read_text(encoding="utf-8-sig")
        values = json.loads(strip_jsonc(text))
    except (OSError, ValueError):
        return None
    if not isinstance(values, dict):
        return None
    configured = values.get("output_root")
    if not configured or not str(configured).strip():
        return None
    return Path(str(configured))


def scan_partitions(source_root: Path | str, dataset: str) -> dict[str, SourcePartition]:
    """枚举一个数据集下的全部分区目录，只做 ``scandir`` 与 ``stat``。

    这是轻量增量检查的 Tier 0 与 Tier 1：既不打开 ``data.csv``，也不解析
    ``_SUCCESS.json``。实测源目录里每个 ``_SUCCESS.json`` 约 106 KB
    （内嵌了整个证券池），逐个解析上万个标记会读上千兆字节，绝不能放在每次启动的
    路径上。

    参数：
        source_root: 大 QMT 落盘根目录。
        dataset: 数据集名，必须是 ``DATASET_BY_NAME`` 中的键。

    返回：
        以分区目录名为键的 ``SourcePartition`` 字典；数据集目录不存在时返回空字典。
        此时返回值中的摘要字段都是占位值，需要 Tier 2 才会填充。
    """
    layout = DATASET_BY_NAME[dataset]
    directory = Path(source_root) / dataset
    prefix = layout.partition_name + "="
    partitions: dict[str, SourcePartition] = {}
    try:
        entries = list(os.scandir(directory))
    except (OSError, NotADirectoryError):
        return partitions
    for entry in entries:
        if not entry.name.startswith(prefix) or not entry.is_dir():
            continue
        partition_directory = Path(entry.path)
        marker_mtime, marker_size = _stat_or_missing(partition_directory / "_SUCCESS.json")
        data_mtime, data_size = _stat_or_missing(partition_directory / "data.csv")
        partitions[entry.name] = SourcePartition(
            dataset=dataset,
            partition_key=entry.name,
            partition_value=entry.name[len(prefix):],
            directory=partition_directory,
            marker_mtime_ns=marker_mtime,
            marker_size=marker_size,
            data_mtime_ns=data_mtime,
            data_size=data_size,
        )
    return partitions


def _stat_or_missing(path: Path) -> tuple[int, int]:
    """读取单个文件的修改时间与大小，缺失时返回哨兵值。

    参数：
        path: 待检查的文件路径。

    返回：
        ``(st_mtime_ns, st_size)``；文件不存在或不可访问时返回 ``(-1, -1)``。
    """
    try:
        status = path.stat()
    except OSError:
        return -1, -1
    return int(status.st_mtime_ns), int(status.st_size)


def read_marker(partition: SourcePartition) -> SourcePartition | None:
    """解析一个分区的 ``_SUCCESS.json``，只取校验需要的少数字段。

    刻意不保留 ``partition_scope``：它内嵌了整个证券池，单个标记就有上百 KB，
    留在内存里会在批量扫描时迅速堆积。

    参数：
        partition: 已经完成 ``stat`` 的分区描述。

    返回：
        补齐了摘要、行数和完成时间的新 ``SourcePartition``；标记缺失、无法解析
        或缺少 ``sha256`` 字段时返回 ``None``。
    """
    try:
        with partition.marker_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict):
        return None
    digest = metadata.get("sha256")
    if not isinstance(digest, str) or not digest:
        return None
    try:
        rows = int(metadata.get("rows", -1))
    except (TypeError, ValueError):
        rows = -1
    identity = metadata.get("identity_sha256")
    return SourcePartition(
        dataset=partition.dataset,
        partition_key=partition.partition_key,
        partition_value=partition.partition_value,
        directory=partition.directory,
        marker_mtime_ns=partition.marker_mtime_ns,
        marker_size=partition.marker_size,
        data_mtime_ns=partition.data_mtime_ns,
        data_size=partition.data_size,
        content_sha256=digest,
        identity_sha256=identity if isinstance(identity, str) else "",
        source_rows=rows,
        completed_at=str(metadata.get("completed_at", "")),
    )


def latest_run_complete_date(source_root: Path | str) -> str:
    """返回下载器整日水位中最新的一个八位日期。

    仅用于轻量检查的水位短路，不校验水位引用的分区是否仍然完整——那是下载器
    自己在决定增量起点时才需要做的事。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        最新的八位日期字符串；没有任何水位时返回空串。
    """
    directory = Path(source_root) / "run_complete"
    latest = ""
    try:
        entries = list(os.scandir(directory))
    except (OSError, NotADirectoryError):
        return latest
    for entry in entries:
        if not entry.name.startswith("date=") or not entry.is_dir():
            continue
        value = entry.name[len("date="):]
        if len(value) == 8 and value.isdigit() and value > latest:
            latest = value
    return latest


def dataset_directory_fingerprint(source_root: Path | str) -> str:
    """汇总四个数据集目录自身的修改时间，用作水位短路的第二重判据。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        形如 ``kline_1d:123456;...`` 的稳定文本；目录缺失的数据集记为 ``-1``。
        注意 NTFS 上目录的修改时间只在增删条目时变化，孙文件被原地重写不会冒泡，
        因此该指纹只能用于 ``auto`` 模式的快速跳过。
    """
    parts = []
    root = Path(source_root)
    for layout in DATASET_LAYOUTS:
        mtime, _ = _stat_or_missing(root / layout.name)
        parts.append(f"{layout.name}:{mtime}")
    return ";".join(parts)
