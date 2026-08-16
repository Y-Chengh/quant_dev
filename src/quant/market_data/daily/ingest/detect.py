"""轻量增量检查：判断 QMT 落盘目录相对日线库有没有新数据。

分四级递进，核心约束是**除非文件字节变了，否则一个 CSV 都不解析**：

- Tier −1 水位短路：只看 ``run_complete`` 最新日期和四个数据集目录的修改时间。
- Tier 0 每数据集一次 ``scandir``，与 ``ingest_state`` 的键做差。
- Tier 1 每分区两次 ``stat``，与库里记录的四项文件指纹比对。
- Tier 2 只对指纹变化的分区解析 ``_SUCCESS.json``，比其中的 ``sha256``。
- Tier 3 可选，重算 ``data.csv`` 的 SHA-256，复用下载器的校验实现。

实测源目录有 6450 个日分区、每个 ``_SUCCESS.json`` 约 106 KB，全量解析等于读
1.3 GB JSON，因此 Tier 2 必须严格限定在指纹已变化的分区上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from quant.qmt_downloader.storage import validate_partition_directory

from .source import (
    DATASET_LAYOUTS,
    SourcePartition,
    dataset_directory_fingerprint,
    latest_run_complete_date,
    read_marker,
    scan_partitions,
)
from .state import PartitionFingerprint

#: ``dataset_metadata`` 中保存水位短路判据的键。
WATERMARK_DATE_KEY = "source_watermark_date"
WATERMARK_FINGERPRINT_KEY = "source_directory_fingerprint"


@dataclass(frozen=True, slots=True)
class SourceDelta:
    """一次增量检查的结论。

    参数：
        dirty: 内容确实变化、需要重新入库的分区。
        refreshed: 只有文件时间戳变了、内容摘要未变的分区；只需刷新指纹记录。
        removed: 源侧已消失、需要从库里清掉的 ``(dataset, partition_key)``。
        pending: 无法判定的分区，元素为 ``(dataset, partition_key, 原因)``；
            这些分区既不入库也不记录指纹，由调用方报告为问题。
        scanned: 本次实际枚举到的分区总数，用于日志与审计。
        short_circuited: 是否在 Tier −1 水位短路处直接返回。
    """

    dirty: tuple[SourcePartition, ...] = ()
    refreshed: tuple[SourcePartition, ...] = ()
    removed: tuple[tuple[str, str], ...] = ()
    pending: tuple[tuple[str, str, str], ...] = ()
    scanned: int = 0
    short_circuited: bool = False

    def is_empty(self) -> bool:
        """判断是否完全没有需要落库的变化。

        返回：
            没有脏分区、没有需要刷新的指纹、也没有删除时返回 ``True``；
            ``pending`` 不算变化，但调用方仍应把它报告出来。
        """
        return not self.dirty and not self.refreshed and not self.removed

    def touched_months(self) -> frozenset[tuple[int, int]]:
        """返回需要重写 Parquet 的月份集合。

        只有日线数据集会产生月度分片；除权数据落在 DuckDB 表里，不参与分片重写。

        返回：
            ``(年, 月)`` 二元组的不可变集合。
        """
        months: set[tuple[int, int]] = set()
        for partition in self.dirty:
            if partition.dataset != "kline_1d":
                continue
            month = partition.month()
            if month is not None:
                months.add(month)
        for dataset, partition_key in self.removed:
            if dataset != "kline_1d":
                continue
            value = partition_key.split("=", 1)[-1]
            if len(value) == 8 and value.isdigit():
                months.add((int(value[:4]), int(value[4:6])))
        return frozenset(months)

    def dirty_datasets(self) -> frozenset[str]:
        """返回内容发生变化的数据集名集合。

        返回：
            出现在 ``dirty`` 或 ``removed`` 里的数据集名。
        """
        names = {partition.dataset for partition in self.dirty}
        names.update(dataset for dataset, _ in self.removed)
        return frozenset(names)

    def dirty_partition_values(self, dataset: str) -> tuple[str, ...]:
        """返回某个数据集里内容变化的分区值，按升序排列。

        参数：
            dataset: 数据集名，例如 ``corporate_actions``。

        返回：
            分区值元组，例如 ``("20240102", "20240103")``。
        """
        values = sorted(
            partition.partition_value
            for partition in self.dirty
            if partition.dataset == dataset
        )
        return tuple(values)


@dataclass
class _DetectionAccumulator:
    """检测过程中逐分区累积的中间结果。"""

    dirty: list[SourcePartition] = field(default_factory=list)
    refreshed: list[SourcePartition] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)
    pending: list[tuple[str, str, str]] = field(default_factory=list)
    scanned: int = 0


def detect_changes(
    source_root: Path | str,
    fingerprints: dict[tuple[str, str], PartitionFingerprint],
    metadata: dict[str, str],
    *,
    mode: str = "auto",
    verify_hash: bool = False,
) -> SourceDelta:
    """比对 QMT 落盘目录与日线库已入库状态，给出增量结论。

    参数：
        source_root: 大 QMT 落盘根目录。
        fingerprints: 从 ``ingest_state`` 整表读出的指纹字典。
        metadata: 从 ``dataset_metadata`` 读出的键值，用于水位短路。
        mode: ``auto`` 允许 Tier −1 水位短路；``full`` 跳过短路，从 Tier 0 开始；
            ``rebuild`` 把磁盘上的全部分区都视为脏分区，用于整库重建。
        verify_hash: 是否启用 Tier 3，对内容变化的分区重算 ``data.csv`` 摘要
            并与完成标记比对。开启后耗时会随脏分区数量线性增长。

    返回：
        描述本次增量的 ``SourceDelta``。
    """
    if mode == "auto" and _watermark_unchanged(source_root, metadata, fingerprints):
        return SourceDelta(short_circuited=True)

    accumulator = _DetectionAccumulator()
    for layout in DATASET_LAYOUTS:
        partitions = scan_partitions(source_root, layout.name)
        accumulator.scanned += len(partitions)
        known = {
            key[1]: value
            for key, value in fingerprints.items()
            if key[0] == layout.name
        }
        for partition_key in sorted(set(known) - set(partitions)):
            accumulator.removed.append((layout.name, partition_key))
        for partition_key in sorted(partitions):
            _classify_partition(
                accumulator,
                partitions[partition_key],
                known.get(partition_key),
                rebuild=mode == "rebuild",
                verify_hash=verify_hash,
            )
    return SourceDelta(
        dirty=tuple(accumulator.dirty),
        refreshed=tuple(accumulator.refreshed),
        removed=tuple(accumulator.removed),
        pending=tuple(accumulator.pending),
        scanned=accumulator.scanned,
    )


def _classify_partition(
    accumulator: _DetectionAccumulator,
    partition: SourcePartition,
    known: PartitionFingerprint | None,
    *,
    rebuild: bool,
    verify_hash: bool,
) -> None:
    """把单个分区归入脏、待刷新、待定三类之一。

    参数：
        accumulator: 累积结果的可变容器，就地追加。
        partition: 已完成 Tier 1 ``stat`` 的源分区。
        known: 库中已有的指纹记录；``None`` 表示这是一个新分区。
        rebuild: 整库重建模式；为 ``True`` 时跳过一切跳过判断。
        verify_hash: 是否对脏分区额外做 Tier 3 字节校验。

    返回：
        无返回值；结果写入 ``accumulator``。
    """
    if partition.data_mtime_ns < 0:
        accumulator.pending.append((partition.dataset, partition.partition_key, "缺少 data.csv"))
        return
    if partition.marker_mtime_ns < 0:
        accumulator.pending.append(
            (partition.dataset, partition.partition_key, "缺少 _SUCCESS.json 完成标记")
        )
        return
    if not rebuild and known is not None and known.stat_matches(partition):
        return

    detailed = read_marker(partition)
    if detailed is None:
        accumulator.pending.append(
            (partition.dataset, partition.partition_key, "_SUCCESS.json 无法解析或缺少 sha256")
        )
        return
    if verify_hash and not validate_partition_directory(partition.directory):
        accumulator.pending.append(
            (partition.dataset, partition.partition_key, "data.csv 与完成标记的 SHA-256 不一致")
        )
        return
    if not rebuild and known is not None and known.content_sha256 == detailed.content_sha256:
        # 下载器幂等重写会刷新时间戳但不改内容，这里只更新指纹，不重算分片。
        accumulator.refreshed.append(detailed)
        return
    accumulator.dirty.append(detailed)


def _watermark_unchanged(
    source_root: Path | str,
    metadata: dict[str, str],
    fingerprints: dict[tuple[str, str], PartitionFingerprint],
) -> bool:
    """判断能否用整日水位与目录时间戳直接跳过本次检查。

    只在 ``auto`` 模式下使用。NTFS 上目录的修改时间只在增删条目时变化，孙文件
    被原地重写不会冒泡，因此这一级可能漏掉「分区被原地修复」的情况；
    ``--sync-mode full`` 与 ``python -m quant.cli.market_check``
    一律不走这条路径。

    参数：
        source_root: 大 QMT 落盘根目录。
        metadata: 库中记录的元数据键值。
        fingerprints: 库中已有的分区指纹；为空说明还没入过库，必须走完整检查。

    返回：
        水位日期与目录指纹都与库中记录一致时返回 ``True``。
    """
    if not fingerprints:
        return False
    recorded_date = metadata.get(WATERMARK_DATE_KEY, "")
    recorded_fingerprint = metadata.get(WATERMARK_FINGERPRINT_KEY, "")
    if not recorded_date or not recorded_fingerprint:
        return False
    if latest_run_complete_date(source_root) != recorded_date:
        return False
    return dataset_directory_fingerprint(source_root) == recorded_fingerprint


def watermark_values(source_root: Path | str) -> dict[str, str]:
    """采集当前源目录的水位判据，供入库成功后写回元数据。

    参数：
        source_root: 大 QMT 落盘根目录。

    返回：
        含 ``source_watermark_date`` 与 ``source_directory_fingerprint`` 的字典。
    """
    return {
        WATERMARK_DATE_KEY: latest_run_complete_date(source_root),
        WATERMARK_FINGERPRINT_KEY: dataset_directory_fingerprint(source_root),
    }
