"""``ingest_state`` 表的读写：记录每个源分区的文件指纹与入库结果。

这张表是轻量增量检查的全部依据。检查时先把它整表读进内存（约一万行，几十毫秒），
再与磁盘上的 ``scandir``/``stat`` 结果比对，从而在没有增量时一个 CSV 都不打开。

写入一律走「注册 ``DataFrame`` + 一条集合语句」：DuckDB 是列式引擎，对带主键的
表逐行 ``executemany`` 会退化到每行一次索引维护，实测写一万三千行要十分钟以上，
而同样的数据用一条 ``INSERT ... SELECT`` 只要毫秒级。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .source import SourcePartition

#: ``ingest_state`` 的列顺序，注册 ``DataFrame`` 时必须与之一致。
_STATE_COLUMNS = (
    "dataset",
    "partition_key",
    "marker_mtime_ns",
    "marker_size",
    "data_mtime_ns",
    "data_size",
    "content_sha256",
    "identity_sha256",
    "source_rows",
    "completed_at",
    "target_shard",
    "ingested_at",
)


@dataclass(frozen=True, slots=True)
class PartitionFingerprint:
    """``ingest_state`` 中的一行，即某个分区上次入库时的状态。

    参数：
        dataset: 数据集名。
        partition_key: 分区目录名，例如 ``date=20240102``。
        marker_mtime_ns: 上次入库时 ``_SUCCESS.json`` 的修改时间纳秒数。
        marker_size: 上次入库时 ``_SUCCESS.json`` 的字节数。
        data_mtime_ns: 上次入库时 ``data.csv`` 的修改时间纳秒数。
        data_size: 上次入库时 ``data.csv`` 的字节数。
        content_sha256: 上次入库时完成标记记录的数据文件摘要。
        source_rows: 上次入库时完成标记记录的源行数。
        target_shard: 该分区写入的月度分片，形如 ``year=2024/month=01``；
            非日期分区为空串。
    """

    dataset: str
    partition_key: str
    marker_mtime_ns: int
    marker_size: int
    data_mtime_ns: int
    data_size: int
    content_sha256: str
    source_rows: int
    target_shard: str

    def stat_matches(self, partition: SourcePartition) -> bool:
        """判断磁盘上的分区与本记录的文件指纹是否完全一致。

        参数：
            partition: 刚刚完成 ``stat`` 的源分区描述。

        返回：
            标记与数据文件的修改时间和大小四项全部相同时返回 ``True``。
        """
        return (
            self.marker_mtime_ns == partition.marker_mtime_ns
            and self.marker_size == partition.marker_size
            and self.data_mtime_ns == partition.data_mtime_ns
            and self.data_size == partition.data_size
        )


def load_fingerprints(connection) -> dict[tuple[str, str], PartitionFingerprint]:
    """整表读出 ``ingest_state``。

    参数：
        connection: 已打开的 DuckDB 连接，只读或读写均可。

    返回：
        以 ``(dataset, partition_key)`` 为键的指纹字典；表为空时返回空字典。
    """
    rows = connection.execute(
        """
        SELECT dataset, partition_key, marker_mtime_ns, marker_size,
               data_mtime_ns, data_size, content_sha256, source_rows, target_shard
        FROM ingest_state
        """
    ).fetchall()
    result: dict[tuple[str, str], PartitionFingerprint] = {}
    for row in rows:
        fingerprint = PartitionFingerprint(
            dataset=str(row[0]),
            partition_key=str(row[1]),
            marker_mtime_ns=int(row[2]),
            marker_size=int(row[3]),
            data_mtime_ns=int(row[4]),
            data_size=int(row[5]),
            content_sha256=str(row[6] or ""),
            source_rows=int(row[7]) if row[7] is not None else -1,
            target_shard=str(row[8] or ""),
        )
        result[(fingerprint.dataset, fingerprint.partition_key)] = fingerprint
    return result


def upsert_fingerprints(connection, partitions, ingested_at: datetime) -> int:
    """把一批分区的最新指纹写回 ``ingest_state``。

    参数：
        connection: 以读写方式打开的 DuckDB 连接，调用方负责事务边界。
        partitions: 待写入的 ``SourcePartition`` 可迭代对象；每一项都应当已经
            通过 ``read_marker`` 补齐摘要，未补齐时摘要按空串写入。
        ingested_at: 本次入库时间，同一批使用同一个时间戳便于审计。

    返回：
        实际写入的行数。
    """
    records = []
    for partition in partitions:
        month = partition.month()
        shard = "" if month is None else f"year={month[0]:04d}/month={month[1]:02d}"
        records.append(
            {
                "dataset": partition.dataset,
                "partition_key": partition.partition_key,
                "marker_mtime_ns": partition.marker_mtime_ns,
                "marker_size": partition.marker_size,
                "data_mtime_ns": partition.data_mtime_ns,
                "data_size": partition.data_size,
                "content_sha256": partition.content_sha256,
                "identity_sha256": partition.identity_sha256,
                "source_rows": partition.source_rows,
                "completed_at": partition.completed_at,
                "target_shard": shard,
                "ingested_at": ingested_at,
            }
        )
    if not records:
        return 0
    frame = pd.DataFrame.from_records(records, columns=list(_STATE_COLUMNS))
    connection.register("fingerprint_batch", frame)
    try:
        # 先按主键删除再整批插入，等价于 upsert，但只扫一遍索引。
        connection.execute(
            """
            DELETE FROM ingest_state s
            WHERE EXISTS (
                SELECT 1 FROM fingerprint_batch b
                WHERE b.dataset = s.dataset AND b.partition_key = s.partition_key
            )
            """
        )
        connection.execute(
            "INSERT INTO ingest_state({0}) SELECT {0} FROM fingerprint_batch".format(
                ", ".join(_STATE_COLUMNS)
            )
        )
    finally:
        connection.unregister("fingerprint_batch")
    return len(records)


def delete_fingerprints(connection, keys) -> int:
    """删除源侧已经消失的分区记录。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        keys: ``(dataset, partition_key)`` 二元组的可迭代对象。

    返回：
        传入的待删除键数量；其中不存在的键不会报错。
    """
    records = [
        {"dataset": dataset, "partition_key": partition_key}
        for dataset, partition_key in keys
    ]
    if not records:
        return 0
    frame = pd.DataFrame.from_records(records, columns=["dataset", "partition_key"])
    connection.register("fingerprint_removed", frame)
    try:
        connection.execute(
            """
            DELETE FROM ingest_state s
            WHERE EXISTS (
                SELECT 1 FROM fingerprint_removed r
                WHERE r.dataset = s.dataset AND r.partition_key = s.partition_key
            )
            """
        )
    finally:
        connection.unregister("fingerprint_removed")
    return len(records)


def read_metadata(connection) -> dict[str, str]:
    """读出 ``dataset_metadata`` 全部键值。

    参数：
        connection: 已打开的 DuckDB 连接。

    返回：
        键到值的字符串字典；表为空时返回空字典。
    """
    rows = connection.execute("SELECT key, value FROM dataset_metadata").fetchall()
    return {str(key): str(value) for key, value in rows}


def write_metadata(connection, values) -> None:
    """更新 ``dataset_metadata`` 中的若干键。

    参数：
        connection: 以读写方式打开的 DuckDB 连接。
        values: 键到值的映射；值一律转成字符串保存，``None`` 会被跳过。

    返回：
        无返回值。
    """
    records = [
        {"key": str(key), "value": str(value)}
        for key, value in values.items()
        if value is not None
    ]
    if not records:
        return
    frame = pd.DataFrame.from_records(records, columns=["key", "value"])
    connection.register("metadata_batch", frame)
    try:
        connection.execute(
            "DELETE FROM dataset_metadata t "
            "WHERE EXISTS (SELECT 1 FROM metadata_batch b WHERE b.key = t.key)"
        )
        connection.execute(
            "INSERT INTO dataset_metadata(key, value) SELECT key, value FROM metadata_batch"
        )
    finally:
        connection.unregister("metadata_batch")
