# -*- coding: utf-8 -*-
"""提供按天分区、原子写入、完成标记和缺失报告能力。"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .dates import normalize_date


class IssueCollector(object):
    """集中收集不会立即终止任务的数据质量问题。"""

    def __init__(self):
        """初始化空问题列表。

        返回：
            无返回值；后续问题通过 ``add`` 追加。
        """
        self.items = []

    def add(self, level, dataset, code, date_value, message):
        """追加一条结构化问题记录。

        参数：
            level: 严重级别，可使用 ``WARNING`` 或 ``ERROR``。
            dataset: 发生问题的数据集名称。
            code: 相关证券代码；无法定位到单只证券时传空字符串。
            date_value: 相关交易日、公告日或报告期；未知时传空字符串。
            message: 供用户排查的中文问题描述。

        返回：
            无返回值。
        """
        self.items.append(
            {
                "level": str(level),
                "dataset": str(dataset),
                "code": str(code or ""),
                "date": str(date_value or ""),
                "message": str(message),
            }
        )

    def extend(self, rows):
        """合并另一批问题记录。

        参数：
            rows: 由问题字典组成的可迭代对象，键应与 ``add`` 生成结果一致。

        返回：
            无返回值。
        """
        self.items.extend(list(rows))


class DailyPartitionStore(object):
    """管理业务数据日分区和可恢复的 staging 批次文件。"""

    def __init__(self, root):
        """初始化输出根目录。

        参数：
            root: 下载结果根目录；其下会创建业务数据、staging、日志和报告目录。
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(parents=True, exist_ok=True)

    def is_partition_complete(self, dataset_path, partition_name, partition_value):
        """判断一个日分区是否已经具有完成标记。

        参数：
            dataset_path: 相对输出根目录的数据集路径，例如 ``kline_1d``。
            partition_name: 分区字段名，例如 ``date`` 或 ``announce_date``。
            partition_value: 八位日期或 ``unknown`` 等分区值。

        返回：
            完成标记、数据文件、记录数和 SHA-256 全部一致时返回 ``True``。
        """
        directory = self.partition_directory(dataset_path, partition_name, partition_value)
        return _validate_partition_directory(directory)

    def partition_directory(self, dataset_path, partition_name, partition_value):
        """生成经过安全校验的分区目录。

        参数：
            dataset_path: 相对输出根目录的数据集路径，允许包含 ``table=...`` 子目录。
            partition_name: 分区字段名，只允许字母、数字和下划线。
            partition_value: 分区值，只允许文件名安全字符。

        返回：
            位于输出根目录内的 ``Path`` 对象。
        """
        relative = Path(str(dataset_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("dataset_path 必须是输出根目录内的相对路径")
        name = _safe_component(partition_name)
        value = _safe_component(partition_value)
        return self.root / relative / "{0}={1}".format(name, value)

    def partition_matches_scope(
        self, dataset_path, partition_name, partition_value, partition_scope
    ):
        """判断完成分区是否属于当前抽取范围。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名。
            partition_value: 当前分区日期或 ``unknown``。
            partition_scope: 包含证券池、财务回看和字段版本的抽取范围字典。

        返回：
            分区完整且完成元数据中的范围完全一致时返回 ``True``。
        """
        if not self.is_partition_complete(dataset_path, partition_name, partition_value):
            return False
        directory = self.partition_directory(dataset_path, partition_name, partition_value)
        try:
            with (directory / "_SUCCESS.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            return metadata.get("partition_scope") == partition_scope
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def write_partition(
        self,
        dataset_path,
        partition_name,
        partition_value,
        frame,
        expected_columns,
        identity_columns,
        sort_columns,
        metadata,
        overwrite=False,
    ):
        """以临时文件加原子替换方式写入一个完整日分区。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名，例如 ``date``。
            partition_value: 当前分区日期或其他安全值。
            frame: 待写入的 ``DataFrame``；空表同样会生成可读表头。
            expected_columns: 空表或缺列时必须保留的规范列顺序。
            identity_columns: 用于检测重复业务记录的联合主键列。
            sort_columns: 为保证文件稳定性而使用的排序列。
            metadata: 写入完成标记的额外元数据字典。
            overwrite: 是否无条件替换已有完成分区，修复模式通常为 ``True``；为
                ``False`` 时，范围不同的完成分区会报错，范围一致的完成分区按业务
                主键集合决定跳过或原地重写。

        返回：
            字典，包含 ``status``、``rows`` 和最终文件路径。
        """
        directory = self.partition_directory(dataset_path, partition_name, partition_value)
        success_path = directory / "_SUCCESS.json"
        data_path = directory / "data.csv"
        if self.is_partition_complete(dataset_path, partition_name, partition_value) and not overwrite:
            with success_path.open("r", encoding="utf-8") as handle:
                existing_metadata = json.load(handle)
            requested_scope = (metadata or {}).get("partition_scope")
            if existing_metadata.get("partition_scope") != requested_scope:
                raise ValueError(
                    "分区已由不同证券池或数据范围写入，请使用 repair 或独立输出目录: {0}".format(
                        directory
                    )
                )
            # 范围一致且业务主键集合未变化时跳过；主键集合变化（例如缺口补齐后新增
            # 记录）时在同一范围内重写。范围守卫始终先于重写判断，避免跨证券池覆盖。
            if self.partition_identity_matches(
                dataset_path, partition_name, partition_value, frame, identity_columns
            ):
                return {"status": "skipped", "rows": None, "path": str(data_path)}

        output = frame.copy() if frame is not None else pd.DataFrame()
        for column in expected_columns:
            if column not in output.columns:
                output[column] = pd.Series(index=output.index, dtype="object")
        extra_columns = [column for column in output.columns if column not in expected_columns]
        output = output[list(expected_columns) + sorted(extra_columns)]
        present_identity = [column for column in identity_columns if column in output.columns]
        if present_identity and output.duplicated(present_identity).any():
            raise ValueError(
                "分区 {0} 存在重复业务主键: {1}".format(
                    partition_value, ",".join(present_identity)
                )
            )
        present_sort = [column for column in sort_columns if column in output.columns]
        if present_sort and not output.empty:
            output = output.sort_values(present_sort, kind="mergesort").reset_index(drop=True)

        directory.mkdir(parents=True, exist_ok=True)
        data_temp = directory / "data.csv.tmp"
        success_temp = directory / "_SUCCESS.json.tmp"
        if success_path.exists():
            success_path.unlink()
        digest = _write_csv_atomic(output, data_temp, data_path)

        success = dict(metadata or {})
        success.update(
            {
                "dataset": str(dataset_path).replace("\\", "/"),
                "partition": "{0}={1}".format(partition_name, partition_value),
                "rows": int(len(output)),
                "identity_sha256": _identity_digest(output, identity_columns),
                "sha256": digest,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        with success_temp.open("w", encoding="utf-8") as handle:
            json.dump(success, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(str(success_temp), str(success_path))
        return {"status": "written", "rows": int(len(output)), "path": str(data_path)}

    def partition_identity_matches(
        self,
        dataset_path,
        partition_name,
        partition_value,
        frame,
        identity_columns,
    ):
        """比较已有分区与待写表的业务主键集合是否一致。

        优先只读 ``_SUCCESS.json``：行数不同直接判定不一致；行数一致且完成标记
        带有 ``identity_sha256`` 时用主键摘要比较，无需解析分区 CSV。只有旧版本
        写出的、缺少该字段的分区才回落到读取并比较完整主键集合。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名，例如 ``date``。
            partition_value: 当前分区日期或其他安全值。
            frame: 本次准备写入的标准数据表。
            identity_columns: 用于比较记录覆盖范围的联合业务主键列。

        返回：
            完成元数据、行数和联合业务主键集合完全一致时返回 ``True``。
        """
        directory = self.partition_directory(
            dataset_path, partition_name, partition_value
        )
        success_path = directory / "_SUCCESS.json"
        try:
            with success_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        expected = frame if frame is not None else pd.DataFrame()
        if any(column not in expected.columns for column in identity_columns):
            return False
        try:
            existing_rows = int(metadata.get("rows", -1))
        except (TypeError, ValueError):
            return False
        if existing_rows != len(expected):
            return False
        existing_digest = metadata.get("identity_sha256")
        if existing_digest is not None:
            return existing_digest == _identity_digest(expected, identity_columns)
        existing = self.read_partition(
            dataset_path,
            partition_name,
            partition_value,
            dtype=dict.fromkeys(identity_columns, str),
        )
        if existing is None or len(existing) != len(expected):
            return False
        if any(column not in existing.columns for column in identity_columns):
            return False
        expected_keys = set(
            tuple(str(value) for value in row)
            for row in expected[list(identity_columns)].itertuples(
                index=False, name=None
            )
        )
        existing_keys = set(
            tuple(str(value) for value in row)
            for row in existing[list(identity_columns)].itertuples(
                index=False, name=None
            )
        )
        return len(expected_keys) == len(expected) and expected_keys == existing_keys

    def read_partition(self, dataset_path, partition_name, partition_value, dtype=None):
        """读取一个业务分区的 ``data.csv``。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名，例如 ``date`` 或 ``snapshot``。
            partition_value: 当前分区日期或其他安全值。
            dtype: 传递给 ``pd.read_csv`` 的列类型映射；缺省不强制类型。

        返回：
            分区数据表；文件不存在或无法解析时返回 ``None``。
        """
        data_path = (
            self.partition_directory(dataset_path, partition_name, partition_value)
            / "data.csv"
        )
        if not data_path.is_file():
            return None
        try:
            return pd.read_csv(str(data_path), encoding="utf-8-sig", dtype=dtype)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            return None

    def write_run_date_complete(self, trade_date, required_partitions, metadata):
        """在所有必需业务分区校验通过后写整日完成水位。

        参数：
            trade_date: 当前实际交易日的八位日期。
            required_partitions: 相对输出根目录的业务分区目录序列。
            metadata: 任务键、模式等审计元数据。

        返回：
            最终整日完成标记路径；任一业务分区不完整时抛出 ``ValueError``。
        """
        relative_values = []
        for value in required_partitions:
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("整日完成标记只能引用输出根目录内的分区")
            directory = self.root / relative
            if not _validate_partition_directory(directory):
                raise ValueError("业务分区未通过完整性校验: {0}".format(relative))
            relative_values.append(str(relative).replace("\\", "/"))
        completion_dir = self.root / "run_complete" / "date={0}".format(
            _safe_component(trade_date)
        )
        completion_dir.mkdir(parents=True, exist_ok=True)
        final_path = completion_dir / "_SUCCESS.json"
        temporary = completion_dir / "_SUCCESS.json.tmp"
        if final_path.exists():
            final_path.unlink()
        payload = dict(metadata or {})
        payload.update(
            {
                "trade_date": str(trade_date),
                "required_partitions": relative_values,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(str(temporary), str(final_path))
        return final_path

    def fragment_exists(self, job_key, dataset, batch_id):
        """判断断点 staging 批次文件是否存在。

        参数：
            job_key: 稳定任务标识。
            dataset: staging 数据集名，可包含财务表后缀。
            batch_id: 零基批次编号。

        返回：
            对应 CSV 文件存在时返回 ``True``。
        """
        return _validate_fragment(self._fragment_path(job_key, dataset, batch_id))

    def fragment_total_rows(self, job_key, dataset, batch_ids):
        """从 staging 元数据累加各批次记录的行数，不读取 CSV 正文。

        只用于回答“这个数据集有没有数据”这类问题。跨数千个交易日逐个读取 CSV
        只为判断是否为空，代价远高于读取同名 ``.meta.json``；数据完整性仍由随后
        真正读取分片时的 SHA-256 与行数校验保证。

        参数：
            job_key: 稳定任务标识。
            dataset: staging 数据集名。
            batch_ids: 需要累加的批次编号可迭代对象。

        返回：
            元数据中记录的总行数；元数据缺失或损坏的批次按 0 计入。
        """
        total = 0
        for batch_id in batch_ids:
            metadata_path = self._fragment_path(job_key, dataset, batch_id).with_suffix(
                ".meta.json"
            )
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                total += max(int(metadata.get("rows", 0)), 0)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return total

    def write_fragment(self, job_key, dataset, batch_id, frame):
        """原子写入一个可复用的批次中间文件。

        参数：
            job_key: 由关键配置生成的稳定任务标识。
            dataset: staging 数据集名。
            batch_id: 零基批次编号。
            frame: 当前批次的 ``DataFrame``，允许为空。

        返回：
            最终 staging CSV 文件路径。
        """
        path = self._fragment_path(job_key, dataset, batch_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        metadata_path = path.with_suffix(".meta.json")
        metadata_temporary = path.with_suffix(".meta.json.tmp")
        digest = _write_csv_atomic(frame, temporary, path)
        with metadata_temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                {"rows": int(len(frame)), "sha256": digest},
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        os.replace(str(metadata_temporary), str(metadata_path))
        return path

    def read_fragments(self, job_key, dataset, batch_ids):
        """读取已完成批次并合并为一个数据表。

        参数：
            job_key: 稳定任务标识。
            dataset: staging 数据集名。
            batch_ids: 需要合并的批次编号可迭代对象。

        返回：
            按批次顺序拼接后的 ``DataFrame``；无数据时返回空表。
        """
        frames = []
        for batch_id in batch_ids:
            path = self._fragment_path(job_key, dataset, batch_id)
            if not path.is_file():
                continue
            if not _validate_fragment(path):
                raise ValueError("staging 批次校验失败，将在下次运行重新下载: {0}".format(path))
            try:
                frame = pd.read_csv(
                    str(path),
                    encoding="utf-8-sig",
                    dtype={
                        "code": str,
                        "trade_date": str,
                        "report_date": str,
                        "announce_date": str,
                        "ex_date": str,
                    },
                )
                frames.append(frame)
            except pd.errors.EmptyDataError:
                frames.append(pd.DataFrame())
        non_empty = [frame for frame in frames if len(frame.columns) > 0]
        if not non_empty:
            return pd.DataFrame()
        # 大 QMT 内置的旧版 pandas.concat 不支持 sort 参数。
        merged = pd.concat(non_empty, ignore_index=True)
        # 合并后整列规范化一次，而不是对每个分片各做一次：向量化的固定开销在
        # 几百行的分片上摊不开，按交易日合并全部批次后才划算。
        for column in ("trade_date", "report_date", "announce_date", "ex_date"):
            if column in merged.columns:
                merged[column] = _normalize_date_column(merged[column])
        return merged

    def write_issue_report(self, rows, run_id):
        """将缺失与异常提示写入外部 CSV 报告。

        参数：
            rows: 结构化问题字典列表。
            run_id: 本次运行标识，用于避免覆盖其他任务报告。

        返回：
            报告文件路径；即使没有问题也会写出仅含表头的报告。
        """
        report_path = self.root / "reports" / "issues_{0}.csv".format(
            _safe_component(run_id)
        )
        temporary = report_path.with_suffix(".csv.tmp")
        columns = ["level", "dataset", "code", "date", "message"]
        pd.DataFrame(list(rows), columns=columns).to_csv(
            str(temporary), index=False, encoding="utf-8-sig"
        )
        os.replace(str(temporary), str(report_path))
        return report_path

    def write_line_correction_report(self, rows, run_id):
        """将行级价格修正记录写入 ``reports/line_correct`` 下的 CSV 报告。

        参数：
            rows: 结构化修正记录字典列表。
            run_id: 本次运行标识，用于避免覆盖其他任务报告。

        返回：
            报告文件路径；即使没有修正也会写出仅含表头的报告。
        """
        report_dir = self.root / "reports" / "line_correct"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "corrections_{0}.csv".format(
            _safe_component(run_id)
        )
        temporary = report_path.with_suffix(".csv.tmp")
        columns = [
            "dataset",
            "code",
            "trade_date",
            "field",
            "original_value",
            "corrected_value",
            "message",
        ]
        pd.DataFrame(list(rows), columns=columns).to_csv(
            str(temporary), index=False, encoding="utf-8-sig"
        )
        os.replace(str(temporary), str(report_path))
        return report_path

    def _fragment_path(self, job_key, dataset, batch_id):
        """生成 staging 批次文件路径。

        参数：
            job_key: 稳定任务标识，只保留文件名安全字符。
            dataset: 数据集名，只保留文件名安全字符。
            batch_id: 零基批次编号。

        返回：
            当前批次对应的 CSV ``Path``。
        """
        return (
            self.root
            / "staging"
            / _safe_component(job_key)
            / _safe_component(dataset)
            / "batch_{0:05d}.csv".format(int(batch_id))
        )


def _safe_component(value):
    """校验单个目录或文件名片段。

    参数：
        value: 待用于路径的字符串值。

    返回：
        通过校验的字符串；包含危险字符时抛出 ``ValueError``。
    """
    text = str(value)
    if not text or not re.match(r"^[A-Za-z0-9_.=-]+$", text):
        raise ValueError("不安全的路径片段: {0}".format(text))
    return text


def _identity_digest(frame, identity_columns):
    """计算分区业务主键集合的稳定摘要。

    参数：
        frame: 已按规范列整理的分区数据表。
        identity_columns: 联合业务主键列。

    返回：
        主键集合的 SHA-256 十六进制摘要；主键列缺失或存在重复主键时返回
        ``None``，调用方据此回落到逐行比较。
    """
    if frame is None or any(column not in frame.columns for column in identity_columns):
        return None
    keys = set()
    for row in frame[list(identity_columns)].itertuples(index=False, name=None):
        keys.add("\x1f".join(str(value) for value in row))
    if len(keys) != len(frame):
        return None
    digest = hashlib.sha256()
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _normalize_date_column(values):
    """规范化 staging 日期列，整列已是标准八位日期时跳过逐行转换。

    分片里的日期几乎都是本程序自己写出的 ``YYYYMMDD``，逐行调用
    ``normalize_date`` 只是把同样的值重算一遍。按交易日读回全量 staging 时这项
    开销会累计到分钟级，因此先用一次向量化匹配确认整列已合规再决定是否跳过。
    存在空值时一律走逐行转换，保持空值统一变成 ``None`` 的既有行为。

    参数：
        values: 从 staging CSV 读出的日期列 ``Series``。

    返回：
        与逐行 ``normalize_date`` 完全一致的 ``Series``。
    """
    if bool(values.notna().all()):
        text = values.astype(str)
        # 旧版 pandas 没有 str.fullmatch，用锚定的 str.match 达到同样效果。
        if bool(text.str.match(r"^(19|20)\d{6}$").all()):
            return text
    return values.apply(normalize_date)


def _write_csv_atomic(frame, temporary, final_path):
    """原子写入 UTF-8 BOM 编码的 CSV，并返回落盘内容的 SHA-256。

    摘要直接对写入的字节计算，无需再把文件读回一遍，也保证摘要与文件内容必然
    一致。

    参数：
        frame: 待写入的 ``DataFrame``；空表同样会写出表头。
        temporary: 同目录下的临时文件路径。
        final_path: 替换完成后的最终文件路径。

    返回：
        六十四位小写十六进制摘要字符串。
    """
    payload = frame.to_csv(index=False).encode("utf-8-sig")
    with Path(temporary).open("wb") as handle:
        handle.write(payload)
    os.replace(str(temporary), str(final_path))
    return hashlib.sha256(payload).hexdigest()


def _digest_and_row_count(path):
    """一次读取同时得到文件 SHA-256 和 CSV 数据行数。

    行数按 ``\\r``、``\\n`` 和 ``\\r\\n`` 三种换行切分后减去表头，与以
    ``newline=""`` 打开文本逐行计数的结果一致。

    参数：
        path: 待校验的 CSV 文件路径。

    返回：
        ``(sha256, row_count)`` 二元组。
    """
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest(), max(len(payload.splitlines()) - 1, 0)


def validate_partition_directory(directory):
    """核验业务分区的数据文件和完成元数据。

    这是本模块对外公开的分区完整性校验入口；``quant.market_data`` 的入库流程
    直接复用它，避免在两处各写一份 SHA-256 与行数比对逻辑。

    参数：
        directory: 同时应包含 ``data.csv`` 和 ``_SUCCESS.json`` 的分区目录。

    返回：
        文件存在、CSV 行数和 SHA-256 均与完成标记一致时返回 ``True``。
    """
    directory = Path(directory)
    data_path = directory / "data.csv"
    success_path = directory / "_SUCCESS.json"
    if not data_path.is_file() or not success_path.is_file():
        return False
    try:
        with success_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        digest, row_count = _digest_and_row_count(data_path)
        if metadata.get("sha256") != digest:
            return False
        return int(metadata.get("rows", -1)) == row_count
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


# 历史内部调用方仍使用下划线名字；保留别名以免改动散落的调用点。
_validate_partition_directory = validate_partition_directory


def _validate_fragment(path):
    """核验 staging CSV 对应的行数和 SHA-256 元数据。

    参数：
        path: staging CSV 文件路径；元数据位于同名 ``.meta.json``。

    返回：
        CSV 和元数据同时存在且行数、摘要一致时返回 ``True``。
    """
    path = Path(path)
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        digest, row_count = _digest_and_row_count(path)
        if metadata.get("sha256") != digest:
            return False
        return int(metadata.get("rows", -1)) == row_count
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
