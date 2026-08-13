# -*- coding: utf-8 -*-
"""下载器 JSON 配置加载与业务校验。"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path


SUPPORTED_MODES = ("backfill", "incremental", "repair")
SUPPORTED_DATASETS = (
    "kline_1d",
    "finance_raw",
    "finance_daily",
    "corporate_actions",
)


class DownloaderConfig(object):
    """保存经过校验的大 QMT 下载任务配置。"""

    def __init__(self, values):
        """初始化下载配置并校验日期、批量大小及数据集。

        参数：
            values: JSON 配置解析后的字典；必须包含输出目录、模式、起止日期，
                股票列表可为空并改由 ``sector`` 指定大 QMT 板块。
        """
        self.output_root = Path(values.get("output_root", r"D:\qmt_data_test"))
        self.mode = str(values.get("mode", "backfill"))
        self.start_date = str(values.get("start_date", ""))
        self.end_date = str(values.get("end_date", ""))
        self.auto_start_requested = self.start_date.lower() == "auto"
        self.incremental_initial_start_date = str(
            values.get("incremental_initial_start_date", "")
        )
        self.incremental_lag_days = int(values.get("incremental_lag_days", 1))
        self.no_work = False
        self.symbols = tuple(str(item).strip() for item in values.get("symbols", []) if str(item).strip())
        self.sector = str(values.get("sector", "")).strip()
        self.batch_size = int(values.get("batch_size", 100))
        self.save_workers = int(values.get("save_workers", 1))
        self.retry_count = int(values.get("retry_count", 3))
        self.download_kline = bool(values.get("download_kline", True))
        self.overwrite_completed_partition = bool(
            values.get("overwrite_completed_partition", self.mode == "repair")
        )
        self.datasets = tuple(values.get("datasets", SUPPORTED_DATASETS))
        self.allow_partial_finance = bool(values.get("allow_partial_finance", False))
        self.calendar_symbol = str(values.get("calendar_symbol", "000001.SH")).strip().upper()
        self.watermark_scope = {
            "datasets": sorted(self.datasets),
            "symbols": sorted(str(code).strip().upper() for code in self.symbols),
            "sector": self.sector,
        }
        self.finance_lookback_start = str(
            values.get("finance_lookback_start", "20000101")
        )
        self.log_max_bytes = int(values.get("log_max_bytes", 20 * 1024 * 1024))
        self.log_backup_count = int(values.get("log_backup_count", 10))
        self._resolve_incremental_dates()
        self._validate()

    @classmethod
    def from_json(cls, path):
        """从 UTF-8 JSON 文件构造下载配置。

        参数：
            path: 配置文件路径；文件内容必须是 JSON 对象，未知键会被忽略。

        返回：
            已完成字段和业务规则校验的 ``DownloaderConfig``。
        """
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8-sig") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("下载器配置根节点必须是 JSON 对象")
        return cls(values)

    def _validate(self):
        """校验配置内部一致性并拒绝可能误写数据的参数。

        返回：
            无返回值；配置非法时抛出 ``ValueError``。
        """
        if self.mode not in SUPPORTED_MODES:
            raise ValueError("mode 必须是 backfill、incremental 或 repair")
        start = _parse_date(self.start_date, "start_date")
        end = _parse_date(self.end_date, "end_date")
        _parse_date(self.finance_lookback_start, "finance_lookback_start")
        if start > end and not self.no_work:
            raise ValueError("start_date 不能晚于 end_date")
        if not self.symbols and not self.sector:
            raise ValueError("symbols 与 sector 至少需要配置一个")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.save_workers <= 0 or self.save_workers > 16:
            raise ValueError("save_workers 必须在 1 到 16 之间")
        if self.retry_count <= 0:
            raise ValueError("retry_count 必须大于 0")
        if self.incremental_lag_days < 0:
            raise ValueError("incremental_lag_days 不能小于 0")
        invalid = sorted(set(self.datasets) - set(SUPPORTED_DATASETS))
        if invalid:
            raise ValueError("不支持的数据集: {0}".format(", ".join(invalid)))
        if not self.datasets:
            raise ValueError("datasets 不能为空")
        if self.mode == "incremental" and self.auto_start_requested:
            daily_datasets = {"kline_1d", "finance_daily", "corporate_actions"}
            if not set(self.datasets) & daily_datasets:
                raise ValueError("自动增量至少需要一个可按交易日完成的数据集")
        if not self.calendar_symbol or "." not in self.calendar_symbol:
            raise ValueError("calendar_symbol 必须包含市场后缀")

    def _resolve_incremental_dates(self):
        """把增量模式中的 ``auto`` 日期解析为无需每日修改的实际区间。

        返回：
            无返回值；会就地更新 ``start_date``、``end_date`` 和 ``no_work``。
        """
        if self.mode != "incremental":
            return
        if self.end_date.lower() == "auto":
            target = datetime.now().date() - timedelta(days=self.incremental_lag_days)
            self.end_date = target.strftime("%Y%m%d")
        if self.start_date.lower() == "auto":
            latest = _latest_completed_run_date(
                self.output_root, self.watermark_scope
            )
            if latest is not None:
                start = datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)
                self.start_date = start.strftime("%Y%m%d")
            else:
                _parse_date(
                    self.incremental_initial_start_date,
                    "incremental_initial_start_date",
                )
                self.start_date = self.incremental_initial_start_date
        start = _parse_date(self.start_date, "start_date")
        end = _parse_date(self.end_date, "end_date")
        self.no_work = start > end


def _parse_date(value, field_name):
    """将八位日期配置解析为 ``datetime``。

    参数：
        value: 形如 ``YYYYMMDD`` 的日期字符串。
        field_name: 配置字段名，仅用于生成可定位的错误消息。

    返回：
        对应日期零点的 ``datetime`` 对象。
    """
    try:
        return datetime.strptime(value, "%Y%m%d")
    except (TypeError, ValueError):
        raise ValueError("{0} 必须是 YYYYMMDD 日期".format(field_name))


def _latest_completed_run_date(output_root, watermark_scope):
    """查找最近一个业务范围一致且所有引用分区完整的整日水位。

    参数：
        output_root: 下载器输出根目录；检查其 ``run_complete/date=*`` 子目录。
        watermark_scope: 当前数据集、显式证券池和板块组成的水位业务范围。

    返回：
        最近整日完成的八位日期；没有任何合法水位时返回 ``None``。
    """
    root = Path(output_root)
    completion_root = root / "run_complete"
    if not completion_root.is_dir():
        return None
    candidates = []
    for directory in completion_root.glob("date=*"):
        value = directory.name.split("=", 1)[-1]
        try:
            _parse_date(value, "partition_date")
        except ValueError:
            continue
        candidates.append((value, directory))
    for value, directory in sorted(candidates, key=lambda item: item[0], reverse=True):
        marker = directory / "_SUCCESS.json"
        try:
            with marker.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if metadata.get("watermark_scope") != watermark_scope:
            continue
        required = metadata.get("required_partitions")
        if not isinstance(required, list) or not required:
            continue
        if all(_valid_referenced_partition(root, item) for item in required):
            return value
    return None


def _valid_referenced_partition(output_root, relative_value):
    """验证整日水位引用的单个业务分区。

    参数：
        output_root: 下载器输出根目录。
        relative_value: 水位元数据中的相对业务分区目录。

    返回：
        路径安全且数据文件 SHA-256 与完成标记一致时返回 ``True``。
    """
    relative = Path(str(relative_value))
    if relative.is_absolute() or ".." in relative.parts:
        return False
    directory = Path(output_root) / relative
    data_path = directory / "data.csv"
    success_path = directory / "_SUCCESS.json"
    if not data_path.is_file() or not success_path.is_file():
        return False
    try:
        with success_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        digest = hashlib.sha256()
        with data_path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return metadata.get("sha256") == digest.hexdigest()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
