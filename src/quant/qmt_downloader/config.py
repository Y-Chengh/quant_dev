# -*- coding: utf-8 -*-
"""下载器 JSON 配置加载与业务校验。"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # 大 QMT 内置 Python 可能未提供 zoneinfo
    ZoneInfo = None


SUPPORTED_MODES = ("backfill", "incremental", "repair")
# 交易日历只是整体覆盖的运行时快照，不按交易日分区，也不影响任何业务分区的内容，
# 因此单独成名并在任务键、分区范围和整日水位范围中排除，见 BUSINESS_DATASETS。
CALENDAR_DATASET = "trading_calendar"
BUSINESS_DATASETS = (
    "kline_1d",
    "finance_raw",
    "finance_daily",
    "corporate_actions",
)
SUPPORTED_DATASETS = BUSINESS_DATASETS + (CALENDAR_DATASET,)
try:
    _SHANGHAI_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo is not None else None
except (OSError, ValueError, KeyError):
    # Python 有 zoneinfo 但未安装时区数据库；ZoneInfoNotFoundError 继承 KeyError，
    # Windows/大 QMT 内置 Python 通常没有 tzdata 包，会走到这里。
    _SHANGHAI_TZ = None
if _SHANGHAI_TZ is None:
    _SHANGHAI_TZ = timezone(timedelta(hours=8), name="CST")


def strip_jsonc(text):
    """把 JSONC 文本转换为标准 JSON 文本，便于交给 ``json`` 解析。

    去掉 ``//`` 行注释、``/* */`` 块注释，并清除对象或数组闭合括号前的多余逗号。
    注释和多余逗号都替换为等长空白且保留换行，因此解析报错的行列号仍与原始文件
    一致。字符串字面量内部的 ``//``、``/*`` 和逗号原样保留，Windows 路径中的
    ``\\\\`` 等转义序列不会被误判为字符串结束。

    参数：
        text: 配置文件原始文本，已去除 BOM，可以包含注释和多余逗号。

    返回：
        与输入等长的标准 JSON 文本；本函数不校验语义，非法结构仍由 ``json`` 报错。
    """
    result = []
    index = 0
    length = len(text)
    in_string = False
    # 最近一个尚未确认是否多余的逗号在 result 中的下标；-1 表示当前没有待定逗号。
    pending_comma = -1
    while index < length:
        char = text[index]
        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < length:
                result.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            pending_comma = -1
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            while index < length and text[index] not in "\r\n":
                result.append(" ")
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("下载器配置存在未闭合的块注释 /*")
            for blanked in text[index : end + 2]:
                result.append(blanked if blanked in "\r\n" else " ")
            index = end + 2
            continue
        if char == ",":
            pending_comma = len(result)
        elif char in "}]":
            if pending_comma >= 0:
                result[pending_comma] = " "
            pending_comma = -1
        elif not char.isspace():
            # 逗号后出现真实内容说明它是分隔符；空白和注释不改变待定状态。
            pending_comma = -1
        result.append(char)
        index += 1
    return "".join(result)


class DownloaderConfig(object):
    """保存经过校验的大 QMT 下载任务配置。"""

    def __init__(self, values, source_path=None):
        """初始化下载配置并校验日期、批量大小及数据集。

        参数：
            values: JSON 配置解析后的字典；必须包含输出目录、模式、起止日期，
                股票列表可为空并改由 ``sector`` 指定大 QMT 板块，``expired_sectors``
                可额外并入过期（退市）板块。
            source_path: 配置文件路径，仅用于日志记录；直接以字典构造时可省略。
        """
        self.source_path = None if source_path is None else Path(source_path)
        self.output_root = Path(values.get("output_root", r"D:\qmt_data_test"))
        self.mode = str(values.get("mode", "backfill"))
        self.start_date = str(values.get("start_date", ""))
        self.end_date = str(values.get("end_date", ""))
        self.auto_start_requested = self.start_date.lower() == "auto"
        self.incremental_initial_start_date = str(
            values.get("incremental_initial_start_date", "")
        )
        lag_value = values.get("incremental_lag_days", 1)
        self.incremental_lag_days_auto = str(lag_value).strip().lower() == "auto"
        self.incremental_lag_auto_cutoff = str(
            values.get("incremental_lag_auto_cutoff", "16:00")
        ).strip()
        # 配置读取时刻同时用于自动滞后判定和 auto 结束日解析，始终记录。
        self.incremental_lag_decision_time = datetime.now(_SHANGHAI_TZ)
        self.incremental_lag_days = _resolve_incremental_lag_days(
            lag_value, self.incremental_lag_decision_time, self.incremental_lag_auto_cutoff
        )
        # 本次运行**观测**到证券名称等快照信息的日期，复用上面的配置读取时刻，保证
        # 同一次运行内取值稳定。它是观测日而不是交易日：回补 2020 年行情时，大 QMT
        # 返回的仍是今天的证券简称，按 end_date 归档等于凭空伪造一段历史。
        self.observation_date = self.incremental_lag_decision_time.strftime("%Y%m%d")
        self.no_work = False
        self.symbols = tuple(str(item).strip() for item in values.get("symbols", []) if str(item).strip())
        self.sector = str(values.get("sector", "")).strip()
        # 过期（退市）板块，例如 ``过期沪深A股``。默认关闭：需要先在大 QMT 界面端
        # “数据管理 → 过期合约数据 → 过期合约列表”下载并重启客户端，板块名以本机
        # ``get_sector_list()`` 的实际返回为准。
        # 去重并排序：板块名是集合语义，重复写同一个板块不应改变证券池，却会让
        # watermark_scope 与历史 _SUCCESS.json 失配，把自动增量静默拖回全量重跑。
        self.expired_sectors = tuple(
            sorted(
                set(
                    str(item).strip()
                    for item in values.get("expired_sectors", [])
                    if str(item).strip()
                )
            )
        )
        self.batch_size = int(values.get("batch_size", 100))
        self.save_workers = int(values.get("save_workers", 1))
        self.retry_count = int(values.get("retry_count", 3))
        self.kline_gap_retry_count = int(values.get("kline_gap_retry_count", 2))
        self.download_kline = bool(values.get("download_kline", True))
        self.download_kline_batch = bool(values.get("download_kline_batch", True))
        self.overwrite_completed_partition = bool(
            values.get("overwrite_completed_partition", self.mode == "repair")
        )
        self.datasets = tuple(values.get("datasets", SUPPORTED_DATASETS))
        # 交易日历与日线平级，通过 datasets 中的 trading_calendar 控制是否落表；它
        # 不产生按日分区，因此下面所有影响断点匹配和分区口径的派生值都只看业务数据集，
        # 使已有输出目录在开关切换后仍能续跑、跳过和推进水位。
        self.save_trading_calendar = CALENDAR_DATASET in self.datasets
        self.business_datasets = tuple(
            name for name in self.datasets if name != CALENDAR_DATASET
        )
        # instrument_info 只写 snapshot=latest 时会被每次运行整体覆盖，历史证券简称
        # （也就是历史 ST 状态）无法回溯，而大 QMT 没有任何接口能补回过去的名称。
        # 打开本开关会在快照之外按观测日再存一份，从此逐日累积出可审计的名称历史。
        # 它刻意不进入 job_key、partition_scope 与 watermark_scope：这份数据不改变
        # 任何业务分区的内容，纳入范围只会让开关一切换就把已有分区判为口径不符。
        self.save_instrument_history = bool(values.get("save_instrument_history", True))
        self.allow_partial_finance = bool(values.get("allow_partial_finance", False))
        self.calendar_symbol = str(values.get("calendar_symbol", "000001.SH")).strip().upper()
        self.watermark_scope = {
            "datasets": sorted(self.business_datasets),
            "symbols": sorted(str(code).strip().upper() for code in self.symbols),
            "sector": self.sector,
        }
        # 未配置过期板块时不写入该键：既有输出目录的水位标记里没有它，无条件写入会让
        # 全部历史水位失配，自动增量退回初始起点重跑。一旦配置了过期板块，键出现导致
        # 水位失配正是期望结果——证券池已经变了，旧水位不再代表同一业务范围。
        if self.expired_sectors:
            self.watermark_scope["expired_sectors"] = list(self.expired_sectors)
        self.finance_lookback_start = str(
            values.get("finance_lookback_start", "20000101")
        )
        self.log_max_bytes = int(values.get("log_max_bytes", 20 * 1024 * 1024))
        self.log_backup_count = int(values.get("log_backup_count", 10))
        self._resolve_incremental_dates()
        self._validate()

    @classmethod
    def from_json(cls, path):
        """从 UTF-8 JSONC 文件构造下载配置。

        参数：
            path: 配置文件路径；内容为 JSON 对象，允许 ``//`` 行注释、``/* */``
                块注释和对象或数组末尾的多余逗号，未知键会被忽略。

        返回：
            已完成字段和业务规则校验的 ``DownloaderConfig``。
        """
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8-sig") as handle:
            text = handle.read()
        try:
            values = json.loads(strip_jsonc(text))
        except ValueError as error:
            # 注释和多余逗号已替换为等长空白，因此报错的行列号仍指向原文件位置。
            raise ValueError(
                "下载器配置 {0} 解析失败：{1}".format(config_path, error)
            )
        if not isinstance(values, dict):
            raise ValueError("下载器配置根节点必须是 JSON 对象")
        return cls(values, config_path)

    def describe(self):
        """列出本次运行的全部生效配置，供日志起始处完整记录。

        自动解析后的起止日期、滞后天数和派生数据集分组都按最终取值给出，因此
        日志中的值与下载器实际使用的值一致，排查时不必再回头对照配置文件。

        返回：
            ``(名称, 文本值)`` 二元组列表，顺序与配置读取顺序一致。
        """
        return [
            ("config_path", "" if self.source_path is None else str(self.source_path)),
            ("output_root", str(self.output_root)),
            ("mode", self.mode),
            ("start_date", self.start_date),
            ("end_date", self.end_date),
            ("auto_start_requested", str(self.auto_start_requested)),
            ("incremental_initial_start_date", self.incremental_initial_start_date),
            ("incremental_lag_days", str(self.incremental_lag_days)),
            ("incremental_lag_days_auto", str(self.incremental_lag_days_auto)),
            ("incremental_lag_auto_cutoff", self.incremental_lag_auto_cutoff),
            (
                "incremental_lag_decision_time",
                self.incremental_lag_decision_time.strftime("%Y-%m-%d %H:%M:%S %z"),
            ),
            ("observation_date", self.observation_date),
            ("no_work", str(self.no_work)),
            ("symbol_count", str(len(self.symbols))),
            ("symbols", ", ".join(self.symbols)),
            ("sector", self.sector),
            ("expired_sectors", ", ".join(self.expired_sectors)),
            ("batch_size", str(self.batch_size)),
            ("save_workers", str(self.save_workers)),
            ("retry_count", str(self.retry_count)),
            ("kline_gap_retry_count", str(self.kline_gap_retry_count)),
            ("download_kline", str(self.download_kline)),
            ("download_kline_batch", str(self.download_kline_batch)),
            (
                "overwrite_completed_partition",
                str(self.overwrite_completed_partition),
            ),
            ("datasets", ", ".join(str(name) for name in self.datasets)),
            ("save_trading_calendar", str(self.save_trading_calendar)),
            ("business_datasets", ", ".join(self.business_datasets)),
            ("save_instrument_history", str(self.save_instrument_history)),
            ("allow_partial_finance", str(self.allow_partial_finance)),
            ("calendar_symbol", self.calendar_symbol),
            (
                "watermark_scope",
                json.dumps(self.watermark_scope, ensure_ascii=False, sort_keys=True),
            ),
            ("finance_lookback_start", self.finance_lookback_start),
            ("log_max_bytes", str(self.log_max_bytes)),
            ("log_backup_count", str(self.log_backup_count)),
        ]

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
        if not self.symbols and not self.sector and not self.expired_sectors:
            raise ValueError("symbols、sector 与 expired_sectors 至少需要配置一个")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        if self.save_workers <= 0 or self.save_workers > 16:
            raise ValueError("save_workers 必须在 1 到 16 之间")
        if self.retry_count <= 0:
            raise ValueError("retry_count 必须大于 0")
        if self.kline_gap_retry_count <= 0 or self.kline_gap_retry_count > 10:
            raise ValueError("kline_gap_retry_count 必须在 1 到 10 之间")
        if self.incremental_lag_days < 0:
            raise ValueError("incremental_lag_days 不能小于 0")
        _parse_lag_cutoff(self.incremental_lag_auto_cutoff)
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
            current_time = self.incremental_lag_decision_time
            target = current_time.date() - timedelta(days=self.incremental_lag_days)
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


def _resolve_incremental_lag_days(value, now=None, cutoff="16:00"):
    """解析增量滞后天数；自动模式以上海时间 ``cutoff`` 为当日数据就绪边界。

    判定发生在读取配置时；请确保定时任务在大 QMT 完成当日数据同步之后启动，
    否则应通过 ``incremental_lag_auto_cutoff`` 推迟边界。

    参数：
        value: 配置中的滞后天数，非负整数表示固定滞后；``"auto"`` 表示自动判断。
        now: 用于自动判断的时间；缺省取当前上海时间，测试时可传入固定时间。
        cutoff: ``HH:MM`` 边界文本，来自 ``incremental_lag_auto_cutoff``。

    返回：
        自动模式在边界前返回 ``1``、边界及以后返回 ``0``；固定模式返回其整数值。
    """
    if str(value).strip().lower() != "auto":
        return int(value)
    hour, minute = _parse_lag_cutoff(cutoff)
    current = now or datetime.now(_SHANGHAI_TZ)
    return 0 if (current.hour, current.minute) >= (hour, minute) else 1


def _parse_lag_cutoff(value):
    """解析自动滞后判定的 ``HH:MM`` 边界文本。

    参数：
        value: 24 小时制 ``HH:MM`` 字符串，例如 ``16:00`` 或 ``17:30``。

    返回：
        ``(hour, minute)`` 整数二元组；格式非法时抛出 ``ValueError``。
    """
    match = re.match(r"^([01]?\d|2[0-3]):([0-5]\d)$", str(value).strip())
    if match is None:
        raise ValueError("incremental_lag_auto_cutoff 必须是 24 小时制 HH:MM")
    return int(match.group(1)), int(match.group(2))


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
