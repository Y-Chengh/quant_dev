# -*- coding: utf-8 -*-
"""分区目录发现、日线根目录定位与分区元数据读取。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import _CheckerState
from .utils import _csv_row_count, _file_sha256, _select_paths_in_range


class _PartitionDiscoveryMixin(_CheckerState):
    """发现业务分区、定位日线根目录并读取分区完成标记。"""

    def _discover_partitions(self) -> None:
        """发现日线分区，并校验完成标记、哈希、行数和证券池口径。

        返回：
            无返回值；发现的问题追加到当前自检器的问题集合。
        """

        kline_root = self._resolve_kline_root()
        if self.source_mode == "staging":
            self._discover_staging_partitions(kline_root)
            return
        if not kline_root.is_dir():
            self._add_issue(
                "KLINE_DATASET_MISSING",
                "ERROR",
                "kline_1d",
                "日线数据集目录不存在，无法执行行情完整性检查。",
                expected="存在 kline_1d/date=YYYYMMDD/data.csv 日分区",
                actual="目录不存在: {0}".format(kline_root),
                possible_causes="日线尚未下载、输出目录配置错误或目录被移动",
                suggested_action="核对下载配置 output_root，并先完成 kline_1d 下载",
                source_file=str(kline_root),
            )
            return
        directories = _select_paths_in_range(
            sorted(kline_root.glob("date=*")),
            r"date=(\d{8})",
            self.config.start_date,
            self.config.end_date,
        )
        self.logger.info(
            "[自检] 待校验日线分区 %d 个 root=%s", len(directories), kline_root
        )
        for index, directory in enumerate(directories):
            # 每个分区都要读完成标记并对 data.csv 做全文件 SHA-256，是整轮自检里
            # 仅次于逐日扫描的耗时环节，因此在循环入口就报告进度。
            self._log_step_progress(
                "分区校验", index, len(directories), directory.name, pending=True
            )
            if not directory.is_dir():
                continue
            raw_date = directory.name.split("=", 1)[-1]
            if re.fullmatch(r"\d{8}", raw_date) is None:
                self._add_issue(
                    "INVALID_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "日线分区目录名不是合法的 YYYYMMDD 日期。",
                    actual="分区名={0}".format(directory.name),
                    expected="date=YYYYMMDD",
                    suggested_action="修复目录名或重新生成该分区，不要把非法目录纳入正式数据",
                    source_file=str(directory),
                )
                continue
            date_value = raw_date
            if date_value in self.partition_paths:
                self._add_issue(
                    "DUPLICATE_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "多个日线目录映射到同一个交易日期。",
                    date=date_value,
                    expected="每个 YYYYMMDD 日期恰好一个 date=YYYYMMDD 目录",
                    actual="重复目录包含: {0}, {1}".format(
                        self.partition_paths[date_value], directory
                    ),
                    possible_causes="分区目录复制、大小写/路径同步冲突或人工移动文件",
                    suggested_action="保留唯一可信分区并使用 repair 重建重复日期",
                    source_file=str(directory),
                )
                continue
            self.partition_paths[date_value] = directory
            metadata = self._read_partition_metadata(directory, date_value)
            if metadata is not None:
                scope = metadata.get("partition_scope")
                if isinstance(scope, dict):
                    if self.reference_scope is None:
                        self.reference_scope = scope
                    elif scope != self.reference_scope:
                        self._add_issue(
                            "PARTITION_SCOPE_MISMATCH",
                            "ERROR",
                            "kline_1d",
                            "该日期分区的证券池或抽取口径与其他分区不一致。",
                            date=date_value,
                            expected=json.dumps(self.reference_scope, ensure_ascii=False, sort_keys=True),
                            actual=json.dumps(scope, ensure_ascii=False, sort_keys=True),
                            evidence="不同口径的数据不能视为同一个完整全样本",
                            possible_causes="曾使用不同证券池或数据集配置写入同一输出目录",
                            suggested_action="按统一配置使用 repair 重建异常分区，或改用独立输出目录",
                            source_file=str(directory / "_SUCCESS.json"),
                        )

    def _resolve_kline_root(self) -> Path:
        """解析最终分区或 staging 日目录的实际来源路径。

        返回：
            最终布局的 ``kline_1d`` 目录，或 staging 作业目录（其中包含
            ``kline_daily_YYYYMMDD`` 子目录）。当最终目录不存在时，会自动
            选择 output_root/staging 下唯一可识别的 QMT 作业目录。
        """

        configured = self.config.staging_root
        if configured is not None:
            candidate = configured.resolve()
            if (candidate / "kline_1d").is_dir() and not any(
                candidate.glob("kline_daily_*")
            ):
                self.source_mode = "staging"
                self.data_root = candidate
                return candidate
            children = [
                item
                for item in candidate.iterdir()
                if item.is_dir() and any(item.glob("kline_daily_*"))
            ] if candidate.is_dir() else []
            if len(children) == 1:
                self.source_mode = "staging"
                self.data_root = children[0]
                return children[0]
            if len(children) > 1:
                self._add_issue(
                    "STAGING_JOB_AMBIGUOUS",
                    "ERROR",
                    "kline_1d",
                    "staging 目录包含多个可审计 QMT 作业，无法安全猜测数据来源。",
                    expected="--staging-root 直接指向唯一作业目录",
                    actual=", ".join(str(item) for item in children),
                    evidence="多个作业都包含 kline_daily_YYYYMMDD 日目录",
                    possible_causes="历史作业未清理或命令行只传入了 staging 父目录",
                    suggested_action="把 --staging-root 改为具体的 staging\\qmt_<job_key> 目录",
                    source_file=str(candidate),
                )
            self.source_mode = "staging"
            self.data_root = candidate
            return candidate
        final_root = self.root / "kline_1d"
        if final_root.is_dir() and any(final_root.glob("date=*")):
            return final_root
        staging_parent = self.root / "staging"
        children = [
            item
            for item in staging_parent.iterdir()
            if item.is_dir() and any(item.glob("kline_daily_*"))
        ] if staging_parent.is_dir() else []
        if len(children) == 1:
            self.source_mode = "staging"
            self.data_root = children[0]
            return children[0]
        return final_root

    def _discover_staging_partitions(self, staging_root: Path) -> None:
        """发现 staging 中按日保存的 QMT 行情目录。

        参数：
            staging_root: 具体 QMT 作业目录，子目录名称应严格为
                ``kline_daily_YYYYMMDD``。
        返回：
            无返回值；发现的日期映射写入 ``partition_paths``，结构问题写入 issues。
        """

        if not staging_root.is_dir():
            self._add_issue(
                "KLINE_DATASET_MISSING",
                "ERROR",
                "kline_1d",
                "staging 行情作业目录不存在，无法执行日线完整性审计。",
                expected="存在 staging/qmt_<job_key>/kline_daily_YYYYMMDD",
                actual="目录不存在: {0}".format(staging_root),
                evidence="未发现任何按日行情目录",
                possible_causes="下载作业尚未开始、作业 key 错误或 staging 被移动",
                suggested_action="检查 D:\\qmt_kline\\staging 下的作业目录并传入 --staging-root",
                source_file=str(staging_root),
            )
            return
        matched = list(staging_root.glob("kline_daily_*"))
        if not matched:
            self._add_issue(
                "KLINE_DATASET_MISSING",
                "ERROR",
                "kline_1d",
                "staging 作业中没有按日行情目录。",
                expected="至少存在一个 kline_daily_YYYYMMDD 目录",
                actual="未找到 kline_daily_*",
                evidence="当前作业可能只有未整理的 kline_1d 批次文件",
                possible_causes="下载尚未完成或使用了不兼容的 staging 布局",
                suggested_action="确认 runner 已生成 kline_daily_YYYYMMDD，或先 finalize 成最终分区",
                source_file=str(staging_root),
            )
            return
        flat_root = staging_root / "kline_1d"
        flat_batch_ids = {
            path.stem
            for path in flat_root.glob("batch_*.csv")
        } if flat_root.is_dir() else set()
        staging_directories = _select_paths_in_range(
            sorted(matched, key=lambda item: item.name),
            r"kline_daily_(\d{8})",
            self.config.start_date,
            self.config.end_date,
        )
        self.logger.info(
            "[自检] 待校验 staging 日目录 %d 个 root=%s",
            len(staging_directories),
            staging_root,
        )
        for index, directory in enumerate(staging_directories):
            self._log_step_progress(
                "staging 目录校验",
                index,
                len(staging_directories),
                directory.name,
                pending=True,
            )
            match = re.fullmatch(r"kline_daily_(\d{8})", directory.name)
            if match is None:
                self._add_issue(
                    "INVALID_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "staging 日线目录名称不是严格的 YYYYMMDD 格式。",
                    expected="kline_daily_YYYYMMDD",
                    actual=directory.name,
                    evidence="目录名无法作为唯一交易日键",
                    possible_causes="目录被重命名、临时目录混入或日期格式错误",
                    suggested_action="修正目录名或重新生成该日期的 staging 批次",
                    source_file=str(directory),
                )
                continue
            date_value = match.group(1)
            if date_value in self.partition_paths:
                self._add_issue(
                    "DUPLICATE_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "多个 staging 日目录映射到同一个交易日。",
                    date=date_value,
                    expected="每个交易日只能有一个 kline_daily_YYYYMMDD 目录",
                    actual=str(directory),
                    evidence="重复日期会使覆盖率和缺失区间无法可靠计算",
                    possible_causes="作业重试残留或目录复制",
                    suggested_action="保留完整且唯一的一份日期目录后重新审计",
                    source_file=str(directory),
                )
                continue
            self.partition_paths[date_value] = directory
            if flat_batch_ids:
                daily_batch_ids = {
                    path.stem for path in directory.glob("batch_*.csv")
                }
                missing_ids = sorted(flat_batch_ids.difference(daily_batch_ids))
                if missing_ids:
                    self.staging_daily_gap_dates.append((date_value, missing_ids))
        if self.staging_daily_gap_dates:
            sample = self.staging_daily_gap_dates[:3]
            self._add_issue(
                "STAGING_DAILY_BATCH_GAPS",
                "ERROR",
                "kline_1d",
                "staging 按日目录缺少 flat kline_1d 中存在的批次，按日覆盖率不能代表完整作业数据。",
                expected="每个 kline_daily_YYYYMMDD 包含与 staging/kline_1d 相同的 batch_*.csv 集合",
                actual="{0} 个日期存在缺批；示例 {1}".format(len(self.staging_daily_gap_dates), sample),
                evidence="flat kline_1d 批次可作为缺失证券的补充来源，但当前扫描只读取按日目录",
                possible_causes="批次写入中断、按日整理步骤漏拷贝或多个下载阶段未合并",
                suggested_action="先修复/重新生成缺失批次，或将 flat kline_1d finalize 成完整 date=YYYYMMDD 分区后再审计",
                source_file=str(staging_root),
            )

    def _read_partition_metadata(
        self, directory: Path, date_value: str
    ) -> dict[str, Any] | None:
        """读取并验证一个日线分区完成标记。

        参数：
            directory: 当前 ``date=YYYYMMDD`` 日线分区目录。
            date_value: 从目录名解析得到的八位交易日期。

        返回：
            成功解析的完成元数据；文件不存在或 JSON 损坏时返回 ``None``。
        """

        data_path = directory / "data.csv"
        marker_path = directory / "_SUCCESS.json"
        if not data_path.is_file() or not marker_path.is_file():
            missing = [
                path.name for path in (data_path, marker_path) if not path.is_file()
            ]
            self._add_issue(
                "PARTITION_FILE_MISSING",
                "ERROR",
                "kline_1d",
                "日线分区缺少数据文件或完成标记。",
                date=date_value,
                expected="data.csv 与 _SUCCESS.json 同时存在",
                actual="缺少: {0}".format(", ".join(missing)),
                evidence="不完整分区不能证明写入已经原子完成",
                possible_causes="下载中断、人工删除文件或磁盘写入失败",
                suggested_action="使用 repair 模式重新生成整个日期分区",
                source_file=str(directory),
            )
            return None
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._add_issue(
                "PARTITION_MARKER_INVALID",
                "ERROR",
                "kline_1d",
                "日线分区完成标记无法解析。",
                date=date_value,
                expected="合法的 UTF-8 JSON 对象",
                actual="{0}: {1}".format(type(exc).__name__, exc),
                possible_causes="完成标记写入中断或文件被修改",
                suggested_action="使用 repair 模式重建分区，不要手工伪造完成标记",
                source_file=str(marker_path),
            )
            return None
        if not isinstance(metadata, dict):
            self._add_issue(
                "PARTITION_MARKER_SCHEMA_INVALID",
                "ERROR",
                "kline_1d",
                "日线分区完成标记的 JSON 根节点不是对象。",
                date=date_value,
                expected="JSON object containing sha256, rows and partition_scope",
                actual="根节点类型={0}".format(type(metadata).__name__),
                possible_causes="完成标记被截断、人工替换或写入了错误格式",
                suggested_action="使用 repair 模式重建该日期分区，不要手工修改 _SUCCESS.json",
                source_file=str(marker_path),
            )
            return None
        expected_hash = str(metadata.get("sha256", ""))
        actual_hash = _file_sha256(data_path)
        if expected_hash != actual_hash:
            self._add_issue(
                "PARTITION_HASH_MISMATCH",
                "ERROR",
                "kline_1d",
                "日线 CSV 内容与完成标记记录的 SHA-256 不一致。",
                date=date_value,
                field="sha256",
                expected=expected_hash or "_SUCCESS.json 中应存在 sha256",
                actual=actual_hash,
                evidence="数据文件在完成标记生成后发生修改或损坏",
                possible_causes="人工编辑、磁盘损坏、非原子覆盖或错误同步",
                suggested_action="停止使用该分区并通过 repair 模式重新下载",
                source_file=str(data_path),
            )
        expected_rows = metadata.get("rows")
        actual_rows = _csv_row_count(data_path)
        try:
            rows_match = int(expected_rows) == actual_rows
        except (TypeError, ValueError):
            rows_match = False
        if not rows_match:
            self._add_issue(
                "PARTITION_ROW_COUNT_MISMATCH",
                "ERROR",
                "kline_1d",
                "日线 CSV 记录数与完成标记不一致。",
                date=date_value,
                field="rows",
                expected=str(expected_rows),
                actual=str(actual_rows),
                evidence="_SUCCESS.json 与实际 CSV 行数核对失败",
                possible_causes="文件被截断、追加、人工编辑或完成标记损坏",
                suggested_action="使用 repair 模式重新生成该日期分区",
                source_file=str(data_path),
            )
        return metadata if isinstance(metadata, dict) else None
