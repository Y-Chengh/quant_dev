# -*- coding: utf-8 -*-
"""上市退市信息的收集与证券池推导。"""

from datetime import datetime, timedelta

import pandas as pd

from ..gateway import INSTRUMENT_INFO_COLUMNS
from .base import _RunnerState
from .helpers import _contains_error

#: 证券名称历史的分区字段名。刻意不用 ``date``：同一根目录下 ``kline_1d/date=``
#: 是交易日，这里是观测日，两者语义不同，同名迟早会被下游当成一回事。
INSTRUMENT_HISTORY_PARTITION = "observed_date"


class _InstrumentInfoMixin(_RunnerState):
    """收集上市退市快照、据此推导实际证券池，并按观测日累积名称历史。"""

    def _collect_instrument_info(self):
        """读取证券池及历史存续证券的上市退市信息并保存当前快照表。

        返回：
            ``(DataFrame, failed)``；快照写入 ``instrument_info/snapshot=latest``，
            并在 ``save_instrument_history`` 打开时额外写一份
            ``instrument_info/observed_date=YYYYMMDD``。当前
            证券池任一代码详情缺失时失败；仅历史快照代码缺失时降级为警告并从新快照
            移除，避免已被大 QMT 清除的退市代码永久阻塞任务。
        """
        info_symbols = self._instrument_info_symbols()
        pool = set(self.symbols)
        carried = {code for code in info_symbols if code not in pool}
        self.logger.info("[instrument_info] 开始获取上市退市信息 证券 1/%d", len(info_symbols))
        frame, issues = self.gateway.fetch_instrument_info(info_symbols)
        issues = [self._downgrade_carried_issue(issue, carried) for issue in issues]
        self.issues.extend(issues)
        for issue in issues:
            log_method = self.logger.error if issue.get("level") == "ERROR" else self.logger.warning
            log_method(
                "详细问题 dataset=instrument_info symbols=%s issue_level=%s code=%s date=%s message=%s",
                ",".join(self.symbols),
                issue.get("level", "WARNING"),
                issue.get("code", ""),
                issue.get("date", ""),
                issue.get("message", ""),
            )
        returned = set(frame["code"].astype(str)) if not frame.empty else set()
        failed = _contains_error(issues) or not pool <= returned
        if len(info_symbols):
            self.logger.info(
                "[instrument_info] 完成获取上市退市信息 证券 %d/%d (%.1f%%)",
                len(frame),
                len(info_symbols),
                len(frame) * 100.0 / len(info_symbols),
            )
        if not failed:
            self._write_partition(
                "instrument_info",
                "snapshot",
                "latest",
                frame,
                INSTRUMENT_INFO_COLUMNS,
                ["code"],
                ["code"],
                overwrite=True,
            )
            if self.config.save_instrument_history:
                self._write_instrument_history(frame)
        return frame, failed

    def _write_instrument_history(self, frame):
        """把本次快照按观测日再存一份 ``instrument_info/observed_date=YYYYMMDD``。

        分区键是**观测日**（``config.observation_date``）而不是交易日或 ``end_date``：
        大 QMT 的证券详情只有当前状态，回补历史行情时拿到的仍是今天的简称，按请求区
        间归档会凭空造出一段假历史。按观测日归档则永远只声明「这一天我们看到的是这
        些名称」，回补、重跑和跨年长任务都不会污染其它日期。分区名刻意不叫 ``date``：
        同根目录下 ``kline_1d/date=`` 是交易日，两者语义不同，同名迟早被当成一回事。

        同一观测日重复运行时与已有分区**求并集**而不是直接覆盖。正常情况下窄证券池
        的任务也不会让快照缩水——``_instrument_info_symbols`` 已经把上一份快照中未
        退市的代码并回查询池——但以下三种情况都会让当日快照真的变小：快照被删除或
        损坏时它退化成只有本次证券池；被 ``_downgrade_carried_issue`` 移除的代码随之
        消失；``end_date`` 靠前的回补任务会按 ``end_date`` 前一日裁掉更早退市的代码。
        这类信息事后无法从任何接口补回，因此宁可保留：本次结果优先，本次没有的代码
        保留已有行。

        落表失败只记 ``WARNING`` 不中断任务：名称历史是纯增量的旁路数据，没有任何
        下游流程依赖它，为它让整轮下载失败得不偿失；快照本身已经写成功。

        参数：
            frame: ``fetch_instrument_info`` 返回并已通过校验的本次证券详情表。

        返回：
            无返回值；分区写入结果只写日志。
        """
        observation_date = self.config.observation_date
        try:
            merged = self._merge_instrument_history(frame, observation_date)
            result = self._write_partition(
                "instrument_info",
                INSTRUMENT_HISTORY_PARTITION,
                observation_date,
                merged,
                INSTRUMENT_INFO_COLUMNS,
                ["code"],
                ["code"],
                overwrite=True,
            )
        except Exception as error:
            self.issues.add(
                "WARNING",
                "instrument_info",
                "",
                observation_date,
                "证券名称历史分区落表失败，不影响本次下载: {0}".format(error),
            )
            # 用 exception 而不是 warning：这里的 except 也会兜住编程错误，没有栈的话
            # 一个笔误就能在无人报警的情况下静默烧掉几年的名称历史。
            self.logger.exception(
                "[instrument_info] 名称历史落表失败 observed_date=%s error=%s",
                observation_date,
                error,
            )
            return
        self.logger.info(
            "[instrument_info] 名称历史落表 observed_date=%s rows=%s path=%s",
            observation_date,
            result["rows"],
            result["path"],
        )

    def _merge_instrument_history(self, frame, observation_date):
        """把本次证券详情与同一观测日已有分区求并集。

        参数：
            frame: 本次取得的证券详情表，同一代码只应出现一行。
            observation_date: 八位观测日，即目标分区的分区值。

        返回：
            合并后的 ``DataFrame``；已有分区不存在、为空或没有 ``code`` 列时原样返回
            ``frame``。已有分区存在且非空却读不出来时抛出 ``ValueError``，由调用方降级
            为警告并**放弃写入**：那份文件里可能还有今天唯一的一份名称，不能覆盖掉。
            零字节文件不在此列，它不可能存有任何名称，照常覆盖，否则该观测日会被一份
            空文件永久挡住。列顺序与去重排序由分区写入层统一负责。
        """
        existing = self.store.read_partition(
            "instrument_info", INSTRUMENT_HISTORY_PARTITION, observation_date, dtype=str
        )
        if existing is None:
            data_path = self.store.partition_directory(
                "instrument_info", INSTRUMENT_HISTORY_PARTITION, observation_date
            ) / "data.csv"
            if data_path.is_file() and data_path.stat().st_size > 0:
                raise ValueError(
                    "同一观测日已有名称历史分区无法解析，拒绝覆盖: {0}".format(data_path)
                )
            return frame
        if "code" not in existing.columns or existing.empty:
            return frame
        if frame is None or frame.empty:
            return existing
        refreshed = set(frame["code"].astype(str))
        carried = existing[~existing["code"].astype(str).isin(refreshed)]
        if carried.empty:
            return frame
        self.logger.info(
            "[instrument_info] 名称历史合并同日已有记录 observed_date=%s carried=%d",
            observation_date,
            len(carried),
        )
        # 先统一转成 object 再拼接：已有分区一律按字符串读回，若它比本次少了某些列，
        # concat 会给本次的整数和布尔列补 NaN 而把它们提升成浮点，归档值就从 0 变成
        # 0.0。大 QMT 内置的旧版 pandas.concat 不支持 sort 参数。
        return pd.concat([frame.astype(object), carried], ignore_index=True)

    def _instrument_info_symbols(self):
        """合并当前证券池与上一快照中前一日尚未退市的证券代码。

        返回：
            去重并按代码排序的证券代码列表；历史快照不可读时仅返回当前证券池。
        """
        current = {str(code).strip().upper() for code in self.symbols if str(code).strip()}
        if not self.store.is_partition_complete("instrument_info", "snapshot", "latest"):
            return sorted(current)
        previous = self.store.read_partition(
            "instrument_info",
            "snapshot",
            "latest",
            dtype={"code": str, "expire_date": str},
        )
        if previous is None or not {"code", "expire_date"}.issubset(previous.columns):
            return sorted(current)

        reference_date = datetime.strptime(self.config.end_date, "%Y%m%d") - timedelta(days=1)
        reference_text = reference_date.strftime("%Y%m%d")
        for row in previous.to_dict("records"):
            code = str(row.get("code") or "").strip().upper()
            if not code or "." not in code:
                continue
            raw_expire_date = row.get("expire_date")
            expire_date = "" if pd.isna(raw_expire_date) else str(raw_expire_date).strip()
            # 空值表示尚未退市；退市日等于前一日时仍属于前一日的证券池。
            if not expire_date:
                current.add(code)
                continue
            try:
                datetime.strptime(expire_date, "%Y%m%d")
            except ValueError:
                # 单行退市日期损坏不应静默丢弃其余行；保留该代码继续跟踪生命周期。
                self.logger.warning(
                    "上市退市快照存在无法解析的退市日期 code=%s expire_date=%s，仍保留该代码",
                    code,
                    expire_date,
                )
                current.add(code)
                continue
            if expire_date >= reference_text:
                current.add(code)
        return sorted(current)
