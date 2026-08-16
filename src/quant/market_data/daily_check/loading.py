"""审计所需参照数据的装载：区间、交易日历、证券生命周期与证券池。"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from quant.qmt_downloader.self_check import AuditIssue

from .base import _DailyCheckerState


class _ReferenceLoadingMixin(_DailyCheckerState):
    """装载审计区间、交易日历与证券静态信息。"""

    def _load_scope(self, connection) -> None:
        """确定审计区间与证券池。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；就地写入 ``start_date``、``end_date`` 与 ``codes``。
            库中没有任何行情时抛出 ``ValueError``。
        """
        row = connection.execute(
            "SELECT min(min_date), max(max_date) FROM monthly_inventory"
        ).fetchone()
        if not row or row[0] is None:
            raise ValueError(
                "日线库中没有任何行情，"
                "请先运行 python -m quant.cli.build_daily_store"
            )
        self.start_date = _resolve_date(self.config.start_date, row[0])
        self.end_date = _resolve_date(self.config.end_date, row[1])
        if self.start_date > self.end_date:
            raise ValueError(f"审计区间为空: {self.start_date} 至 {self.end_date}")
        if self.config.codes:
            self.codes = tuple(sorted({str(code).strip().upper() for code in self.config.codes}))
        else:
            rows = connection.execute("SELECT code FROM symbols ORDER BY code").fetchall()
            self.codes = tuple(str(item[0]) for item in rows)

    def _load_calendar(self, connection) -> None:
        """装载审计区间内的交易日历。

        ``trading_calendar`` 表缺失或覆盖不足时回落为「库中出现过的日期」，
        并记一条 ERROR：这种模式下无法发现整天缺失的情况。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；就地写入 ``calendar`` 与 ``calendar_is_authoritative``。
        """
        rows = connection.execute(
            "SELECT trade_date FROM trading_calendar WHERE trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date",
            [self.start_date, self.end_date],
        ).fetchall()
        # 统一归一化成 date：分片若被写成 TIMESTAMP，直接拿去做集合运算会让
        # 每一个交易日都被判成「日历之外」，刷出成千上万条假问题。
        calendar = [_as_date(item[0]) for item in rows]
        observed = [
            _as_date(item[0])
            for item in connection.execute(
                "SELECT DISTINCT trade_date FROM bars_1d WHERE trade_date BETWEEN ? AND ? "
                "ORDER BY trade_date",
                [self.start_date, self.end_date],
            ).fetchall()
        ]
        if calendar:
            self.calendar = tuple(calendar)
            self.calendar_is_authoritative = True
            missing = sorted(set(observed) - set(calendar))
            if missing:
                self._add_issue(
                    "DAILY_DATE_NOT_IN_CALENDAR",
                    "ERROR",
                    f"日线库存在交易日历之外的交易日 {len(missing)} 个",
                    start_date=str(missing[0]),
                    end_date=str(missing[-1]),
                    expected="全部交易日都在 trading_calendar 中",
                    actual="；".join(str(value) for value in missing[:10]),
                    possible_causes="交易日历未同步到最新，或行情里混入了非交易日",
                    suggested_action="重新导出 QMT 交易日历后再次入库",
                )
            return
        self.calendar = tuple(observed)
        self.calendar_is_authoritative = False
        self._add_issue(
            "DAILY_CALENDAR_UNAVAILABLE",
            "ERROR",
            "缺少交易日历，只能按库中出现过的日期推断，无法发现整天缺失",
            expected="trading_calendar 表覆盖审计区间",
            actual="区间内交易日历为空",
            possible_causes="下载器未开启 trading_calendar 数据集，或日历尚未入库",
            suggested_action="在下载器 datasets 中加入 trading_calendar 后重新同步",
        )

    def _load_instruments(self, connection) -> None:
        """装载证券静态信息并检查生命周期本身的合法性。

        参数：
            connection: 只读日线库连接。

        返回：
            无返回值；就地写入 ``instruments``。
        """
        frame = connection.execute(
            """
            SELECT code, instrument_name, open_date, expire_date, board, is_st
            FROM instruments ORDER BY code
            """
        ).df()
        if not frame.empty:
            frame["code"] = frame["code"].astype(str)
            frame["open_date"] = pd.to_datetime(frame["open_date"], errors="coerce")
            frame["expire_date"] = pd.to_datetime(frame["expire_date"], errors="coerce")
        self.instruments = frame.set_index("code") if not frame.empty else frame

        if frame.empty:
            self._add_issue(
                "DAILY_LIFECYCLE_UNAVAILABLE",
                "ERROR",
                "instruments 表为空，无法校验上市退市区间",
                expected="instruments 表包含全部证券的上市日",
                actual="0 行",
                possible_causes="入库时 instrument_info 快照缺失",
                suggested_action="确认下载器已产出 instrument_info 后重新同步",
            )
            return

        missing_open = frame[frame["open_date"].isna()]
        for code in missing_open["code"].tolist()[:200]:
            self._add_issue(
                "DAILY_OPEN_DATE_MISSING",
                "WARNING",
                "证券缺少上市日期，无法判定其行情区间是否完整",
                code=code,
                expected="open_date 非空",
                actual="空值",
                possible_causes="QMT 未返回该证券的 OpenDate，或全部是无日期哨兵",
                suggested_action="在 QMT 中补齐该证券详情后重新导出 instrument_info",
            )
        invalid = frame[
            frame["open_date"].notna()
            & frame["expire_date"].notna()
            & (frame["expire_date"] < frame["open_date"])
        ]
        for code in invalid["code"].tolist():
            self._add_issue(
                "DAILY_INVALID_LIFECYCLE_RANGE",
                "ERROR",
                "退市日早于上市日",
                code=code,
                expected="expire_date >= open_date",
                actual="open_date={0} expire_date={1}".format(
                    invalid.loc[invalid["code"] == code, "open_date"].iloc[0],
                    invalid.loc[invalid["code"] == code, "expire_date"].iloc[0],
                ),
                possible_causes="QMT 证券详情本身有误",
                suggested_action="核对该证券在 QMT 中的上市与退市日期",
            )
        if frame["expire_date"].notna().sum() == 0:
            self._add_issue(
                "DAILY_NO_DELISTED_SYMBOLS",
                "WARNING",
                "证券池里没有任何已退市证券，历史回测存在幸存者偏差",
                expected="至少部分证券带有 expire_date",
                actual=f"全部 {len(frame)} 只证券的 expire_date 均为空",
                evidence="QMT 的 instrument_info 快照只包含当前仍在证券池中的代码",
                possible_causes="大 QMT 未保留退市证券，或下载器首次运行时还没有历史快照可继承",
                suggested_action=(
                    "该问题源自数据源本身，无法在入库侧修复；做长周期回测时"
                    "需明确知道结果被幸存者偏差抬高"
                ),
            )

    def _add_issue(self, issue_code: str, level: str, message: str, **fields) -> None:
        """追加一条审计问题记录。

        参数：
            issue_code: 问题编号，日线库侧统一以 ``DAILY_`` 或 ``CROSS_5M_`` 开头。
            level: 严重级别，``ERROR``、``WARNING`` 或 ``INFO``。
            message: 面向用户的中文问题描述。
            **fields: ``AuditIssue`` 的其余可选字段，逐项含义与该数据类一致：
                ``code`` 相关证券代码、``date`` 相关交易日、``start_date`` 与
                ``end_date`` 相关区间、``field`` 涉及的字段名、``expected`` 理论值、
                ``actual`` 实际值、``evidence`` 佐证、``possible_causes`` 可能原因、
                ``suggested_action`` 处理建议、``source_file`` 与 ``source_row`` 定位信息。

        返回：
            无返回值。
        """
        # DuckDB 的 DATE 列经 pandas 取出会变成 datetime64，直接 str() 会带上
        # 一串没有意义的 00:00:00，这里统一裁成八位可读日期。
        for key in ("date", "start_date", "end_date"):
            if key in fields:
                fields[key] = _format_date(fields[key])
        self.issues.append(
            AuditIssue(
                issue_code=issue_code,
                level=level,
                dataset="bars_1d",
                message=message,
                **fields,
            )
        )


def _format_date(value) -> str:
    """把问题记录里的日期字段格式化为 ``YYYY-MM-DD``。

    参数：
        value: 日期、时间戳或已经成型的文本；空值与无法解析的文本原样返回。

    返回：
        八位可读日期文本。
    """
    if value is None or value == "":
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)


def _as_date(value) -> date:
    """把 DuckDB 返回的日期或时间戳统一成 ``date``。

    参数：
        value: ``DATE`` 或 ``TIMESTAMP`` 列取出的值。

    返回：
        对应的 ``date`` 对象。
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _resolve_date(configured: str | None, fallback) -> date:
    """把配置里的八位日期解析为 ``date``，缺省时使用库中边界。

    参数：
        configured: 配置中的 ``YYYYMMDD`` 文本；``None`` 或空串表示未设置。
        fallback: 未设置时采用的日期，来自 ``monthly_inventory``。

    返回：
        解析后的 ``date`` 对象。
    """
    if configured:
        return datetime.strptime(configured, "%Y%m%d").date()
    if isinstance(fallback, date):
        return fallback
    return pd.Timestamp(fallback).date()
