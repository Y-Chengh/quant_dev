"""入库过滤原因的取值、说明文案与汇总渲染。

从 ``shards`` 拆出来的展示层：``shards`` 只负责读源 CSV、判定并写 Parquet，
「这个原因是什么意思、非 0 该怎么办、汇总怎么排版」全在这里。拆分的直接原因是
``shards`` 越过了 AGENTS.md 的 700 行阈值，更本质的原因是这两件事变化频率不同——
判定 SQL 很少动，说明文案会随着排查经验不断补。

依赖方向单向：``shards`` 导入本模块，本模块不导入 ``shards``。

日志（``sync``）与命令行摘要（``quant.cli.build_daily_store``）都走这里的
``format_filter_summary``，避免两处各写一遍格式化、改一处忘一处。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

#: 单行被丢弃的全部原因，取值与 ``ShardResult.filtered`` 的键一致。顺序即 SQL 里
#: ``CASE`` 分支的判定顺序，日志与报告一律按该顺序展示；一行只会命中第一个成立的
#: 分支，因此各原因的行数互不重叠。``kept`` 不是丢弃原因，不出现在这里。
#:
#: 前三个原因是 ``trade_date`` 字段本身的三种坏法，分开统计是为了让日志直接指出
#: 该去查源 CSV 的哪一类问题：整列缺失、位数不对、还是位数对但不是合法日期。
#:
#: 上市前的行同样分两类，判据是 ``suspend_flag``：等于 1 的是 QMT 的停牌占位填充行
#: （无害，占绝大多数），不等于 1 的是上市前出现了真实行情（要么上市日错、要么行情
#: 归属错，必须人工核对）。缺失按 0 处理，即归入需要核对的那一类，宁可多提醒。
#: 该判据对齐 ``quant.qmt_downloader.self_check`` 的 ``DATA_BEFORE_LISTING`` 与
#: ``quant.market_data.daily_check`` 的 ``DAILY_DATA_BEFORE_LISTING``：在 QMT 实际
#: 只发 0/1 的前提下三处结论一致（``daily_check`` 查的是入库后 ``round()`` 过的
#: TINYINT，理论上 0.6 这种中间值两边会分到不同桶，实务上不会出现）。改判据时
#: 三处必须同步改。
FILTER_REASONS: tuple[str, ...] = (
    "trade_date_null",
    "trade_date_bad_length",
    "trade_date_unparsable",
    "unknown_code",
    "before_listing_padding",
    "before_listing_with_data",
    "after_delisting",
)

#: 每个过滤原因保留的样例行数上限。样例只用于人工核对过滤是否合理，取小值即可。
FILTERED_SAMPLE_LIMIT = 10

_ReasonValue = TypeVar("_ReasonValue")


@dataclass(frozen=True, slots=True)
class FilterReasonInfo:
    """一个过滤原因的判定口径与处理建议，供日志和摘要直接展示。

    这些文本是给运维人看的：日志里只报一个原因名和行数，看的人还得回来翻代码
    才知道该干什么；把口径和处置写在原因旁边，排查才闭环。

    参数：
        condition: 判定条件的人话版，与 ``shards.rewrite_month_shard`` 里那条
            ``CASE`` 的对应分支逐条对应。
        meaning: 这类行是什么、为什么会出现在源数据里。
        action: 该怎么处理，含「这个原因非 0 是否正常」的判断；里面的命令按
            AGENTS.md 命令行输出约定一律写成可直接复制的 ``python -m <模块>``。
        needs_review: 行数大于 0 时是否需要人工核对。``before_listing_padding``
            是 QMT 取数方式的产物，非 0 属预期；其余六个正常都应为 0。
    """

    condition: str
    meaning: str
    action: str
    needs_review: bool


#: 每个过滤原因的说明。键必须与 ``FILTER_REASONS`` 一一对应，新增原因时同步补齐。
FILTER_REASON_INFO: dict[str, FilterReasonInfo] = {
    "trade_date_null": FilterReasonInfo(
        condition="trade_date 字段为 NULL",
        meaning="源 CSV 该行的 trade_date 是空字段，或同月某个日分区文件缺了这一列"
        "（该月全部文件都缺则读 CSV 直接报错，走不到过滤这一步）",
        action="正常应为 0。非 0 说明落盘时该字段没写出来，用 "
        "python -m quant.cli.qmt_self_check 定位坏分区后重新下载对应交易日",
        needs_review=True,
    ),
    "trade_date_bad_length": FilterReasonInfo(
        condition="trade_date 去掉首尾空格后长度不是 8 位",
        meaning="日期被写成 2024-01-02 这类带分隔符的格式，或被截断成 7 位",
        action="正常应为 0。非 0 说明落盘的日期格式变了，先用 "
        "python -m quant.cli.qmt_self_check 定位坏分区，再检查下载器的日期格式化"
        "逻辑并重新下载对应交易日",
        needs_review=True,
    ),
    "trade_date_unparsable": FilterReasonInfo(
        condition="长度是 8 位但 strptime('%Y%m%d') 解析不出来",
        meaning="20241332 这类不存在的日期，或恰好八位的乱码",
        action="正常应为 0。非 0 说明源文件被损坏，用 "
        "python -m quant.cli.qmt_self_check 定位后核对该分区 data.csv 并重新下载",
        needs_review=True,
    ),
    "unknown_code": FilterReasonInfo(
        condition="证券代码不在 instrument_info 快照里",
        meaning="日线里出现了生命周期快照没有的证券，通常是快照比日线旧，或代码写错",
        action="正常应为 0。非 0 先重跑下载器刷新 instrument_info，再执行 "
        "python -m quant.cli.build_daily_store --rebuild-all",
        needs_review=True,
    ),
    "before_listing_padding": FilterReasonInfo(
        condition="trade_date 早于上市日，且 suspend_flag = 1",
        meaning="大 QMT fill_data=True 给尚未上市的证券造的占位行，四价与量额全为 0，"
        "从取数区间最左端一路填到上市日前一天；实测约占源数据一半",
        action="预期内，必须滤除，无需处理。不滤会让库里的行数翻一倍（全库实测源 3360 万行"
        "对入库 1635 万行），并让每只证券在 IPO 当天出现由平价段跳到真实价的假跳变，"
        "直接污染波动率与收益类因子",
        needs_review=False,
    ),
    "before_listing_with_data": FilterReasonInfo(
        condition="trade_date 早于上市日，且 suspend_flag 不等于 1（缺失按 0 算）",
        meaning="上市前出现了非停牌占位的行情，说明这不是 QMT 的填充数据",
        action="正常应为 0。非 0 说明 instrument_info 的上市日不对、证券代码被复用，"
        "或行情归错了证券；先核对 QMT 的 OpenDate，确认前不要直接改数据。注意这些行"
        "本次已被滤掉，重建后 python -m quant.cli.market_check 的 "
        "DAILY_DATA_BEFORE_LISTING 必然不再报，「没报」不等于问题消失",
        needs_review=True,
    ),
    "after_delisting": FilterReasonInfo(
        condition="trade_date 晚于退市日",
        meaning="证券退市之后仍被填充出来的行",
        action="正常应为 0。非 0 说明 instrument_info 的退市日不对，或哨兵值没被识别成"
        "「无退市日」；核对 QMT 的 ExpireDate 后执行 "
        "python -m quant.cli.build_daily_store --rebuild-all",
        needs_review=True,
    ),
}

#: 原因名不在 ``FILTER_REASON_INFO`` 里时的兜底说明，避免新增原因忘了补说明就崩。
_UNKNOWN_REASON_INFO = FilterReasonInfo(
    condition="（该原因没有登记判定说明）",
    meaning="代码里新增了过滤原因但没同步补 FILTER_REASON_INFO",
    action="到 quant.market_data.daily.ingest.filter_reasons 补上这个原因的说明",
    needs_review=True,
)


def describe_reason(reason: str) -> FilterReasonInfo:
    """查一个过滤原因的判定口径与处理建议。

    参数：
        reason: 过滤原因名，正常取值见 ``FILTER_REASONS``。

    返回：
        对应的 ``FilterReasonInfo``；未登记的原因返回兜底说明而不是抛异常，
        免得日志输出因为漏补一条说明就整个失败。
    """
    return FILTER_REASON_INFO.get(reason, _UNKNOWN_REASON_INFO)


@dataclass(frozen=True, slots=True)
class FilteredSample:
    """一条被过滤掉的源行的样例，用于在日志里直接展示具体 case。

    参数：
        reason: 命中的过滤原因，取值见 ``FILTER_REASONS``。
        code: 归一化（去空格转大写）后的证券代码；源值为空时是 ``None``。
        trade_date: 源 CSV 里 ``trade_date`` 的**原样文本**，不做解析，以便
            ``trade_date_*`` 三类原因能看到真正的坏值；源值为空时是 ``None``。
        open_date: 该证券在生命周期表里的上市日，ISO 文本；缺失或代码不在表里
            时是 ``None``。
        expire_date: 该证券在生命周期表里的退市日，ISO 文本；缺失或代码不在表里
            时是 ``None``。
        suspend_flag: 源行的停牌标记，1 表示停牌；缺失时是 ``None``。
        volume: 源行的成交量，单位股；缺失时是 ``None``。
        close: 源行的收盘价，单位元，未复权；缺失时是 ``None``。
    """

    reason: str
    code: str | None
    trade_date: str | None
    open_date: str | None
    expire_date: str | None
    suspend_flag: float | None
    volume: float | None
    close: float | None

    def describe(self) -> str:
        """返回一行可直接打印进日志或摘要的样例说明。

        返回：
            形如 ``000001.SZ 20000104 open=2001-01-01 expire=- suspend=1
            volume=0 close=10.0000`` 的单行文本；缺失字段一律显示 ``-``。
        """
        return (
            f"{_sample_text(self.code)} {_sample_text(self.trade_date)}"
            f" open={_sample_text(self.open_date)}"
            f" expire={_sample_text(self.expire_date)}"
            f" suspend={_sample_number(self.suspend_flag)}"
            f" volume={_sample_number(self.volume)}"
            f" close={_sample_number(self.close, digits=4)}"
        )


def _sample_text(value: str | None) -> str:
    """把样例里的文本字段格式化为日志片段。

    参数：
        value: 原始文本；``None`` 或去空格后为空都视为缺失。

    返回：
        去掉首尾空格的原文；缺失时返回 ``-``。
    """
    if value is None:
        return "-"
    text = str(value).strip()
    return text or "-"


def _sample_number(value: float | None, digits: int = 0) -> str:
    """把样例里的数值字段格式化为日志片段。

    参数：
        value: 原始数值；``None`` 与 ``NaN`` 都视为缺失。
        digits: 保留的小数位数，0 表示按整数展示（价格类传 4）。

    返回：
        定点格式的数值文本；缺失时返回 ``-``。
    """
    if value is None:
        return "-"
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return "-"
    return f"{number:.{digits}f}"


def sort_by_reason(
    items: Iterable[tuple[str, _ReasonValue]],
) -> tuple[tuple[str, _ReasonValue], ...]:
    """按 ``FILTER_REASONS`` 的固定顺序排列 ``(原因, 值)`` 序列。

    日志、同步报告与命令行摘要都走这一个排序，保证同一次运行里各处的原因顺序
    完全一致，也保证顺序与 SQL 判定顺序对得上。

    参数：
        items: 待排序的 ``(原因, 值)`` 序列，值可以是行数、样例元组等任意类型；
            不在 ``FILTER_REASONS`` 里的原因排在末尾并按原因名升序，以免将来
            新增原因时漏排。

    返回：
        排序后的元组。
    """
    ranks = {reason: index for index, reason in enumerate(FILTER_REASONS)}
    return tuple(
        sorted(items, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))
    )


def fill_missing_reasons(
    counts: Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    """把稀疏的过滤计数补齐成 ``FILTER_REASONS`` 全集。

    只在**同步结束的总汇总**处用。分片级统计刻意保持稀疏：分片行一个月只打一行，
    把 0 铺开不会多出行数，却会让每行从约 38 字符涨到约 170 字符，而其中绝大多数
    是 ``=0``，终端折行后反而更难找异常月份。总汇总只出现一次，把 0 显式写出来
    才能区分「查过了，一行都没有」和「压根没跑这条判据」——后者正是没列 0 时读
    日志的人会产生的误解。

    参数：
        counts: 实际发生过的 ``(原因, 行数)``；缺席的原因视为 0 行。不在
            ``FILTER_REASONS`` 里的原因会原样保留，不会被丢掉。

    返回：
        按 ``FILTER_REASONS`` 排序、含全部原因的 ``((原因, 行数), ...)``。
    """
    filled = dict.fromkeys(FILTER_REASONS, 0)
    for reason, count in counts:
        filled[reason] = filled.get(reason, 0) + count
    return sort_by_reason(filled.items())


def format_filter_summary(
    totals: Iterable[tuple[str, int]],
    samples: Iterable[tuple[str, tuple[FilteredSample, ...]]] = (),
) -> tuple[str, ...]:
    """把过滤汇总渲染成日志与命令行摘要共用的文本行。

    两处共用同一个渲染函数，是为了杜绝「日志说一套、摘要说另一套」——此前两边各写
    一遍格式化，改一处忘一处就会不一致。返回的每行自带缩进，调用方只负责逐行输出
    （日志侧会再给每行加自己的 ``[daily-sync]`` 前缀，因此两边是内容一致而非逐字
    一致）。

    参数：
        totals: ``((原因, 行数), ...)``，通常来自 ``SyncReport.filtered_totals``，
            已经按 ``FILTER_REASONS`` 排好序并补过 0；为空表示本次没跑过过滤，
            此时整节都不输出。
        samples: ``((原因, (样例, ...)), ...)``，通常来自
            ``SyncReport.filtered_samples``；只含计数非 0 的原因，缺省表示不带样例。
            出现在 ``samples`` 里但不在 ``totals`` 里的原因会被忽略。

    返回：
        可直接逐行打印的文本元组；``totals`` 为空时返回空元组。最后一行是本次的
        结论：要么点名需要人工核对的原因，要么明说全部落在预期内。
    """
    ordered = tuple(totals)
    if not ordered:
        return ()
    samples_by_reason = dict(samples)
    lines = ["本次同步累计过滤（按原因）:"]
    flagged: list[str] = []
    for reason, count in ordered:
        info = describe_reason(reason)
        lines.append(f"  {reason}: {count} 行")
        lines.append(f"    判定: {info.condition}")
        lines.append(f"    含义: {info.meaning}")
        lines.append(f"    处理: {info.action}")
        picked = samples_by_reason.get(reason, ())
        if picked:
            lines.append(f"    随机样例 {len(picked)} 条:")
            lines.extend(f"      {sample.describe()}" for sample in picked)
        if count > 0 and info.needs_review:
            flagged.append(f"{reason}({count} 行)")
    if flagged:
        lines.append("需人工核对: " + "、".join(flagged))
    else:
        lines.append("本次过滤全部落在预期原因内，无需人工核对。")
    return tuple(lines)
