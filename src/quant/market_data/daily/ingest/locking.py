"""日线库的进程间同步锁与 DuckDB 写锁降级。

DuckDB 是单写多读：一个活跃的读写连接会同时挡住其它进程的读连接。因此入库流程
把重活全部放在内存连接上，只在最后用一个极短的读写事务落库；自动增量同步遇到
写锁时必须降级为跳过，而不是让整个实验挂掉。
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import duckdb

#: 同步锁文件名，位于日线库数据根目录下。
LOCK_FILENAME = ".sync.lock"

#: 锁文件超过该秒数视为上一个进程异常退出留下的残留锁。
DEFAULT_LOCK_TIMEOUT_SECONDS = 3600.0


class DailyStoreLockedError(RuntimeError):
    """日线库正被其它进程写入时抛出。

    自动增量同步应当捕获它并降级为跳过；显式的构建命令则应当把它作为错误上报。
    """


@contextmanager
def sync_lock(daily_root, *, timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS):
    """获取日线库的独占同步锁。

    先于 DuckDB 自身的文件锁生效，好处是能给同一套工具链的其它进程一个明确的
    中文提示，而不是让它们撞上 DuckDB 的 ``Conflicting lock`` 原始报错。

    参数：
        daily_root: 日线库数据根目录；锁文件创建在该目录下的 ``.sync.lock``。
        timeout_seconds: 残留锁的判定秒数。锁文件的修改时间早于该秒数时视为
            上一个进程异常退出，直接夺锁；传 0 表示任何已存在的锁都算残留。

    返回：
        上下文管理器；进入时返回锁文件路径，退出时删除锁文件。
        锁被其它活跃进程持有时抛出 ``DailyStoreLockedError``。
    """
    root = Path(daily_root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_FILENAME
    descriptor = _acquire(lock_path, timeout_seconds)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                handle,
                ensure_ascii=False,
            )
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


def _acquire(lock_path: Path, timeout_seconds: float) -> int:
    """尝试独占创建锁文件，必要时清理残留锁后重试一次。

    参数：
        lock_path: 锁文件路径。
        timeout_seconds: 残留锁的判定秒数。

    返回：
        已打开的文件描述符；锁被活跃进程持有时抛出 ``DailyStoreLockedError``。
    """
    try:
        return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    if not _is_stale(lock_path, timeout_seconds):
        raise DailyStoreLockedError(
            f"日线库同步锁被占用: {lock_path}；请等待正在运行的同步结束，或删除该锁文件"
        )
    try:
        lock_path.unlink()
    except OSError:
        pass
    try:
        return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise DailyStoreLockedError(
            f"日线库同步锁被占用: {lock_path}；另一个进程刚刚抢到锁"
        )


def _is_stale(lock_path: Path, timeout_seconds: float) -> bool:
    """判断锁文件是否是上一个进程异常退出留下的残留。

    参数：
        lock_path: 锁文件路径。
        timeout_seconds: 残留判定秒数；小于等于 0 时任何已存在的锁都算残留。

    返回：
        锁文件不存在、无法读取状态，或修改时间早于超时阈值时返回 ``True``。
    """
    if timeout_seconds <= 0:
        return True
    try:
        age = datetime.now().timestamp() - lock_path.stat().st_mtime
    except OSError:
        return True
    return age > timeout_seconds


@contextmanager
def open_catalog(database_path, *, read_only: bool):
    """打开日线库目录连接，并把 DuckDB 的文件锁冲突翻译成明确异常。

    参数：
        database_path: ``qmt_daily.duckdb`` 文件路径；读写模式下会自动创建父目录。
        read_only: ``True`` 以只读方式打开（可与其它读进程共存），
            ``False`` 以读写方式打开（与任何其它连接互斥）。

    返回：
        上下文管理器，进入时返回 DuckDB 连接，退出时确保关闭。
        文件被其它进程锁定时抛出 ``DailyStoreLockedError``。
    """
    path = Path(database_path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError(
            f"日线库不存在: {path}；"
            "请先运行 python -m quant.cli.build_daily_store"
        )
    try:
        connection = duckdb.connect(str(path), read_only=read_only)
    except duckdb.Error as error:
        if _is_lock_conflict(error):
            raise DailyStoreLockedError(
                f"日线库 {path} 正被其它进程占用: {error}"
            )
        raise
    try:
        connection.execute("SET threads = 4")
        yield connection
    finally:
        connection.close()


#: DuckDB 表达「这个库文件已经被别人以不兼容的方式打开」的几种消息特征。
#: 跨进程占用报 ``Conflicting lock``；**同一进程内**已经存在一个只读连接时，
#: 再以读写方式打开同一路径会报 ``different configuration``——两者都应该降级，
#: 而不是让调用方崩溃。
_LOCK_CONFLICT_MARKERS = (
    "conflicting lock",
    "could not set lock",
    "different configuration",
    "already open",
)


def _is_lock_conflict(error: BaseException) -> bool:
    """判断一个 DuckDB 异常是否由文件被占用引起。

    参数：
        error: ``duckdb.connect`` 抛出的异常对象。

    返回：
        异常消息命中任一占用特征时返回 ``True``。
    """
    message = str(error).lower()
    return any(marker in message for marker in _LOCK_CONFLICT_MARKERS)
