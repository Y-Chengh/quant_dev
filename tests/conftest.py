"""pytest 全局夹具。

测试禁止读写用户真实的行情库与配置目录，因此这里默认清掉相关环境变量，
让被测代码回落到内置默认值或用例显式提供的临时路径。
需要验证环境变量本身的用例应自行用 ``patch.dict`` 覆盖。
"""

from __future__ import annotations

import os

import pytest

from quant.config import MARKET_DATABASE_ENV, PROJECT_ROOT_ENV


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """在每个用例执行前移除会影响路径解析的环境变量。

    参数：
        monkeypatch: pytest 内置夹具，用于在用例结束后自动还原环境变量。

    返回：
        无返回值；仅产生环境隔离副作用。
    """
    for name in (MARKET_DATABASE_ENV, PROJECT_ROOT_ENV):
        if name in os.environ:
            monkeypatch.delenv(name, raising=False)
