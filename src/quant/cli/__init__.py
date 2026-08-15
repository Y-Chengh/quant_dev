"""命令行入口集合。

每个模块对应一个可执行命令，只负责参数解析、配置装载、日志初始化和调用库层，
不承载可复用的业务逻辑；可复用逻辑应下沉到 ``quant.factor_research``、
``quant.market_data`` 或 ``quant.qmt_downloader``。

本模块不导入任何子模块，避免使用某一个命令时被迫加载其余命令的三方依赖。
"""

from __future__ import annotations

__all__: list[str] = []
