"""本地行情服务启动命令。

服务只监听回环地址，不对局域网或公网暴露。
"""

from __future__ import annotations

import argparse

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
APP_IMPORT_PATH = "quant.market_data.app:app"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析行情服务启动参数。

    参数：
        argv: 命令行参数列表，不含程序名；``None`` 表示读取 ``sys.argv``。

    返回：
        含 ``host``、``port`` 和 ``reload`` 三个字段的命名空间。
    """
    parser = argparse.ArgumentParser(description="启动本地行情服务")
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="监听地址，默认 %s；请勿改为 0.0.0.0 以免暴露到网络。" % DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="监听端口，默认 %d。" % DEFAULT_PORT,
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开启代码热重载，仅用于本地开发调试。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """按命令行参数启动 uvicorn 服务。

    参数：
        argv: 命令行参数列表，不含程序名；``None`` 表示读取 ``sys.argv``。

    返回：
        无返回值；服务退出后函数才返回。
    """
    args = parse_args(argv)
    uvicorn.run(APP_IMPORT_PATH, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
