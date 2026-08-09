from __future__ import annotations


def normalize_security_code(code: str) -> str:
    """规范证券代码，并为沪深市场的六位裸代码补充交易所后缀。

    参数：
        code: 用户输入的证券代码；允许首尾空白和小写交易所后缀。六位数字代码
            按首位补全交易所，其中 ``5``、``6``、``9`` 补 ``.SH``，
            ``0``、``1``、``2``、``3`` 补 ``.SZ``。

    返回：
        去除首尾空白并转为大写的代码；无法判定交易所时保留无后缀形式，
        由后续查询按原有规则返回空结果。
    """
    normalized = code.upper().strip()
    if len(normalized) != 6 or not normalized.isascii() or not normalized.isdigit():
        return normalized
    if normalized[0] in "569":
        return f"{normalized}.SH"
    if normalized[0] in "0123":
        return f"{normalized}.SZ"
    return normalized
