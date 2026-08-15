"""日频因子表达式的执行上下文。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .expression import ExpressionNamespace, ExpressionNode, FactorExpression
from .registry import get_operator


class DailyFactorFrame(ExpressionNamespace):
    """包装日频表，并确保所有时序运算排序正确且最终对齐原始索引。"""

    REQUIRED_COLUMNS = {"code", "trade_date"}

    def __init__(self, daily: pd.DataFrame):
        """复制并规范化日频表，建立稳定排序、原序恢复和节点缓存。

        参数：
            daily: 每行唯一对应一个证券交易日的日频数据表。
        """

        missing = self.REQUIRED_COLUMNS.difference(daily.columns)
        if missing:
            raise ValueError(f"日频数据缺少键列: {sorted(missing)}")
        normalized = daily.copy()
        normalized["code"] = normalized["code"].astype(str)
        normalized["trade_date"] = pd.to_datetime(
            normalized["trade_date"], errors="raise"
        ).dt.normalize()
        # 唯一性必须在规范化之后检查：例如整数代码 1 与字符串代码 "1"，或同一
        # 自然日的不同时刻，在原始值上不同，但都会映射到同一个日频主键。若允许
        # 这种碰撞继续计算，会把同一证券同一天错误地当成两条时序观测。
        if normalized.duplicated(["code", "trade_date"]).any():
            raise ValueError("日频数据规范化后存在重复的 (code, trade_date)")

        # 时序算子必须先按证券和日期排序。保存位置映射而不是依赖标签索引，因而
        # 即使调用方索引重复，也能逐行恢复到完全相同的原始顺序。
        positions = np.arange(len(normalized))
        order = (
            normalized.assign(_factor_position=positions)
            .sort_values(["code", "trade_date", "_factor_position"], kind="stable")[
                "_factor_position"
            ]
            .to_numpy(dtype=int)
        )
        self._ordered = normalized.iloc[order].reset_index(drop=True)
        self._order = order
        self._original_index = daily.index.copy()
        self._cache: dict[ExpressionNode, pd.Series] = {}
        super().__init__(frame=self)

    @property
    def daily(self) -> pd.DataFrame:
        """返回内部规范化日频数据的副本，避免调用方修改执行上下文。"""

        restored = self._restore_frame_order(self._ordered)
        restored.index = self._original_index
        return restored

    def bind(self, node: ExpressionNode) -> FactorExpression:
        """把搜索生成的纯符号节点绑定到当前日频执行上下文。

        参数：
            node: 不携带实际数据的不可变因子表达式节点。
        """

        return FactorExpression(node=node, frame=self)

    def evaluate(
        self,
        expression: FactorExpression | ExpressionNode,
        name: str | None = None,
    ) -> pd.Series:
        """计算一个因果表达式，并按调用方原始行顺序和索引返回结果。

        参数：
            expression: 要执行的已绑定表达式或纯符号节点。
            name: 返回序列名；缺省时使用稳定因子 ID。
        """

        node = expression.node if isinstance(expression, FactorExpression) else expression
        if isinstance(expression, FactorExpression) and expression.frame not in {None, self}:
            raise ValueError("表达式绑定到了另一个 DailyFactorFrame")
        if not node.causal:
            raise ValueError("拒绝执行包含非因果算子的表达式")
        ordered_values = self._evaluate_node(node)
        result = np.empty(len(self._ordered), dtype=float)
        result[self._order] = pd.to_numeric(ordered_values, errors="coerce").to_numpy(
            dtype=float
        )
        return pd.Series(result, index=self._original_index, name=name or node.factor_id)

    def clear_cache(self) -> None:
        """清理表达式节点缓存；批量搜索可在批次之间释放中间数组。"""

        self._cache.clear()

    def _evaluate_node(self, node: ExpressionNode) -> pd.Series:
        """递归计算单个节点，并复用当前批次已得到的公共子表达式。

        参数：
            node: 当前要计算或从缓存读取的表达式节点。
        """

        cached = self._cache.get(node)
        if cached is not None:
            return cached

        parameters = node.parameter_map
        if node.operator == "column":
            column = str(parameters["name"])
            if column not in self._ordered.columns:
                raise ValueError(f"日频数据缺少表达式列: {column!r}")
            values = pd.to_numeric(self._ordered[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        elif node.operator == "constant":
            values = pd.Series(
                float(parameters["value"]), index=self._ordered.index, dtype=float
            )
        else:
            definition = get_operator(node.operator)
            if len(node.inputs) != definition.arity:
                raise ValueError(
                    f"算子 {node.operator!r} 需要 {definition.arity} 个输入，"
                    f"实际为 {len(node.inputs)}"
                )
            # ExpressionNode 构造器是公开的；执行时再次校验参数，避免调用方绕过
            # operation_node 手工创建不合法或非规范节点。
            normalized = definition.normalize_parameters(parameters)
            if normalized != parameters:
                raise ValueError(
                    f"算子 {node.operator!r} 的参数尚未规范化，请使用 operation_node"
                )
            inputs = tuple(self._evaluate_node(child) for child in node.inputs)
            values = definition.evaluator(self._ordered, inputs, parameters)

        values = pd.Series(values, index=self._ordered.index)
        if len(values) != len(self._ordered):
            raise ValueError(
                f"算子 {node.operator!r} 返回错误行数: "
                f"expected={len(self._ordered)} actual={len(values)}"
            )
        self._cache[node] = values
        return values

    def _restore_frame_order(self, frame: pd.DataFrame) -> pd.DataFrame:
        """使用构造时保存的位置映射把内部排序表恢复为调用方行顺序。

        参数：
            frame: 当前按内部证券与日期顺序排列的表。
        """

        result = frame.iloc[np.argsort(self._order)].copy()
        return result.reset_index(drop=True)
