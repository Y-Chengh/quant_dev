from __future__ import annotations

import re
import unittest
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "market_service" / "static" / "app.js"


class MarketServiceFrontendTest(unittest.TestCase):
    """验证行情网页查询配置记忆和K线默认展示范围。"""

    @classmethod
    def setUpClass(cls) -> None:
        """读取一次网页脚本，供当前测试类检查发布资产。"""
        cls.script = APP_JS.read_text(encoding="utf-8")

    def test_saved_query_config_covers_chart_and_raw_filters(self) -> None:
        """持久化白名单应覆盖K线条件、周期和全部原始行情筛选字段。"""
        expected_chart_fields = {"chartCode", "chartStart", "chartEnd"}
        expected_raw_fields = {
            "rawCode", "rawDate",
            "startTime", "endTime", "minClose", "maxClose", "minVolume",
            "maxVolume", "minAmount", "maxAmount",
        }
        chart_declaration = re.search(
            r"const chartFieldIds=\[(?P<fields>[^]]+)\]", self.script
        )
        raw_declaration = re.search(
            r"const rawFieldIds=\[(?P<fields>[^]]+)\]", self.script
        )

        self.assertIsNotNone(chart_declaration)
        self.assertIsNotNone(raw_declaration)
        self.assertEqual(
            set(re.findall(r"'([^']+)'", chart_declaration.group("fields"))),
            expected_chart_fields,
        )
        self.assertEqual(
            set(re.findall(r"'([^']+)'", raw_declaration.group("fields"))),
            expected_raw_fields,
        )
        self.assertIn("marketService.queryConfig.v1", self.script)
        self.assertIn("window.localStorage.setItem", self.script)
        self.assertIn("window.localStorage.getItem", self.script)
        self.assertRegex(self.script, r"next\.chart=\{\.\.\.collectFieldValues\(chartFieldIds\),period\}")
        self.assertIn("selectPeriod(config.chart.period)", self.script)
        self.assertIn("activateTab(config.activeTab)", self.script)
        self.assertIn(
            "Object.prototype.hasOwnProperty.call(periodLabels,value)",
            self.script,
        )

    def test_restored_dates_are_not_overwritten_by_metadata_defaults(self) -> None:
        """有效恢复值或元数据返回前的用户输入均应阻止默认日期覆盖。"""
        self.assertIn(
            "input.value===values[id]&&(!requiredDateIds.has(id)||input.value)",
            self.script,
        )
        self.assertIn(
            "protectDateFromMetadataDefault(event){restoredFieldIds.add(event.target.id)}",
            self.script,
        )
        self.assertIn(
            "['chartStart','chartEnd','rawDate'].forEach(id=>$(id).addEventListener('input',protectDateFromMetadataDefault))",
            self.script,
        )
        for field in ("chartStart", "chartEnd", "rawDate"):
            self.assertIn(f"if(!restoredFieldIds.has('{field}'))", self.script)

    def test_chart_uses_full_range_and_refreshes_after_date_change(self) -> None:
        """每次K线结果应展示全区间，两个日期控件变化后均触发防抖查询。"""
        zoom_ranges = re.findall(
            r"type:'(?:inside|slider)'[^}]*?start:(\d+),end:(\d+)",
            self.script,
        )

        self.assertEqual(zoom_ranges, [("0", "100"), ("0", "100")])
        self.assertIn(
            "['chartStart','chartEnd'].forEach(id=>$(id).addEventListener('change',scheduleChartLoad))",
            self.script,
        )
        schedule_body = re.search(
            r"function scheduleChartLoad\(\)\{(?P<body>.*)", self.script
        )
        self.assertIsNotNone(schedule_body)
        self.assertLess(
            schedule_body.group("body").index("++chartRequestVersion"),
            schedule_body.group("body").index("setTimeout(loadChart,300)"),
        )
        self.assertIn("chartDateTimer=setTimeout(loadChart,300)", self.script)
        self.assertIn("requestPeriod=period", self.script)
        self.assertIn("period:requestPeriod", self.script)
        self.assertIn("periodLabels[requestPeriod]", self.script)
        self.assertIn("if(requestVersion!==chartRequestVersion)return", self.script)
        self.assertRegex(
            self.script,
            r"periods button'\)\.forEach\(btn=>btn\.onclick=.*\+\+chartRequestVersion",
        )


if __name__ == "__main__":
    unittest.main()
