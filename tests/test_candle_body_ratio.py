from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_research.factor_factories import FACTOR_FACTORIES


class CandleBodyRatioTest(unittest.TestCase):
    def test_exact_values_preserve_index_and_do_not_mix_securities(self):
        daily = pd.DataFrame(
            {
                "code": ["A", "B", "A"],
                "open": [10.0, 20.0, 11.0],
                "high": [13.0, 22.0, 12.0],
                "low": [9.0, 18.0, 10.0],
                "close": [12.0, 18.0, 11.0],
            },
            index=[7, 3, 11],
        )

        values = FACTOR_FACTORIES["candle_body_ratio"].compute(
            pd.DataFrame(), daily
        )

        pd.testing.assert_index_equal(values.index, daily.index)
        np.testing.assert_allclose(values.to_numpy(), [0.5, 0.5, 0.0])

    def test_zero_or_invalid_range_and_missing_prices_return_nan(self):
        daily = pd.DataFrame(
            {
                "code": ["A", "B", "C", "D", "E", "F"],
                "open": [10.0, 10.0, np.nan, 10.0, 10.0, 10.0],
                "high": [10.0, 9.0, 12.0, np.nan, 12.0, 12.0],
                "low": [10.0, 11.0, 10.0, 9.0, np.nan, 9.0],
                "close": [10.0, 10.0, 11.0, 11.0, 11.0, np.nan],
            }
        )

        values = FACTOR_FACTORIES["candle_body_ratio"].compute(
            pd.DataFrame(), daily
        )

        self.assertTrue(values.isna().all())


if __name__ == "__main__":
    unittest.main()
