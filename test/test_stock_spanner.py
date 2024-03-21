from typing import List
import pytest
from src.leetcode.stock_spanner import StockSpanner


class TestStockSpanner:
    @pytest.mark.parametrize(
        "prices, expected",
        [
            (
                [
                    100,
                    80,
                    60,
                    70,
                    60,
                    75,
                    85,
                ],
                [1, 1, 1, 2, 1, 4, 6],
            ),
        ],
    )
    def test_stock_spanner(self, prices: List[int], expected: List[int]):
        stock_spanner = StockSpanner()
        actual = []
        for price in prices:
            actual.append(stock_spanner.next(price))
        assert expected == actual
