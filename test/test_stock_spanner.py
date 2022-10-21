import pytest
from leetcode.stock_spanner import StockSpanner


class TestStockSpanner:
    stock_spanner = StockSpanner()

    @pytest.mark.parametrize(
        "price, expected",
        [
            (100, 1),
            (80, 1),
            (60, 1),
            (70, 1),
            (60, 1),
            (75, 1),
            (85, 1),
        ],
    )
    def test_stock_spanner(self, price: int, expected: int):
        actual = self.stock_spanner.next(price)
        assert expected == actual
