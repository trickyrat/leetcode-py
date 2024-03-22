from typing import List
from math import inf


class StockSpanner:
    def __init__(self) -> None:
        self.stack: List = [(-1, inf)]
        self.index: int = -1

    def next(self, price: int) -> int:
        self.index += 1
        while price >= self.stack[-1][1]:
            self.stack.pop()
        self.stack.append((self.index, price))
        return self.index - self.stack[-2][0]
