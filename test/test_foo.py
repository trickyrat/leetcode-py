import asyncio
from typing import List

import pytest

from src.leetcode.foo import Foo


class TestFoo:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([1, 2, 3], "firstsecondthird"),
            ([1, 3, 2], "firstsecondthird"),
            ([2, 1, 3], "firstsecondthird"),
            ([2, 3, 1], "firstsecondthird"),
            ([3, 1, 2], "firstsecondthird"),
            ([3, 2, 1], "firstsecondthird"),
        ],
    )
    async def test_foo(self, nums: List[int], expected: str) -> None:
        result = []

        def print_first() -> None:
            result.append("first")

        def print_second() -> None:
            result.append("second")

        def print_third() -> None:
            result.append("third")

        foo = Foo()
        tasks = []
        for num in nums:
            if num == 1:
                tasks.append(asyncio.create_task(foo.first(print_first)))
            elif num == 2:
                tasks.append(asyncio.create_task(foo.second(print_second)))
            else:
                tasks.append(asyncio.create_task(foo.third(print_third)))

        await asyncio.gather(*tasks)

        actual = "".join(result)
        assert expected == actual
