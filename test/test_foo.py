import asyncio
from typing import List

import pytest

from src.leetcode.foo import Foo


class TestFoo:
    @pytest.fixture
    def print_functions(self):
        result = []

        def print_first() -> None:
            result.append("first")

        def print_second() -> None:
            result.append("second")

        def print_third() -> None:
            result.append("third")

        return print_first, print_second, print_third, result


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
    async def test_foo(self, nums: List[int], expected: str, print_functions) -> None:
        print_first, print_second, print_third, result = print_functions
        foo = Foo()

        task_mapping = {
            1: lambda: foo.first(print_first),
            2: lambda: foo.second(print_second),
            3: lambda: foo.third(print_third),
        }

        tasks = [asyncio.create_task(task_mapping[num]()) for num in nums]
        await asyncio.gather(*tasks)

        assert "".join(result) == expected
        result.clear()
