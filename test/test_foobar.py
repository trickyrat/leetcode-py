import asyncio

import pytest

from src.leetcode.foobar import Foobar


class TestFooBar:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, "foobar"),
            (2, "foobarfoobar"),
        ],
    )
    async def test_foobar(self, n: int, expected: str) -> None:
        result = []
        foobar = Foobar(n)

        def print_foo() -> None:
            result.append("foo")

        def print_bar() -> None:
            result.append("bar")

        foo_task = asyncio.create_task(foobar.foo(print_foo))
        bar_task = asyncio.create_task(foobar.bar(print_bar))

        await asyncio.gather(foo_task, bar_task)

        actual = "".join(result)
        assert expected == actual
