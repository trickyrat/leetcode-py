import asyncio
import threading

import pytest

from src.leetcode.foobar import Foobar


class TestFooBar:
    @pytest.fixture
    def print_functions(self):
        result = []

        def print_foo() -> None:
            thread_id = threading.current_thread().ident
            task_name = asyncio.current_task().get_name()
            print(f"print_foo running in thread {thread_id}, task name {task_name}")
            result.append("foo")

        def print_bar() -> None:
            thread_id = threading.current_thread().ident
            task_name = asyncio.current_task().get_name()
            print(f"print_foo running in thread {thread_id}, task name {task_name}")
            result.append("bar")

        return print_foo, print_bar, result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "n, expected",
        [
            (1, "foobar"),
            (2, "foobarfoobar"),
        ],
    )
    async def test_foobar(self, n: int, expected: str, print_functions) -> None:
        print_foo, print_bar, result = print_functions
        foobar = Foobar(n)

        foo_task = asyncio.create_task(foobar.foo(print_foo))
        bar_task = asyncio.create_task(foobar.bar(print_bar))

        await asyncio.gather(foo_task, bar_task)

        assert "".join(result) == expected
        result.clear()
