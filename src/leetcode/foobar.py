from typing import Callable
import asyncio


class Foobar:
    def __init__(self, n: int) -> None:
        self.n = n
        self.foo_lock = asyncio.Semaphore(1)
        self.bar_lock = asyncio.Semaphore(0)

    async def foo(self, print_foo: "Callable[[], None]") -> None:
        for i in range(self.n):
            await self.foo_lock.acquire()
            print_foo()
            self.bar_lock.release()

    async def bar(self, print_bar: "Callable[[], None]") -> None:
        for i in range(self.n):
            await self.bar_lock.acquire()
            print_bar()
            self.foo_lock.release()
