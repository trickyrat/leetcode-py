from typing import Callable
from asyncio import Semaphore


class Foo:
    def __init__(self):
        self.first_job_done_semaphore = Semaphore(0)
        self.second_job_done_semaphore = Semaphore(0)

    async def first(self, print_first: "Callable[[], None]") -> None:
        print_first()
        self.first_job_done_semaphore.release()

    async def second(self, print_second: "Callable[[], None]") -> None:
        await self.first_job_done_semaphore.acquire()
        print_second()
        self.second_job_done_semaphore.release()

    async def third(self, print_third: "Callable[[], None]") -> None:
        await self.second_job_done_semaphore.acquire()
        print_third()
