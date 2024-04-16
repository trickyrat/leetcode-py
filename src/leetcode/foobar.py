from threading import Lock
from typing import Callable


class Foobar:
    def __init__(self, n: int) -> None:
        self.n = n
        self.FooLock = Lock()
        self.BarLock = Lock()
        self.BarLock.acquire()

    def foo(self, print_foo: "Callable[[], None]") -> None:
        for i in range(self.n):
            self.FooLock.acquire()
            print_foo()
            self.BarLock.release()

    def bar(self, print_bar: "Callable[[], None]") -> None:
        for i in range(self.n):
            self.BarLock.acquire()
            print_bar()
            self.FooLock.release()
