from threading import Lock


class Foo:
    def __init__(self):
        self.firstJobDone = Lock()
        self.secondJobDone = Lock()
        self.firstJobDone.acquire()
        self.secondJobDone.acquire()

    def first(self, print_first: 'Callable[[], None]') -> None:
        print_first()
        self.firstJobDone.release()

    def second(self, print_second: 'Callable[[], None]') -> None:
        with self.firstJobDone:
            print_second()
            self.secondJobDone.release()

    def third(self, print_third: 'Callable[[], None]') -> None:
        with self.secondJobDone:
            print_third()
