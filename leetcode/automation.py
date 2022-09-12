INT_MAX = 2 ** 31 - 1
INT_MIN = -2 ** 31


class Automation:
    def __init__(self):
        self.state: str = "start"
        self.sign: int = 1
        self.res: int = 0
        self.table: dict = {
            "start": ["start", "signed", "in_number", "end"],
            "signed": ["end", "end", "in_number", "end"],
            "in_number": ["end", "end", "in_number", "end"],
            "end": ["end", "end", "end", "end"],
        }

    def _get_col(self, c: str) -> int:
        if c.isspace():
            return 0
        if c == "+" or c == "-":
            return 1
        if c.isdigit():
            return 2
        return 3

    def get(self, c: str) -> int:
        self.state = self.table[self.state][self._get_col(c)]
        if self.state == "in_number":
            self.res = self.res * 10 + int(c)
            self.res = min(self.res, INT_MAX) if self.sign == 1 else min(self.res, -INT_MIN)
        elif self.state == "signed":
            self.sign = 1 if c == "+" else -1
