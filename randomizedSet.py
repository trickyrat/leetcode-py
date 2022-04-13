from random import choice


class RandomizedSet(object):
    def __init__(self):
        self.nums = []
        self.indices = {}

    def insert(self, val: int) -> bool:
        if val in self.indices:
            return False
        self.indices[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.indices:
            return False
        index = self.indices[val]
        self.nums[index] = self.nums[-1]
        self.indices[self.nums[index]] = index
        self.nums.pop()
        del self.indices[val]
        return True

    def getRandom(self) -> int:
        return choice(self.nums)
