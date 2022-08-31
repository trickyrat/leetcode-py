from random import randrange

from solution import ListNode
from typing import Optional


class RandomNode(object):
    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def get_random(self) -> int:
        curr, i, ans = self.head, 1, 0
        while curr:
            if randrange(i) == 0:
                ans = curr.val
            i += 1
            curr = curr.next
        return ans
