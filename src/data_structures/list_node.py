from typing import Optional


class ListNode(object):
    """
    Definition for singly-linked list.
    """

    def __init__(self, val: int = 0, _next: Optional["ListNode"] = None) -> None:
        self.val: int = val
        self.next: Optional[ListNode] = _next

    def __lt__(self, other):
        return True if self.val < other.val else False
    
