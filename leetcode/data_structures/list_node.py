from typing import Optional


class ListNode(object):
    """
    Definition fro singly-linked list.
    """

    def __init__(self, val=0, next=None) -> None:
        self.val: int = val
        self.next: Optional[ListNode] = next
