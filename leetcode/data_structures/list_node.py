from typing import Optional


class ListNode(object):
    """
    Definition fro singly-linked list.
    """

    def __init__(self, val: int = 0, next: Optional["ListNode"] = None) -> None:
        self.val: int = val
        self.next: Optional[ListNode] = next

    def __eq__(self, other):
        dummy_head1 = self
        dummy_head2 = other
        while dummy_head1 is not None and dummy_head2 is not None and dummy_head1.val == dummy_head2.val:
            dummy_head1 = dummy_head1.next
            dummy_head2 = dummy_head2.next

        return True if dummy_head1 is None and dummy_head2 is None else False

    def __lt__(self, other):
        return True if self.val < other.val else False
