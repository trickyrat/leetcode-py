from solution import ListNode
from typing import List


def initListNode(nodes: List[int]) -> ListNode:
    head = ListNode(0)
    dummy = head
    for node in nodes:
        dummy = ListNode(node)
        dummy = dummy.next
    return head.next


def printListNode(head: ListNode) -> str:
    res = ""
    while head is not None:
        res += head.val
        if head is not None:
            res += "->"
    return res
