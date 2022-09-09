from leetcode.data_structures.list_node import ListNode
from leetcode.data_structures.tree_node import TreeNode
from typing import List, Optional
import queue


def create_list_node(nodes: List[int]) -> Optional[ListNode]:
    head = ListNode(0)
    dummy = head
    for node in nodes:
        dummy.next = ListNode(node)
        dummy = dummy.next
    return head.next


def print_list_node(head: ListNode) -> str:
    res = ""
    while head is not None:
        res += head.val
        if head is not None:
            res += "->"
    return res


def list_node_to_list(head: ListNode) -> List[int]:
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


def create_treenode(nums: List[int | None]) -> Optional[TreeNode]:
    """Create Binary Tree iteratively"""
    n = len(nums)
    if n == 0 or nums[0] is None:
        return None
    root = TreeNode(nums[0])
    q = queue.Queue()
    q.put(root)
    cursor = 1
    while cursor < n:
        node = q.get()
        if cursor > n - 1 or nums[cursor] is None:
            node.left = None
        else:
            left_node = TreeNode(nums[cursor])
            if left_node:
                node.left = left_node
            q.put(left_node)
        if cursor + 1 > n - 1 or nums[cursor + 1] is None:
            node.right = None
        else:
            right_node = TreeNode(nums[cursor + 1])
            if right_node:
                node.right = right_node
            q.put(right_node)
        cursor += 2
    return root


def preorder_traversal(root: TreeNode) -> List[int]:
    res = list()
    if not root:
        return res
    stack = []
    node = root
    while stack or node:
        while node:
            res.append(node.val)
            stack.append(node)
            node = node.left
        node = stack.pop()
        node = node.right
    return res
