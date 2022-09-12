from leetcode.data_structures.list_node import ListNode
from leetcode.data_structures.tree_node import TreeNode
from typing import List, Optional
from collections import deque


def create_list_node(nodes: List[int]) -> Optional[ListNode]:
    """Create a linked-list with a list"""
    head = ListNode(0)
    dummy = head
    for node in nodes:
        dummy.next = ListNode(node)
        dummy = dummy.next
    return head.next


def print_list_node(head: ListNode) -> str:
    """Convert a linked-list to string"""
    res = ""
    while head is not None:
        res += head.val
        if head is not None:
            res += "->"
    return res


def list_node_to_list(head: ListNode) -> List[int]:
    """Convert a linked-list to a list"""
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


def create_treenode(nums: List[int | None]) -> Optional[TreeNode]:
    """Create Binary Tree Node"""
    n = len(nums)
    if n == 0 or nums[0] is None:
        return None
    root = TreeNode(nums[0])
    q = deque()
    q.appendleft(root)
    cursor = 1
    while cursor < n:
        node = q.pop()
        if cursor > n - 1 or nums[cursor] is None:
            node.left = None
        else:
            left_node = TreeNode(nums[cursor])
            if left_node:
                node.left = left_node
            q.appendleft(left_node)
        if cursor + 1 > n - 1 or nums[cursor + 1] is None:
            node.right = None
        else:
            right_node = TreeNode(nums[cursor + 1])
            if right_node:
                node.right = right_node
            q.appendleft(right_node)
        cursor += 2
    return root


def preorder_traversal(root: TreeNode) -> List[int]:
    """Traversal a binary tree node with preoder"""
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


# def is_same_tree(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
#     """Judges if two binary tree nodes are equal"""
#     if not root1 and not root2:
#         return True
#     if not root1 or not root2:
#         return False
#
#     queue1 = deque([root1])
#     queue2 = deque([root2])
#
#     while queue1 and queue2:
#         node1 = queue1.popleft()
#         node2 = queue2.popleft()
#         if node1.val != node2.val:
#             return False
#         left1, right1 = node1.left, node1.right
#         left2, right2 = node2.left, node2.right
#         if (not left1) ^ (not left2):
#             return False
#         if (not right1) ^ (not right2):
#             return False
#         if left1:
#             queue1.append(left1)
#         if right1:
#             queue1.append(right1)
#         if left2:
#             queue2.append(left2)
#         if right2:
#             queue2.append(right2)
#
#     return not queue1 and not queue2
