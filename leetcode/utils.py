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


def create_treenode_iteratively(nums: List[int | None]) -> Optional[TreeNode]:
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


def create_treenode_recursively(nums: List[int | None]) -> Optional[TreeNode]:
    def create_treenode(data: List[int | None], index: int) -> Optional[TreeNode]:
        if index >= len(data) or data[index] is None:
            return None
        root = TreeNode(data[index])
        root.left = create_treenode(data, 2 * index + 1)
        root.right = create_treenode(data, 2 * index + 2)
        return root

    return create_treenode(nums, 0)


def create_treenode_with_dfs(data: str) -> Optional[TreeNode]:
    str_list = data.split(",")

    def dfs(data_list: List[str]):
        if len(data_list) == 0:
            return None
        if data_list[0] == "null":
            data_list.pop(0)
            return None
        root = TreeNode(int(data_list[0]))
        data_list.pop(0)
        root.left = dfs(data_list)
        root.right = dfs(data_list)
        return root

    return dfs(str_list)


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
