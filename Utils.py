from ListNode import ListNode
from TreeNode import TreeNode
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


def create_treenode_with_bfs(data: str) -> Optional[TreeNode]:
    nums = data.split(",")
    n = len(nums)
    if n == 0:
        return None
    if nums[0] == "null":
        return None
    root = TreeNode(int(nums[0]))
    q = queue.Queue()
    q.put(root)
    index = 1
    while index < n:
        node = q.get()
        if index > len(nums) - 1 or nums[index] == "null":
            node.left = None
        else:
            left_node = TreeNode(int(nums[index]))
            if left_node:
                node.left = left_node
            q.put(left_node)
        if index + 1 > len(nums) - 1 or nums[index + 1] == "null":
            node.right = None
        else:
            right_node = TreeNode(int(nums[index + 1]))
            if right_node:
                node.right = right_node
            q.put(right_node)
        index += 2
    return root


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
