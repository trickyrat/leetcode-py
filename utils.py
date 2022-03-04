from solution import ListNode, TreeNode
from typing import List, Optional
import queue


def initListNode(nodes: List[int]) -> Optional[ListNode]:
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


def createTreeNodeWithBFS(data: str) -> Optional[TreeNode]:
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
        leftVal, rightVal = nums[index], nums[index + 1]
        if leftVal != "null":
            leftNode = TreeNode(int(leftVal))
            if not leftNode:
                node.left = leftNode
            q.put(leftNode)
        if rightVal != "null":
            rightNode = TreeNode(int(rightVal))
            if not rightNode:
                node.right = rightNode
            q.put(rightNode)
        index += 2
    return root


def createTreeNodeWithDFS(data: str) -> Optional[TreeNode]:
    list = data.split(",")

    def dfs(dataList: List[str]):
        if len(dataList) == 0:
            return None
        if dataList[0] == "null":
            dataList.pop(0)
            return None
        root = TreeNode(int(dataList[0]))
        dataList.pop(0)
        root.left = dfs(dataList)
        root.right = dfs(dataList)
        return root

    return dfs(list)
