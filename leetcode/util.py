from leetcode.data_structures.list_node import ListNode
from leetcode.data_structures.tree_node import TreeNode
from leetcode.data_structures.node import Node
from typing import List, Optional
from collections import deque


class Util:
    def create_list_node(self, nodes: List[int]) -> Optional[ListNode]:
        """Create a linked-list with a list"""
        head = ListNode(0)
        dummy = head
        for node in nodes:
            dummy.next = ListNode(node)
            dummy = dummy.next
        return head.next

    def print_list_node(self, head: Optional[ListNode]) -> str:
        """Convert a linked-list to string"""
        res = ""
        while head is not None:
            res += str(head.val)
            if head is not None:
                res += "->"
        return res

    def list_node_to_list(self, head: Optional[ListNode]) -> List[int]:
        """Convert a linked-list to a list"""
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res

    def create_tree_node(self, nums: List[int | None]) -> Optional[TreeNode]:
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

    def preorder_traversal(self, root: TreeNode) -> List[int]:
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

    def create_n_tree_node(self, nums: List[Optional[int]]) -> Optional[Node]:
        """Create a n-ary tree node with a list"""
        if nums is None or len(nums) == 0 or nums[0] is None:
            return None

        root = Node(nums[0], [])
        q = [root]
        for i in range(1, len(nums)):
            if nums[i] is not None:
                parent = q[0]
                child = Node(nums[i], [])
                parent.children.append(child)
                q.append(child)
            elif nums[i] is None and len(q) < 2:
                continue
            else:
                q.pop(0)
        return root
