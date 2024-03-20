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
            if head.next is not None:
                res += "->"
            head = head.next
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
        if nums is None or len(nums) == 0 or nums[0] is None:
            return None
        
        root = TreeNode(nums[0])
        q = [root]
        fill_left = True
        for i in range(1, len(nums)):
            node = TreeNode(nums[i]) if nums[i] is not None else None
            if fill_left:
                q[0].left = node
                fill_left = False
            else:
                q[0].right = node
                fill_left = True

            if node is not None:
                q.append(node)
            
            if fill_left:
                q.pop(0)
        return root

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
