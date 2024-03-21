from src.data_structures.list_node import ListNode
from src.data_structures.tree_node import TreeNode
from src.data_structures.node import Node
from typing import List, Optional


class Util:
    def generate_list_node(self, nodes: List[int]) -> Optional[ListNode]:
        """
        
        Generate a linked-list with a list
        
        Args:

        nodes List[int]: The list of nodes

        Returns:

        Optional[ListNode]: The head of the linked-list

        """
        head = ListNode(0)
        dummy = head
        for node in nodes:
            dummy.next = ListNode(node)
            dummy = dummy.next
        return head.next

    def list_node_to_string(self, head: Optional[ListNode], sep: str = "->") -> str:
        """
        
        Convert a linked-list to string

        Args:

        head Optional[ListNode]: The head of the linked-list

        sep str: The separator between each node, default is "->"

        Returns:

        str: The string representation of the linked-list
        
        """

        def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
            if head is None:
                return None

            slow = fast = head
            while fast is not None:
                slow = slow.next
                if fast.next is not None:
                    fast = fast.next.next
                else:
                    return None
                if fast == slow:
                    ptr = head
                    while ptr != slow:
                        ptr = ptr.next
                        slow = slow.next
                    return ptr
            return None

        cycle_node = detect_cycle(head)

        res = ""
        while head:
            res += f"{head.val}"
            if head.next:
                res += sep
            head = head.next
            if head == cycle_node and head and cycle_node:
                res += f"{cycle_node.val}"
                break
        return res

    def list_node_to_list(self, head: Optional[ListNode]) -> List[int]:
        """Convert a linked-list to a list"""
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res

    def generate_tree_node(self, nums: List[int | None]) -> Optional[TreeNode]:
        """
        
        Generate a binary tree node
        
        Args:

        nums: (List[int | None]): List of values

        Returns:

        Optional[TreeNode]: The root node of binary tree

        """
        if not nums or len(nums) == 0 or not nums[0]:
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

    def generate_n_tree_node(self, nums: List[Optional[int]]) -> Optional[Node]:
        """
        
        Generate a n-ary tree node with a list

        Args:

        nums: (List[Optional[int]]): List of values

        Returns: 
        
        Optional[Node]: The root node of n-ary tree
        """
        if not nums or len(nums) == 0 or not nums[0]:
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
