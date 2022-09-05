from collections import deque
from typing import Optional


class TreeNode(object):
    """
    Definition for a binary tree node.
    """

    def __init__(self, val=0, left=None, right=None) -> None:
        self.val: int = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right

    def __eq__(self, other):
        if not self and not other:
            return True
        if not self or not other:
            return False

        queue1 = deque([self])
        queue2 = deque([other])

        while queue1 and queue2:
            node1 = queue1.popleft()
            node2 = queue2.popleft()
            if node1.val != node2.val:
                return False
            left1, right1 = node1.left, node1.right
            left2, right2 = node2.left, node2.right
            if (not left1) ^ (not left2):
                return False
            if (not right1) ^ (not right2):
                return False
            if left1:
                queue1.append(left1)
            if right1:
                queue1.append(right1)
            if left2:
                queue2.append(left2)
            if right2:
                queue2.append(right2)

        return not queue1 and not queue2
