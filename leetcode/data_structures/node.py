from typing import Optional, List


class Node(object):
    def __init__(self, val: int = 0, children: Optional[List["Node"]] = []):
        self.val: int = val
        self.children: Optional[List[Node]] = children
