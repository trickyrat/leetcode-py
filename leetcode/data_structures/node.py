from typing import Optional, List


class Node(object):
    def __init__(self, val=None, children=None):
        self.val: int = val
        self.children: Optional[List[Node]] = children
