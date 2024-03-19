from leetcode.random_node import RandomNode
from leetcode.util import Util


class TestRandomNode:
    util = Util()
    def test_random_node(self):
        node = RandomNode(self.util.create_list_node([1, 2, 3]))
        expected = [1, 2, 3]
        actual = node.get_random()
        assert actual in expected
        actual = node.get_random()
        assert actual in expected
        actual = node.get_random()
        assert actual in expected
        actual = node.get_random()
        assert actual in expected
        actual = node.get_random()
        assert actual in expected
