from leetcode.random_node import RandomNode
from leetcode.utils import create_list_node


class TestRandomNode:
    def test_random_node(self):
        node = RandomNode(create_list_node([1, 2, 3]))
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

