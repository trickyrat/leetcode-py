import pytest
from src.leetcode.random_node import RandomNode
from src.util import Util


class TestRandomNode:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.util = Util()
    def test_random_node(self):
        node = RandomNode(self.util.generate_list_node([1, 2, 3]))
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
