from typing import List

import pytest
from src.data_structures.list_node import ListNode
from src.data_structures.tree_node import TreeNode
from src.util import Util


class TestUtil:
    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([2, 7, 11, 15], "2->7->11->15"),
            ([3, 2, 4], "3->2->4"),
            ([3, 3], "3->3"),
        ],
    )
    def test_generate_list_node(
        self, test_input: List[int], expected: str
    ):
        head = Util.generate_list_node(test_input)
        actual = Util.list_node_to_string(head)
        assert expected == actual

    def test_generate_list_node_empty(self):
        nodes = []
        result = Util.generate_list_node(nodes)
        assert result is None

    def test_generate_list_node_single_element(self):
        nodes = [1]
        result = Util.generate_list_node(nodes)
        assert result is not None
        assert result.val == 1
        assert result.next is None

    def test_generate_list_node_multiple_elements(self):
        nodes = [1, 2, 3]
        result = Util.generate_list_node(nodes)
        assert result is not None
        assert result.val == 1
        assert result.next is not None and result.next.val == 2
        assert result.next.next is not None and result.next.next.val == 3
        assert result.next.next.next is None

    @pytest.mark.parametrize(
        "head, expected",
        [
            (ListNode(2, ListNode(7, ListNode(11, ListNode(15)))), "2->7->11->15"),
            (ListNode(3), "3"),
            (None, ""),
        ],
    )
    def test_list_node_to_string(self, head: ListNode, expected: str):
        actual = Util.list_node_to_string(head)
        assert expected == actual

    def test_empty_list(self):
        head = None
        expected = ""
        assert Util.list_node_to_string(head) == expected

    def test_single_node(self):
        head = ListNode(1)
        expected = "1"
        assert Util.list_node_to_string(head) == expected

    def test_multiple_nodes(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        expected = "1->2->3"
        assert Util.list_node_to_string(head) == expected

    def test_circular_list(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = head
        expected = "1->2->3->1"
        assert Util.list_node_to_string(head) == expected

    def test_list_node_to_string_circular_start_in_middle(self):
        head = ListNode(2)
        head.next = ListNode(3)
        head.next.next = ListNode(1)
        head.next.next.next = head
        assert Util.list_node_to_string(head) == "2->3->1->2"

    def test_list_node_to_string_non_circular_same_values(self):
        head = ListNode(1)
        head.next = ListNode(1)
        head.next.next = ListNode(1)
        assert Util.list_node_to_string(head) == "1->1->1"

    @pytest.mark.parametrize(
        "head, expected",
        [
            (ListNode(2, ListNode(7, ListNode(11, ListNode(15)))), [2, 7, 11, 15]),
            (ListNode(3), [3]),
            (None, []),
        ],
    )
    def test_list_node_to_list(self, head: ListNode, expected: List[int]):
        actual = Util.list_node_to_list(head)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([], None),
            ([1], TreeNode(1)),
            ([1, 2, 3], TreeNode(1, TreeNode(2), TreeNode(3))),
            ([1, None, 3], TreeNode(1, None, TreeNode(3))),
            ([1, 2, None, None, 5], TreeNode(1, TreeNode(2, None, TreeNode(5)), None)),
        ],
    )
    def test_generate_tree_node(self, nums, expected):
        actual_root = Util.generate_tree_node(nums)

        def assert_tree_nodes_equal(node1: TreeNode, node2: TreeNode) -> bool:
            if not node1 and not node2:
                return True
            elif (not node1 or not node2) or node1.val != node2.val:
                return False
            return assert_tree_nodes_equal(
                node1.left, node2.left
            ) and assert_tree_nodes_equal(node1.right, node2.right)

        assert assert_tree_nodes_equal(actual_root, expected)

    def test_generate_n_tree_node_empty_list(self):
        nums = []
        root = Util.generate_n_tree_node(nums)
        assert root is None

    def test_generate_n_tree_node_single_node(self):
        nums = [1]
        root = Util.generate_n_tree_node(nums)
        assert root is not None
        assert root.val == 1
        assert len(root.children) == 0

    def test_generate_n_tree_node_multiple_nodes(self):
        nums = [1, None, 2, 3, None, 4, 5, 6]
        root = Util.generate_n_tree_node(nums)
        assert root is not None
        assert root.val == 1
        assert len(root.children) == 2
        assert root.children[0].val == 2
        assert root.children[1].val == 3
        assert len(root.children[1].children) == 0
        assert root.children[0].children[0].val == 4
        assert root.children[0].children[1].val == 5
        assert root.children[0].children[1].children == []

    def test_generate_n_tree_node_nested_structure(self):
        nums = [1, None, 2, 3, None, 4, None, 5, 6]
        root = Util.generate_n_tree_node(nums)
        assert root is not None
        assert len(root.children) == 2
        assert root.children[0].val == 2
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].val == 4
        assert len(root.children[0].children) == 1
        assert root.children[1].children[0].val == 5
