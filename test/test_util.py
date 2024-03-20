from typing import List, Optional

import pytest
from leetcode.data_structures.list_node import ListNode
from leetcode.util import Util


class TestUtil:
    util = Util()

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([2, 7, 11, 15], util.create_list_node([2, 7, 11, 15])),
            ([3, 2, 4], util.create_list_node([3, 2, 4])),
            ([3, 3], util.create_list_node([3, 3])),
            ([2, 7, 11, 15], util.create_list_node([2, 7, 11, 15])),
        ],
    )
    def test_create_list_node(
        self, test_input: List[int], expected: Optional[ListNode]
    ):
        actual = self.util.create_list_node(test_input)
        assert expected == actual


    @pytest.mark.parametrize(
        "head, expected",
        [
            (util.create_list_node([2, 7, 11, 15]),  "2->7->11->15"),
            (util.create_list_node([3]), "3"),
            (util.create_list_node([]), "")
        ],
    )
    def test_print_list_node(self, head: ListNode, expected: str):
        actual = self.util.print_list_node(head)
        assert expected == actual

    
    @pytest.mark.parametrize(
        "head, expected",
        [
            (util.create_list_node([2, 7, 11, 15]),  [2, 7, 11, 15]),
            (util.create_list_node([3]), [3]),
            (util.create_list_node([]), [])
        ],
    )
    def test_list_node_to_list(self, head: ListNode, expected: List[int]):
        actual = self.util.list_node_to_list(head)
        assert expected == actual