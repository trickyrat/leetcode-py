import pytest

from solution import Solution
from utils import *

solution = Solution()


@pytest.mark.parametrize(
    "test_input, target, expect",
    [([2, 7, 11, 15], 9, [0, 1])],
)
def test_two_sum(test_input: List[int], target: int, expect: List[int]):
    actual = solution.twoSum(test_input, target)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input1, test_input2, expect",
    [
        (create_list_node([2, 4, 3]), create_list_node([5, 6, 4]), [7, 0, 8]),
        (create_list_node([0]), create_list_node([0]), [0]),
        (create_list_node([9, 9, 9, 9, 9, 9, 9]), create_list_node([9, 9, 9, 9]), [8, 9, 9, 9, 0, 0, 0, 1]),
    ],
)
def test_add_two_numbers(
        test_input1: ListNode, test_input2: ListNode, expect: List[int]
):
    actual = list_node_to_list(solution.addTwoNumbers(test_input1, test_input2))
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]
)
def test_longest_substring_without_repeat(test_input: str, expect: int):
    actual = solution.longestSubstringWithoutRepeat(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input1, test_input2, expect",
    [
        ([1, 2, 3, 4], [5, 6, 7, 8], 4.5),
        ([1, 3], [2], 2.0),
        ([0, 0], [0, 0], 0.0),
        ([], [1], 1.0),
        ([2], [], 2.0),
    ],
)
def test_find_median_sorted_arrays(
        test_input1: List[int], test_input2: List[int], expect: float
):
    actual = solution.findMedianSortedArrays(test_input1, test_input2)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ("babad", "bab"),
        ("cbbd", "bb"),
        ("a", "a"),
        ("ac", "a"),
    ],
)
def test_longest_palindrome(test_input: str, expect: str):
    actual = solution.longestPalindrome(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, num_rows, expect",
    [
        ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
        ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
        ("A", 1, "A"),
    ],
)
def test_z_convert(test_input: str, num_rows: int, expect: str):
    actual = solution.zconvert(test_input, num_rows)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        (123, 321),
        (-123, -321),
        (120, 21),
        (100, 1),
        (120, 21),
        (-(2 ** 31), 0),
        (2 ** 31 - 1, 0),
    ],
)
def test_reverse_int(test_input: int, expect: str):
    actual = solution.reverseInt(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
    ],
)
def test_merge(test_input: List[List[int]], expect: List[List[int]]):
    actual = solution.merge(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        ([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]], [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]),
    ],
)
def test_set_zeroes(test_input: List[List[int]], expect: List[List[int]]):
    solution.setZeroes(test_input)
    assert expect == test_input


@pytest.mark.parametrize(
    "test_input, target, expect",
    [
        ([4, 5, 6, 7, 0, 1, 2], 0, 4),
        ([4, 5, 6, 7, 0, 1, 2], 3, -1),
        ([1], 0, -1),
    ],
)
def test_eval(test_input: List[int], target: int, expect: int):
    actual = solution.search(test_input, target)
    assert expect == actual


@pytest.mark.parametrize(
    "matrix, target, expect",
    [
        ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3, True),
        ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13, False),
    ],
)
def test_search_matrix(matrix: List[List[int]], target: int, expect: bool):
    actual = solution.searchMatrix(matrix, target)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        (create_treenode_with_bfs("3,9,20,null,null,15,7"), 2),
        (create_treenode_with_bfs("2,null,3,null,4,null,5,null,6"), 5),
    ],
)
def test_min_depth(test_input: TreeNode, expect: int):
    actual = solution.minDepth(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, target_num, expect",
    [
        (
                create_treenode_with_bfs("5,4,8,11,null,13,4,7,2,null,null,5,1"),
                22,
                [[5, 4, 11, 2], [5, 8, 4, 5]],
        ),
        (create_treenode_with_bfs("1,2,3"), 5, []),
    ],
)
def test_path_sum(test_input: TreeNode, target_num: int, expect: List[List[int]]):
    actual = solution.pathSum(test_input, target_num)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, target, expect",
    [([2, 7, 11, 15], 9, [1, 2]), ([2, 3, 4], 6, [1, 3]), ([-1, 0], -1, [1, 2])],
)
def test_two_sum(test_input: List[int], target: int, expect: List[int]):
    actual = solution.twoSumII(test_input, target)
    assert actual == expect


@pytest.mark.parametrize(
    "actual, k, expect",
    [
        ([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]),
        ([-1, -100, 3, 99], 2, [3, 99, -1, -100]),
    ],
)
def test_rotate_array(actual: List[int], k: int, expect: List[int]):
    solution.rotate(actual, k)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect", [([0], [0]), ([0, 1, 0, 3, 12], [1, 3, 12, 0, 0])]
)
def test_move_zeroes(test_input: List[int], expect: List[int]):
    solution.moveZeroes(test_input)
    assert test_input == expect


@pytest.mark.parametrize(
    "test_input, expect", [([1, 3, 4, 2, 2], 2), ([3, 1, 3, 4, 2], 3)]
)
def test_find_duplicate(test_input: List[int], expect: int):
    actual = solution.findDuplicate(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [([1, 9], 10), ([-1, 9], 8), ([1, 2], 3)]
)
def test_get_sum(test_input: List[int], expect: int):
    actual = solution.getSum(test_input[0], test_input[1])
    assert expect == actual


@pytest.mark.parametrize("test_input, expect", [([197, 130, 1], True), ([235, 140, 4], False)])
def test_valid_utf8(test_input: List[int], expect: bool):
    actual = solution.validUtf8(test_input)
    assert expect == actual


@pytest.mark.parametrize("test_input, expect", [("Hello, my name is John", 5)])
def test_count_segment(test_input: str, expect: int):
    actual = solution.countSegment(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 7, 5, 3, 6, 8, 9]),
     ([[1, 2], [3, 4]], [1, 2, 3, 4])]
)
def test_find_diagonal_order(test_input: List[List[int]], expect: List[int]):
    actual = solution.findDiagonalOrder(test_input)
    assert actual == expect


@pytest.mark.parametrize("test_input, expect", [(100, "202"), (-7, "-10")])
def test_convert_to_base7(test_input: int, expect: int):
    actual = solution.convertToBase7(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "a, b, expect", [("aba", "cdc", 3), ("aaa", "bbb", 3), ("aaa", "aaa", -1)]
)
def test_find_lus_length(a: str, b: str, expect: int):
    actual = solution.findLUSLength(a, b)
    assert expect == actual


@pytest.mark.parametrize(
    "input1, input2, expect",
    [("1+1i", "1+1i", "0+2i"), ("1+-1i", "1+-1i", "0+-2i")],
)
def test_complex_number_multiply(input1: str, input2: str, expect: str):
    actual = solution.complexNumberMultiply(input1, input2)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([1000, 100, 10, 2], "1000/(100/10/2)")],
)
def test_optimal_division(test_input: List[int], expect: str):
    actual = solution.optimalDivision(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "l1, l2, expect",
    [(["Shogun", "Tapioca Express", "Burger King", "KFC"],
      ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"], ["Shogun"]),
     (["Shogun", "Tapioca Express", "Burger King", "KFC"], ["KFC", "Shogun", "Burger King"], ["Shogun"])],
)
def test_find_restaurant(l1: List[str], l2: List[str], expect: List[str]):
    actual = solution.findRestaurant(l1, l2)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([1, 2, 3, 4, 5, 6], [4, 5, 6]), ([1, 2, 3, 4, 5], [3, 4, 5])],
)
def test_middle_node(test_input: List[int], expect: List[int]):
    actual = list_node_to_list(solution.middleNode(create_list_node(test_input)))
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
        ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
    ],
)
def test_sorted_squares(test_input: List[int], expect: List[int]):
    actual = solution.sortedSquares(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ("lee(t(c)o)de)", "lee(t(c)o)de"),
        ("a)b(c)d", "ab(c)d"),
        ("))((", ""),
    ],
)
def test_min_remove_to_make_valid(test_input: str, expect: str):
    actual = solution.minRemoveToMakeValid(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([2, 3, -1, 8, 4], 3),
        ([1, -1, 4], 2),
        ([2, 5], -1),
    ],
)
def test_pivot_index(test_input: List[int], expect: int):
    actual = solution.pivotIndex(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, k, expect",
    [
        ([1, 2, 2, 1], 1, 4),
        ([1, 3], 3, 0),
        ([3, 2, 1, 5, 4], 2, 3),
    ],
)
def test_count_k_difference(test_input: List[int], k: int, expect: int):
    actual = solution.countKDifference(test_input, k)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([7, 1, 5, 4], 4),
        ([9, 4, 3, 2], -1),
        ([1, 5, 2, 10], 9),
    ],
)
def test_maximum_difference(test_input: List[int], expect: int):
    actual = solution.maximumDifference(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([3, 1], 2),
        ([2, 2, 2], 7),
        ([3, 2, 1, 5], 6),
    ],
)
def test_count_max_or_subsets(test_input: List[int], expect: int):
    actual = solution.countMaxOrSubsets(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, queries, expect",
    [
        ("**|**|***|", [[2, 5], [5, 9]], [2, 3]),
        (
                "***|**|*****|**||**|*",
                [[1, 17], [4, 5], [14, 17], [5, 11], [15, 16]],
                [9, 0, 0, 0, 0],
        ),
    ],
)
def test_plates_between_candles(
        test_input: str, queries: List[List[int]], expect: List[int]
):
    actual = solution.platesBetweenCandles(test_input, queries)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        (
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ],
                [
                    [7, 4, 1],
                    [8, 5, 2],
                    [9, 6, 3]
                ]
        ),
        (
                [
                    [5, 1, 9, 11],
                    [2, 4, 8, 10],
                    [13, 3, 6, 7],
                    [15, 14, 12, 16]
                ],
                [
                    [15, 13, 2, 5],
                    [14, 3, 4, 1],
                    [12, 6, 8, 9],
                    [16, 7, 10, 11]
                ]
        )
    ],
)
def test_rotate(
        test_input: List[List[int]], expect: List[List[int]]
):
    solution.rotate_matrix(test_input)
    assert test_input == expect
