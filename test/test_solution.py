from typing import List
from utils import initListNode, printListNode
from solution import Solution

import pytest

solution = Solution()


@pytest.mark.parametrize(
    "test_input, target, expected",
    [([2, 7, 11, 15], 9, [0, 1])],
)
def test_two_sum(test_input: List[int], target: int, expected: List[int]):
    actual = solution.twoSum(test_input, target)
    assert len(actual) == len(expected)
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


@pytest.mark.parametrize(
    "test_input1, test_input2,expected_input",
    [
        ([2, 4, 3], [5, 6, 4], [7, 0, 8]),
        ([0], [0], [0]),
        ([9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9], [8, 9, 9, 9, 0, 0, 0, 1]),
    ],
)
def test_add_two_numbers(
    test_input1: List[int], test_input2: List[int], expected_input: List[int]
):
    head1 = initListNode(test_input1)
    head2 = initListNode(test_input2)
    actual = printListNode(solution.addTwoNumbers(head1, head2))
    expected = printListNode(initListNode(expected_input))
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected", [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]
)
def test_longest_substring_without_repeat(test_input: str, expected: int):
    actual = solution.longestSubstringWithoutRepeat(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input1, test_input2, expected",
    [
        ([1, 2, 3, 4], [5, 6, 7, 8], 4.5),
        ([1, 3], [2], 2.0),
        ([0, 0], [0, 0], 0.0),
        ([], [1], 1.0),
        ([2], [], 2.0),
    ],
)
def test_find_median_sorted_arrays(
    test_input1: List[int], test_input2: List[int], expected: float
):
    actual = solution.findMedianSortedArrays(test_input1, test_input2)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("babad", "bab"),
        ("cbbd", "bb"),
        ("a", "a"),
        ("ac", "a"),
    ],
)
def test_longest_palindrome(test_input: str, expected: str):
    actual = solution.longestPalindrome(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, target, expected",
    [
        ([4, 5, 6, 7, 0, 1, 2], 0, 4),
        ([4, 5, 6, 7, 0, 1, 2], 3, -1),
        ([1], 0, -1),
    ],
)
def test_eval(test_input: List[int], target: int, expected: int):
    actual = solution.search(test_input, target)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, target, expected",
    [([2, 7, 11, 15], 9, [1, 2]), ([2, 3, 4], 6, [1, 3]), ([-1, 0], -1, [1, 2])],
)
def test_two_sumII(test_input: List[int], target: int, expected: List[int]):
    actual = solution.twoSumII(test_input, target)
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


@pytest.mark.parametrize(
    "actual, k, expected",
    [
        ([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]),
        ([-1, -100, 3, 99], 2, [3, 99, -1, -100]),
    ],
)
def test_rotate_array(actual: List[int], k: int, expected: List[int]):
    solution.rotate(actual, k)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected", [([0], [0]), ([0, 1, 0, 3, 12], [1, 3, 12, 0, 0])]
)
def test_move_zeroes(test_input: List[int], expected: List[int]):
    solution.moveZeroes(test_input)
    for i in range(0, len(expected)):
        assert test_input[i] == expected[i]


@pytest.mark.parametrize(
    "test_input, expected", [([1, 9], 10), ([-1, 9], 8), ([1, 2], 3)]
)
def test_get_sum(test_input: List[int], expected: int):
    actual = solution.getSum(test_input[0], test_input[1])
    assert expected == actual


@pytest.mark.parametrize("test_input, expected", [("Hello, my name is John", 5)])
def test_count_segment(test_input: str, expected: int):
    actual = solution.countSegment("Hello, my name is John")
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [([1, 2, 3, 4, 5, 6], [4, 5, 6]), ([1, 2, 3, 4, 5], [3, 4, 5])],
)
def test_middle_node(test_input: List[int], expected: List[int]):
    actualString = printListNode(solution.middleNode(initListNode(test_input)))
    expectedString = printListNode(initListNode(expected))
    assert expectedString == actualString


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
        ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
    ],
)
def test_sorted_squares(test_input: List[int], expected: List[int]):
    actual = solution.sortedSquares(test_input)
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]
