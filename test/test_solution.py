import pytest

from Solution import Solution
from Utils import *

solution = Solution()


@pytest.mark.parametrize(
    "test_input, target, expect",
    [([2, 7, 11, 15], 9, [0, 1])],
)
def test_two_sum(test_input: List[int], target: int, expect: List[int]):
    actual = solution.two_sum(test_input, target)
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
    actual = list_node_to_list(solution.add_two_numbers(test_input1, test_input2))
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]
)
def test_longest_substring_without_repeat(test_input: str, expect: int):
    actual = solution.longest_substring_without_repeat(test_input)
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
    actual = solution.find_median_sorted_arrays(test_input1, test_input2)
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
    actual = solution.longest_palindrome(test_input)
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
    actual = solution.z_convert(test_input, num_rows)
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
    actual = solution.reverse_int(test_input)
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
    solution.set_zeroes(test_input)
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
    actual = solution.search_matrix(matrix, target)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        (create_treenode([3, 9, 20, None, None, 15, 7]), 2),
        (create_treenode([2, None, 3, None, 4, None, 5, None, 6]), 5),
    ],
)
def test_min_depth(test_input: TreeNode, expect: int):
    actual = solution.min_depth(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, target_num, expect",
    [
        (
                create_treenode([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]),
                22,
                [[5, 4, 11, 2], [5, 8, 4, 5]],
        ),
        (create_treenode([1, 2, 3]), 5, []),
    ],
)
def test_path_sum(test_input: TreeNode, target_num: int, expect: List[List[int]]):
    actual = solution.path_sum(test_input, target_num)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, target, expect",
    [([2, 7, 11, 15], 9, [1, 2]), ([2, 3, 4], 6, [1, 3]), ([-1, 0], -1, [1, 2])],
)
def test_two_sum(test_input: List[int], target: int, expect: List[int]):
    actual = solution.two_sum_ii(test_input, target)
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
    solution.move_zeroes(test_input)
    assert test_input == expect


@pytest.mark.parametrize(
    "test_input, expect", [([1, 3, 4, 2, 2], 2), ([3, 1, 3, 4, 2], 3)]
)
def test_find_duplicate(test_input: List[int], expect: int):
    actual = solution.find_duplicate(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [(2, 91), (0, 1)]
)
def test_count_numbers_with_unique_digits(test_input: int, expect: int):
    actual = solution.count_numbers_with_unique_digits(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [([1, 9], 10), ([-1, 9], 8), ([1, 2], 3)]
)
def test_get_sum(test_input: List[int], expect: int):
    actual = solution.get_sum(test_input[0], test_input[1])
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect", [(13, [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]), (2, [1, 2])]
)
def test_lexical_order(test_input: int, expect: List[int]):
    actual = solution.lexical_order(test_input)
    assert expect == actual


@pytest.mark.parametrize("test_input, expect", [([197, 130, 1], True), ([235, 140, 4], False)])
def test_valid_utf8(test_input: List[int], expect: bool):
    actual = solution.valid_utf8(test_input)
    assert expect == actual


@pytest.mark.parametrize("test_input, expect", [("Hello, my name is John", 5)])
def test_count_segment(test_input: str, expect: int):
    actual = solution.count_segment(test_input)
    assert expect == actual


@pytest.mark.parametrize("p, expect", [("a", 1), ("cac", 2), ("zab", 6)])
def test_find_substring_wraparound_string(p: str, expect: int):
    actual = solution.find_substring_wraparound_string(p)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 7, 5, 3, 6, 8, 9]),
     ([[1, 2], [3, 4]], [1, 2, 3, 4])]
)
def test_find_diagonal_order(test_input: List[List[int]], expect: List[int]):
    actual = solution.find_diagonal_order(test_input)
    assert actual == expect


@pytest.mark.parametrize("test_input, expect", [(100, "202"), (-7, "-10")])
def test_convert_to_base7(test_input: int, expect: int):
    actual = solution.convert_to_base7(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "a, b, expect", [("aba", "cdc", 3), ("aaa", "bbb", 3), ("aaa", "aaa", -1)]
)
def test_find_lus_length(a: str, b: str, expect: int):
    actual = solution.find_lus_length(a, b)
    assert expect == actual


@pytest.mark.parametrize(
    "input1, input2, expect",
    [("1+1i", "1+1i", "0+2i"), ("1+-1i", "1+-1i", "0+-2i")],
)
def test_complex_number_multiply(input1: str, input2: str, expect: str):
    actual = solution.complex_number_multiply(input1, input2)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([1000, 100, 10, 2], "1000/(100/10/2)")],
)
def test_optimal_division(test_input: List[int], expect: str):
    actual = solution.optimal_division(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "l1, l2, expect",
    [(["Shogun", "Tapioca Express", "Burger King", "KFC"],
      ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"], ["Shogun"]),
     (["Shogun", "Tapioca Express", "Burger King", "KFC"], ["KFC", "Shogun", "Burger King"], ["Shogun"])],
)
def test_find_restaurant(l1: List[str], l2: List[str], expect: List[str]):
    actual = solution.find_restaurant(l1, l2)
    assert expect == actual


@pytest.mark.parametrize(
    "root, val, depth, expect",
    [(create_treenode([4, 2, 6, 3, 1, 5]), 1, 2, create_treenode([4, 1, 1, 2, None, None, 6, 3, 1, 5])),
     (create_treenode([4, 2, None, 3, 1]), 1, 3, create_treenode([4, 2, None, 1, 1, 3, None, None, 1]))]
)
def test_add_one_row(root: TreeNode | None, val: int, depth: int, expect: TreeNode | None):
    actual = solution.add_one_row(root, val, depth)
    expect = preorder_traversal(expect)
    actual = preorder_traversal(actual)
    assert expect == actual


@pytest.mark.parametrize(
    "left, right, expect",
    [(1, 22, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]), (47, 85, [48, 55, 66, 77])],
)
def test_self_dividing_number(left: int, right: int, expect: List[int]):
    actual = solution.self_dividing_number(left, right)
    assert expect == actual


@pytest.mark.parametrize(
    "letters, target, expect",
    [(["c", "f", "j"], "a", "c"), (["c", "f", "j"], "c", "f"), (["c", "f", "j"], "d", "f")],
)
def test_next_greatest_letter(letters: List[str], target: str, expect: str):
    actual = solution.next_greatest_letter(letters, target)
    assert expect == actual


@pytest.mark.parametrize(
    "words, expect",
    [(["gin", "zen", "gig", "msg"], 2), (["a"], 1)],
)
def test_unique_morse_representations(words: List[str], expect: int):
    actual = solution.unique_morse_representations(words)
    assert expect == actual


@pytest.mark.parametrize(
    "widths, s, expect",
    [([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
      "abcdefghijklmnopqrstuvwxyz",
      [3, 60]), (
             [4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             "bbbcccdddaaa", [2, 4])],
)
def test_number_of_lines(widths: List[int], s: str, expect: List[int]):
    actual = solution.number_of_lines(widths, s)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([1, 2, 3, 4, 5, 6], [4, 5, 6]), ([1, 2, 3, 4, 5], [3, 4, 5])],
)
def test_middle_node(test_input: List[int], expect: List[int]):
    actual = list_node_to_list(solution.middle_node(create_list_node(test_input)))
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [([[1, 2], [3, 4]], 17), ([[2]], 5), ([[1, 0], [0, 2]], 8)],
)
def test_projection_area(test_input: List[List[int]], expect: int):
    actual = solution.projection_area(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
        ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
    ],
)
def test_sorted_squares(test_input: List[int], expect: List[int]):
    actual = solution.sorted_squares(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([4, 3, 10, 9, 8], [10, 9]),
        ([4, 4, 7, 6, 7], [7, 7, 6]),
        ([6], [6]),
    ],
)
def test_min_subsequence(test_input: List[int], expect: List[int]):
    actual = solution.min_subsequence(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ("a0b1c2", ["0a1b2c", "a0b1c2", "0a1b2c", "0c2a1b"]),
        ("leetcode", [""]),
        ("1229857369", [""]),
    ],
)
def test_reformat(test_input: str, expect: List[str]):
    actual = solution.reformat(test_input)
    assert actual in expect


@pytest.mark.parametrize(
    "start_time, end_time, query_time, expect",
    [
        ([1, 2, 3], [3, 2, 7], 4, 1),
        ([4], [4], 4, 1),
    ],
)
def test_busy_student(start_time: List[int], end_time: List[int], query_time: int, expect: int):
    actual = solution.busy_student(start_time, end_time, query_time)
    assert actual == expect

@pytest.mark.parametrize(
    "sentence, search_word, expect",
    [
        ("i love eating burger", "burg", 4),
        ("this problem is an easy problem", "pro", 2),
        ("i am tired", "you", -1),
    ],
)
def test_is_prefix_of_word(sentence: str, search_word: str,  expect: int):
    actual = solution.is_prefix_of_word(sentence, search_word)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([3, 5, 1], True),
        ([1, 2, 4], False),
    ],
)
def test_can_make_arithmetic_progression(test_input: List[int], expect: bool):
    actual = solution.can_make_arithmetic_progression(test_input)
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
    actual = solution.min_remove_to_make_valid(test_input)
    assert expect == actual


@pytest.mark.parametrize(
    "n, k, expect",
    [
        (5, 2, 3),
        (6, 5, 1),
    ],
)
def test_find_the_winner(n: int, k: int, expect: int):
    actual = solution.find_the_winner(n, k)
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
    actual = solution.pivot_index(test_input)
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
    actual = solution.count_k_difference(test_input, k)
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
    actual = solution.maximum_difference(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "test_input, expect",
    [
        ([3, 1, 4, 2], [2, 4, 1, 3]),
        ([0], [0]),
    ],
)
def test_sort_array_by_parity(test_input: List[int], expect: List[int]):
    actual = solution.sort_array_by_parity(test_input)
    assert actual == expect


@pytest.mark.parametrize(
    "s, expect",
    [
        ("IDID", [0, 4, 1, 3, 2]),
        ("III", [0, 1, 2, 3]),
        ("DDI", [3, 2, 0, 1]),
    ],
)
def test_di_string_match(s: str, expect: List[int]):
    actual = solution.di_string_match(s)
    assert actual == expect


@pytest.mark.parametrize(
    "strs, expect",
    [
        (["cba", "daf", "ghi"], 1),
        (["a", "b"], 0),
        (["zyx", "wvu", "tsr"], 3),
    ],
)
def test_min_deletion_size(strs: List[str], expect: int):
    actual = solution.min_deletion_size(strs)
    assert actual == expect


@pytest.mark.parametrize(
    "nums, expect",
    [
        ([1, 2, 3, 3], 3),
        ([2, 1, 2, 5, 3, 2], 2),
        ([5, 1, 5, 2, 5, 3, 5, 4], 5),
    ],
)
def test_repeated_n_times(nums: List[int], expect: int):
    actual = solution.repeated_n_times(nums)
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
    actual = solution.count_max_or_subsets(test_input)
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
    actual = solution.plates_between_candles(test_input, queries)
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
