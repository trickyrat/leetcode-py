import pytest
from leetcode.solution import Solution
from leetcode.utils import *

solution = Solution()


@pytest.mark.parametrize(
    "test_input, target, expected",
    [([2, 7, 11, 15], 9, [0, 1])],
)
def test_two_sum(test_input: List[int], target: int, expected: List[int]):
    actual = solution.two_sum(test_input, target)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input1, test_input2, expected",
    [
        (create_list_node([2, 4, 3]), create_list_node([5, 6, 4]), [7, 0, 8]),
        (create_list_node([0]), create_list_node([0]), [0]),
        (create_list_node([9, 9, 9, 9, 9, 9, 9]), create_list_node([9, 9, 9, 9]), [8, 9, 9, 9, 0, 0, 0, 1]),
    ],
)
def test_add_two_numbers(
        test_input1: ListNode, test_input2: ListNode, expected: List[int]
):
    actual = list_node_to_list(solution.add_two_numbers(test_input1, test_input2))
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected", [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]
)
def test_longest_substring_without_repeat(test_input: str, expected: int):
    actual = solution.longest_substring_without_repeat(test_input)
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
    actual = solution.find_median_sorted_arrays(test_input1, test_input2)
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
    actual = solution.longest_palindrome(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, num_rows, expected",
    [
        ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
        ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
        ("A", 1, "A"),
    ],
)
def test_z_convert(test_input: str, num_rows: int, expected: str):
    actual = solution.z_convert(test_input, num_rows)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
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
def test_reverse_int(test_input: int, expected: str):
    actual = solution.reverse_int(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
        ([[1, 4], [4, 5]], [[1, 5]]),
    ],
)
def test_merge(test_input: List[List[int]], expected: List[List[int]]):
    actual = solution.merge(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        ([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]], [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]),
    ],
)
def test_set_zeroes(test_input: List[List[int]], expected: List[List[int]]):
    solution.set_zeroes(test_input)
    assert expected == test_input


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
    "matrix, target, expected",
    [
        ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3, True),
        ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13, False),
    ],
)
def test_search_matrix(matrix: List[List[int]], target: int, expected: bool):
    actual = solution.search_matrix(matrix, target)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (create_treenode_iteratively([3, 9, 20, None, None, 15, 7]), 2),
        (create_treenode_iteratively([2, None, 3, None, 4, None, 5, None, 6]), 5),
    ],
)
def test_min_depth(test_input: TreeNode, expected: int):
    actual = solution.min_depth(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, target_num, expected",
    [
        (
                create_treenode_iteratively([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]),
                22,
                [[5, 4, 11, 2], [5, 8, 4, 5]],
        ),
        (create_treenode_iteratively([1, 2, 3]), 5, []),
    ],
)
def test_path_sum(test_input: TreeNode, target_num: int, expected: List[List[int]]):
    actual = solution.path_sum(test_input, target_num)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, target, expected",
    [([2, 7, 11, 15], 9, [1, 2]), ([2, 3, 4], 6, [1, 3]), ([-1, 0], -1, [1, 2])],
)
def test_two_sum(test_input: List[int], target: int, expected: List[int]):
    actual = solution.two_sum_ii(test_input, target)
    assert actual == expected


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
    solution.move_zeroes(test_input)
    assert test_input == expected


@pytest.mark.parametrize(
    "test_input, expected", [([1, 3, 4, 2, 2], 2), ([3, 1, 3, 4, 2], 3)]
)
def test_find_duplicate(test_input: List[int], expected: int):
    actual = solution.find_duplicate(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected", [(2, 91), (0, 1)]
)
def test_count_numbers_with_unique_digits(test_input: int, expected: int):
    actual = solution.count_numbers_with_unique_digits(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected", [([1, 9], 10), ([-1, 9], 8), ([1, 2], 3)]
)
def test_get_sum(test_input: List[int], expected: int):
    actual = solution.get_sum(test_input[0], test_input[1])
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected", [(13, [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]), (2, [1, 2])]
)
def test_lexical_order(test_input: int, expected: List[int]):
    actual = solution.lexical_order(test_input)
    assert expected == actual


@pytest.mark.parametrize("test_input, expected", [([197, 130, 1], True), ([235, 140, 4], False)])
def test_valid_utf8(test_input: List[int], expected: bool):
    actual = solution.valid_utf8(test_input)
    assert expected == actual


@pytest.mark.parametrize("test_input, expected", [("Hello, my name is John", 5)])
def test_count_segment(test_input: str, expected: int):
    actual = solution.count_segment(test_input)
    assert expected == actual


@pytest.mark.parametrize("p, expected", [("a", 1), ("cac", 2), ("zab", 6)])
def test_find_substring_wraparound_string(p: str, expected: int):
    actual = solution.find_substring_wraparound_string(p)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 7, 5, 3, 6, 8, 9]),
     ([[1, 2], [3, 4]], [1, 2, 3, 4])]
)
def test_find_diagonal_order(test_input: List[List[int]], expected: List[int]):
    actual = solution.find_diagonal_order(test_input)
    assert actual == expected


@pytest.mark.parametrize("test_input, expected", [(100, "202"), (-7, "-10")])
def test_convert_to_base7(test_input: int, expected: int):
    actual = solution.convert_to_base7(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "a, b, expected", [("aba", "cdc", 3), ("aaa", "bbb", 3), ("aaa", "aaa", -1)]
)
def test_find_lus_length(a: str, b: str, expected: int):
    actual = solution.find_lus_length(a, b)
    assert expected == actual


@pytest.mark.parametrize(
    "input1, input2, expected",
    [("1+1i", "1+1i", "0+2i"), ("1+-1i", "1+-1i", "0+-2i")],
)
def test_complex_number_multiply(input1: str, input2: str, expected: str):
    actual = solution.complex_number_multiply(input1, input2)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [([1000, 100, 10, 2], "1000/(100/10/2)")],
)
def test_optimal_division(test_input: List[int], expected: str):
    actual = solution.optimal_division(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "l1, l2, expected",
    [(["Shogun", "Tapioca Express", "Burger King", "KFC"],
      ["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"], ["Shogun"]),
     (["Shogun", "Tapioca Express", "Burger King", "KFC"], ["KFC", "Shogun", "Burger King"], ["Shogun"])],
)
def test_find_restaurant(l1: List[str], l2: List[str], expected: List[str]):
    actual = solution.find_restaurant(l1, l2)
    assert expected == actual


@pytest.mark.parametrize(
    "root, val, depth, expected",
    [(create_treenode_iteratively([4, 2, 6, 3, 1, 5]), 1, 2,
      create_treenode_iteratively([4, 1, 1, 2, None, None, 6, 3, 1, 5])),
     (create_treenode_iteratively([4, 2, None, 3, 1]), 1, 3,
      create_treenode_iteratively([4, 2, None, 1, 1, 3, None, None, 1]))]
)
def test_add_one_row(root: Optional[TreeNode], val: int, depth: int, expected: Optional[TreeNode]):
    actual = solution.add_one_row(root, val, depth)
    assert expected == actual


@pytest.mark.parametrize(
    "pairs, expected",
    [([[1, 2], [2, 3], [3, 4]], 2), ([[1, 2], [7, 8], [4, 5]], 3)]
)
def test_find_longest_chain(pairs: List[List[int]], expected: int):
    actual = solution.find_longest_chain(pairs)
    assert expected == actual


@pytest.mark.parametrize(
    "root, expected",
    [(create_treenode_iteratively([1, 2, 3, 4, None, 2, 4, None, None, 4]),
      [create_treenode_iteratively([4]), create_treenode_iteratively([2, 4])
       ]),
     (create_treenode_iteratively([2, 1, 1]), [create_treenode_iteratively([1])]),
     (create_treenode_iteratively([2, 2, 2, 3, None, 3, None]),
      [create_treenode_iteratively([3]),
       create_treenode_iteratively([2, 3])]),
     ])
def test_find_duplicate_subtrees(root: Optional[TreeNode], expected: List[Optional[TreeNode]]):
    actual = solution.find_duplicate_subtrees(root)
    assert expected == actual


@pytest.mark.parametrize(
    "root, expected",
    [(create_treenode_iteratively([1, 2]), [["", "1", ""], ["2", "", ""]]),
     (create_treenode_iteratively([1, 2, 3, None, 4]), [["", "", "", "1", "", "", ""],
                                                        ["", "2", "", "", "", "3", ""],
                                                        ["", "", "4", "", "", "", ""]]
      )])
def test_print_tree(root: Optional[TreeNode], expected: List[List[str]]):
    actual = solution.print_tree(root)
    assert expected == actual


@pytest.mark.parametrize(
    "arr, k, x, expected",
    [([1, 2, 3, 4, 5], 4, 3, [1, 2, 3, 4]), ([1, 2, 3, 4, 5], 4, -1, [1, 2, 3, 4])],
)
def test_find_closest_elements(arr: List[int], k: int, x: int, expected: List[int]):
    actual = solution.find_closest_elements(arr, k, x)
    assert expected == actual


@pytest.mark.parametrize(
    "root, expected",
    [(create_treenode_iteratively([1, 3, 2, 5, 3, None, 9]), 4),
     (create_treenode_iteratively([1, 3, 2, 5, None, None, 9, 6, None, 7]), 7),
     (create_treenode_iteratively([1, 3, 2, 5]), 2)],
)
def test_width_of_binary_tree(root: Optional[TreeNode], expected: int):
    actual = solution.width_of_binary_tree(root)
    assert expected == actual


@pytest.mark.parametrize(
    "root, expected",
    [(create_treenode_iteratively([5, 4, 5, 1, 1, None, 5]), 2),
     (create_treenode_iteratively([1, 4, 5, 4, 4, None, 5]), 2)])
def test_longest_univalue_path(root: Optional[TreeNode], expected: int):
    actual = solution.longest_univalue_path(root)
    assert expected == actual


@pytest.mark.parametrize(
    "left, right, expected",
    [(1, 22, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]), (47, 85, [48, 55, 66, 77])],
)
def test_self_dividing_number(left: int, right: int, expected: List[int]):
    actual = solution.self_dividing_number(left, right)
    assert expected == actual


@pytest.mark.parametrize(
    "letters, target, expected",
    [(["c", "f", "j"], "a", "c"), (["c", "f", "j"], "c", "f"), (["c", "f", "j"], "d", "f")],
)
def test_next_greatest_letter(letters: List[str], target: str, expected: str):
    actual = solution.next_greatest_letter(letters, target)
    assert expected == actual


@pytest.mark.parametrize(
    "k, expected",
    [(0, 5), (5, 0), (3, 5)],
)
def test_preimage_size_fzf(k: int, expected: int):
    actual = solution.preimage_size_fzf(k)
    assert expected == actual


@pytest.mark.parametrize(
    "words, expected",
    [(["gin", "zen", "gig", "msg"], 2), (["a"], 1)],
)
def test_unique_morse_representations(words: List[str], expected: int):
    actual = solution.unique_morse_representations(words)
    assert expected == actual


@pytest.mark.parametrize(
    "widths, s, expected",
    [([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
      "abcdefghijklmnopqrstuvwxyz",
      [3, 60]), (
             [4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             "bbbcccdddaaa", [2, 4])],
)
def test_number_of_lines(widths: List[int], s: str, expected: List[int]):
    actual = solution.number_of_lines(widths, s)
    assert expected == actual


@pytest.mark.parametrize(
    "s, expected",
    [("ABC", 10), ("ABA", 8), ("LEETCODE", 92)],
)
def test_unique_letter_string(s: str, expected: int):
    actual = solution.unique_letter_string(s)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [([1, 2, 3, 4, 5, 6], [4, 5, 6]), ([1, 2, 3, 4, 5], [3, 4, 5])],
)
def test_middle_node(test_input: List[int], expected: List[int]):
    actual = list_node_to_list(solution.middle_node(create_list_node(test_input)))
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [([[1, 2], [3, 4]], 17), ([[2]], 5), ([[1, 0], [0, 2]], 8)],
)
def test_projection_area(test_input: List[List[int]], expected: int):
    actual = solution.projection_area(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
        ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
    ],
)
def test_sorted_squares(test_input: List[int], expected: List[int]):
    actual = solution.sorted_squares(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([4, 3, 10, 9, 8], [10, 9]),
        ([4, 4, 7, 6, 7], [7, 7, 6]),
        ([6], [6]),
    ],
)
def test_min_subsequence(test_input: List[int], expected: List[int]):
    actual = solution.min_subsequence(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("a0b1c2", ["0a1b2c", "a0b1c2", "0a1b2c", "0c2a1b"]),
        ("leetcode", [""]),
        ("1229857369", [""]),
    ],
)
def test_reformat(test_input: str, expected: List[str]):
    actual = solution.reformat(test_input)
    assert actual in expected


@pytest.mark.parametrize(
    "start_time, end_time, query_time, expected",
    [
        ([1, 2, 3], [3, 2, 7], 4, 1),
        ([4], [4], 4, 1),
    ],
)
def test_busy_student(start_time: List[int], end_time: List[int], query_time: int, expected: int):
    actual = solution.busy_student(start_time, end_time, query_time)
    assert actual == expected


@pytest.mark.parametrize(
    "sentence, search_word, expected",
    [
        ("i love eating burger", "burg", 4),
        ("this problem is an easy problem", "pro", 2),
        ("i am tired", "you", -1),
    ],
)
def test_is_prefix_of_word(sentence: str, search_word: str, expected: int):
    actual = solution.is_prefix_of_word(sentence, search_word)
    assert actual == expected


@pytest.mark.parametrize(
    "target, arr, expected",
    [
        ([1, 2, 3, 4], [2, 4, 1, 3], True),
        ([7], [7], True),
        ([3, 7, 9], [3, 7, 11], False),
    ],
)
def test_can_be_equal(target: List[int], arr: List[int], expected: bool):
    actual = solution.can_be_equal(target, arr)
    assert actual == expected


@pytest.mark.parametrize(
    "nums, n, expected",
    [
        ([2, 5, 1, 3, 4, 7], 3, [2, 3, 5, 4, 1, 7]),
        ([1, 2, 3, 4, 4, 3, 2, 1], 4, [1, 4, 2, 3, 3, 2, 4, 1]),
        ([1, 1, 2, 2], 2, [1, 2, 1, 2]),
    ],
)
def test_shuffle(nums: List[int], n: int, expected: List[int]):
    actual = solution.shuffle(nums, n)
    assert actual == expected


@pytest.mark.parametrize(
    "prices, expected",
    [
        ([8, 4, 6, 2, 3], [4, 2, 4, 2, 3]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([10, 1, 1, 6], [9, 0, 1, 6]),
    ],
)
def test_final_prices(prices: List[int], expected: List[int]):
    actual = solution.final_prices(prices)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 5, 1], True),
        ([1, 2, 4], False),
    ],
)
def test_can_make_arithmetic_progression(test_input: List[int], expected: bool):
    actual = solution.can_make_arithmetic_progression(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "root, val, expected",
    [
        (create_treenode_iteratively([4, 1, 3, None, None, 2]), 5,
         create_treenode_iteratively([5, 4, None, 1, 3, None, None, 2])),
        (create_treenode_iteratively([5, 2, 4, None, 1]), 3,
         create_treenode_iteratively([5, 2, 4, None, 1, None, 3])),
        (create_treenode_iteratively([5, 2, 3, None, 1]), 4,
         create_treenode_iteratively([5, 2, 4, None, 1, 3])),
    ],
)
def test_insert_into_max_tree(root: Optional[TreeNode], val: int, expected: Optional[TreeNode]):
    actual = solution.insert_into_max_tree(root, val)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("lee(t(c)o)de)", "lee(t(c)o)de"),
        ("a)b(c)d", "ab(c)d"),
        ("))((", ""),
    ],
)
def test_min_remove_to_make_valid(test_input: str, expected: str):
    actual = solution.min_remove_to_make_valid(test_input)
    assert expected == actual


@pytest.mark.parametrize(
    "mat, expected",
    [
        ([[1, 0, 0], [0, 0, 1], [1, 0, 0]], 1),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),
    ],
)
def test_num_special(mat: List[List[int]], expected: int):
    actual = solution.num_special(mat)
    assert expected == actual


@pytest.mark.parametrize(
    "s, expected",
    [
        ("  this   is  a sentence ", "this   is   a   sentence"),
        (" practice   makes   perfect", "practice   makes   perfect "),
    ],
)
def test_reorder_spaces(s: str, expected: str):
    actual = solution.reorder_spaces(s)
    assert expected == actual


@pytest.mark.parametrize(
    "n, k, expected",
    [
        (5, 2, 3),
        (6, 5, 1),
    ],
)
def test_find_the_winner(n: int, k: int, expected: int):
    actual = solution.find_the_winner(n, k)
    assert expected == actual


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([2, 3, -1, 8, 4], 3),
        ([1, -1, 4], 2),
        ([2, 5], -1),
    ],
)
def test_pivot_index(test_input: List[int], expected: int):
    actual = solution.pivot_index(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, k, expected",
    [
        ([1, 2, 2, 1], 1, 4),
        ([1, 3], 3, 0),
        ([3, 2, 1, 5, 4], 2, 3),
    ],
)
def test_count_k_difference(test_input: List[int], k: int, expected: int):
    actual = solution.count_k_difference(test_input, k)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([7, 1, 5, 4], 4),
        ([9, 4, 3, 2], -1),
        ([1, 5, 2, 10], 9),
    ],
)
def test_maximum_difference(test_input: List[int], expected: int):
    actual = solution.maximum_difference(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 1, 4, 2], [2, 4, 1, 3]),
        ([0], [0]),
    ],
)
def test_sort_array_by_parity(test_input: List[int], expected: List[int]):
    actual = solution.sort_array_by_parity(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("IDID", [0, 4, 1, 3, 2]),
        ("III", [0, 1, 2, 3]),
        ("DDI", [3, 2, 0, 1]),
    ],
)
def test_di_string_match(s: str, expected: List[int]):
    actual = solution.di_string_match(s)
    assert actual == expected


@pytest.mark.parametrize(
    "strs, expected",
    [
        (["cba", "daf", "ghi"], 1),
        (["a", "b"], 0),
        (["zyx", "wvu", "tsr"], 3),
    ],
)
def test_min_deletion_size(strs: List[str], expected: int):
    actual = solution.min_deletion_size(strs)
    assert actual == expected


@pytest.mark.parametrize(
    "pushed, popped, expected",
    [
        ([1, 2, 3, 4, 5], [4, 5, 3, 2, 1], True),
        ([1, 2, 3, 4, 5], [4, 3, 5, 1, 2], False),
    ],
)
def test_validate_stack_sequences(pushed: List[int], popped: List[int], expected: bool):
    actual = solution.validate_stack_sequences(pushed, popped)
    assert expected == actual


@pytest.mark.parametrize(
    "nums, expected",
    [
        ([1, 2, 3, 3], 3),
        ([2, 1, 2, 5, 3, 2], 2),
        ([5, 1, 5, 2, 5, 3, 5, 4], 5),
    ],
)
def test_repeated_n_times(nums: List[int], expected: int):
    actual = solution.repeated_n_times(nums)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 1], 2),
        ([2, 2, 2], 7),
        ([3, 2, 1, 5], 6),
    ],
)
def test_count_max_or_subsets(test_input: List[int], expected: int):
    actual = solution.count_max_or_subsets(test_input)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, queries, expected",
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
        test_input: str, queries: List[List[int]], expected: List[int]
):
    actual = solution.plates_between_candles(test_input, queries)
    assert actual == expected


@pytest.mark.parametrize(
    "test_input, expected",
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
        test_input: List[List[int]], expected: List[List[int]]
):
    solution.rotate_matrix(test_input)
    assert test_input == expected
