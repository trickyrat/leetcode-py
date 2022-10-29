import pytest
from leetcode.solution import Solution
from leetcode.utils import *


class TestSolution:
    solution = Solution()

    @pytest.mark.parametrize(
        "test_input, target, expected",
        [([2, 7, 11, 15], 9, [0, 1])],
    )
    def test_two_sum(self, test_input: List[int], target: int, expected: List[int]):
        actual = self.solution.two_sum(test_input, target)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input1, test_input2, expected",
        [
            (create_list_node([2, 4, 3]), create_list_node([5, 6, 4]), [7, 0, 8]),
            (create_list_node([0]), create_list_node([0]), [0]),
            (
                    create_list_node([9, 9, 9, 9, 9, 9, 9]),
                    create_list_node([9, 9, 9, 9]),
                    [8, 9, 9, 9, 0, 0, 0, 1],
            ),
        ],
    )
    def test_add_two_numbers(
            self,
            test_input1: Optional[ListNode],
            test_input2: Optional[ListNode],
            expected: List[int],
    ):
        actual = list_node_to_list(
            self.solution.add_two_numbers(test_input1, test_input2)
        )
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected", [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]
    )
    def test_longest_substring_without_repeat(self, test_input: str, expected: int):
        actual = self.solution.longest_substring_without_repeat(test_input)
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
            self, test_input1: List[int], test_input2: List[int], expected: float
    ):
        actual = self.solution.find_median_sorted_arrays(test_input1, test_input2)
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
    def test_longest_palindrome(self, test_input: str, expected: str):
        actual = self.solution.longest_palindrome(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, num_rows, expected",
        [
            ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
            ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
            ("A", 1, "A"),
        ],
    )
    def test_z_convert(self, test_input: str, num_rows: int, expected: str):
        actual = self.solution.z_convert(test_input, num_rows)
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
    def test_reverse_int(self, test_input: int, expected: str):
        actual = self.solution.reverse_int(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "s, expected",
        [("42", 42), ("   -42", -42), ("4193 with words", 4193)],
    )
    def test_my_atoi(self, s: str, expected: int):
        actual = self.solution.my_atoi(s)
        assert actual == expected

    @pytest.mark.parametrize(
        "x, expected",
        [(121, True), (-121, False), (10, False)],
    )
    def test_is_palindrome(self, x: int, expected: bool):
        actual = self.solution.is_palindrome(x)
        assert actual == expected

    @pytest.mark.parametrize(
        "s, p, expected", [("aa", "a", False), ("aa", "a*", True), ("aa", ".*", True)]
    )
    def test_is_match(self, s: str, p: str, expected: bool):
        actual = self.solution.is_match(s, p)
        assert expected == actual

    @pytest.mark.parametrize(
        "height, expected", [([1, 8, 6, 2, 5, 4, 8, 3, 7], 49), ([1, 1], 1)]
    )
    def test_max_area(self, height: List[int], expected: int):
        actual = self.solution.max_area(height)
        assert expected == actual

    @pytest.mark.parametrize(
        "num, expected", [(3, "III"), (58, "LVIII"), (1994, "MCMXCIV")]
    )
    def test_int_to_roman(self, num: int, expected: str):
        actual = self.solution.int_to_roman(num)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected", [("III", 3), ("LVIII", 58), ("MCMXCIV", 1994)]
    )
    def test_roman_to_int(self, s: str, expected: int):
        actual = self.solution.roman_to_int(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "strs, expected",
        [(["flower", "flow", "flight"], "fl"), (["dog", "racecar", "car"], "")],
    )
    def test_longest_common_prefix(self, strs: List[str], expected: str):
        actual = self.solution.longest_common_prefix(strs)
        assert expected == actual

    @pytest.mark.parametrize(
        "strs, expected",
        [
            (
                    ["eat", "tea", "tan", "ate", "nat", "bat"],
                    [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]],
            ),
            ([""], [[""]]),
            (["a"], [["a"]]),
        ],
    )
    def test_group_anagrams(self, strs: List[str], expected: List[List[str]]):
        actual = self.solution.group_anagrams(strs)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
            ([[1, 4], [4, 5]], [[1, 5]]),
        ],
    )
    def test_merge(self, test_input: List[List[int]], expected: List[List[int]]):
        actual = self.solution.merge(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
            (
                    [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]],
                    [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]],
            ),
        ],
    )
    def test_set_zeroes(self, test_input: List[List[int]], expected: List[List[int]]):
        self.solution.set_zeroes(test_input)
        assert expected == test_input

    @pytest.mark.parametrize(
        "test_input, target, expected",
        [
            ([4, 5, 6, 7, 0, 1, 2], 0, 4),
            ([4, 5, 6, 7, 0, 1, 2], 3, -1),
            ([1], 0, -1),
        ],
    )
    def test_eval(self, test_input: List[int], target: int, expected: int):
        actual = self.solution.search(test_input, target)
        assert expected == actual

    @pytest.mark.parametrize(
        "matrix, target, expected",
        [
            ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 3, True),
            ([[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]], 13, False),
        ],
    )
    def test_search_matrix(self, matrix: List[List[int]], target: int, expected: bool):
        actual = self.solution.search_matrix(matrix, target)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            (create_treenode([3, 9, 20, None, None, 15, 7]), 2),
            (create_treenode([2, None, 3, None, 4, None, 5, None, 6]), 5),
        ],
    )
    def test_min_depth(self, test_input: TreeNode, expected: int):
        actual = self.solution.min_depth(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, target_num, expected",
        [
            (
                    create_treenode([5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]),
                    22,
                    [[5, 4, 11, 2], [5, 8, 4, 5]],
            ),
            (create_treenode([1, 2, 3]), 5, []),
        ],
    )
    def test_path_sum(
            self, test_input: TreeNode, target_num: int, expected: List[List[int]]
    ):
        actual = self.solution.path_sum(test_input, target_num)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, target, expected",
        [([2, 7, 11, 15], 9, [1, 2]), ([2, 3, 4], 6, [1, 3]), ([-1, 0], -1, [1, 2])],
    )
    def test_two_sum_ii(self, test_input: List[int], target: int, expected: List[int]):
        actual = self.solution.two_sum_ii(test_input, target)
        assert actual == expected

    @pytest.mark.parametrize(
        "actual, k, expected",
        [
            ([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]),
            ([-1, -100, 3, 99], 2, [3, 99, -1, -100]),
        ],
    )
    def test_rotate_array(self, actual: List[int], k: int, expected: List[int]):
        self.solution.rotate(actual, k)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected", [([0], [0]), ([0, 1, 0, 3, 12], [1, 3, 12, 0, 0])]
    )
    def test_move_zeroes(self, test_input: List[int], expected: List[int]):
        self.solution.move_zeroes(test_input)
        assert test_input == expected

    @pytest.mark.parametrize(
        "test_input, expected", [([1, 3, 4, 2, 2], 2), ([3, 1, 3, 4, 2], 3)]
    )
    def test_find_duplicate(self, test_input: List[int], expected: int):
        actual = self.solution.find_duplicate(test_input)
        assert expected == actual

    @pytest.mark.parametrize("test_input, expected", [(2, 91), (0, 1)])
    def test_count_numbers_with_unique_digits(self, test_input: int, expected: int):
        actual = self.solution.count_numbers_with_unique_digits(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected", [([1, 9], 10), ([-1, 9], 8), ([1, 2], 3)]
    )
    def test_get_sum(self, test_input: List[int], expected: int):
        actual = self.solution.get_sum(test_input[0], test_input[1])
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [(13, [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]), (2, [1, 2])],
    )
    def test_lexical_order(self, test_input: int, expected: List[int]):
        actual = self.solution.lexical_order(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected", [([197, 130, 1], True), ([235, 140, 4], False)]
    )
    def test_valid_utf8(self, test_input: List[int], expected: bool):
        actual = self.solution.valid_utf8(test_input)
        assert expected == actual

    @pytest.mark.parametrize("test_input, expected", [("Hello, my name is John", 5)])
    def test_count_segment(self, test_input: str, expected: int):
        actual = self.solution.count_segment(test_input)
        assert expected == actual

    @pytest.mark.parametrize("p, expected", [("a", 1), ("cac", 2), ("zab", 6)])
    def test_find_substring_wraparound_string(self, p: str, expected: int):
        actual = self.solution.find_substring_wraparound_string(p)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 7, 5, 3, 6, 8, 9]),
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ],
    )
    def test_find_diagonal_order(
            self, test_input: List[List[int]], expected: List[int]
    ):
        actual = self.solution.find_diagonal_order(test_input)
        assert actual == expected

    @pytest.mark.parametrize("test_input, expected", [(100, "202"), (-7, "-10")])
    def test_convert_to_base7(self, test_input: int, expected: int):
        actual = self.solution.convert_to_base7(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "a, b, expected", [("aba", "cdc", 3), ("aaa", "bbb", 3), ("aaa", "aaa", -1)]
    )
    def test_find_lus_length(self, a: str, b: str, expected: int):
        actual = self.solution.find_lus_length(a, b)
        assert expected == actual

    @pytest.mark.parametrize(
        "input1, input2, expected",
        [("1+1i", "1+1i", "0+2i"), ("1+-1i", "1+-1i", "0+-2i")],
    )
    def test_complex_number_multiply(self, input1: str, input2: str, expected: str):
        actual = self.solution.complex_number_multiply(input1, input2)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [([1000, 100, 10, 2], "1000/(100/10/2)")],
    )
    def test_optimal_division(self, test_input: List[int], expected: str):
        actual = self.solution.optimal_division(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "l1, l2, expected",
        [
            (
                    ["Shogun", "Tapioca Express", "Burger King", "KFC"],
                    [
                        "Piatti",
                        "The Grill at Torrey Pines",
                        "Hungry Hunter Steakhouse",
                        "Shogun",
                    ],
                    ["Shogun"],
            ),
            (
                    ["Shogun", "Tapioca Express", "Burger King", "KFC"],
                    ["KFC", "Shogun", "Burger King"],
                    ["Shogun"],
            ),
        ],
    )
    def test_find_restaurant(self, l1: List[str], l2: List[str], expected: List[str]):
        actual = self.solution.find_restaurant(l1, l2)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, val, depth, expected",
        [
            (
                    create_treenode([4, 2, 6, 3, 1, 5]),
                    1,
                    2,
                    create_treenode([4, 1, 1, 2, None, None, 6, 3, 1, 5]),
            ),
            (
                    create_treenode([4, 2, None, 3, 1]),
                    1,
                    3,
                    create_treenode([4, 2, None, 1, 1, 3, None, None, 1]),
            ),
        ],
    )
    def test_add_one_row(
            self,
            root: Optional[TreeNode],
            val: int,
            depth: int,
            expected: Optional[TreeNode],
    ):
        actual = self.solution.add_one_row(root, val, depth)
        assert expected == actual

    @pytest.mark.parametrize(
        "pairs, expected",
        [([[1, 2], [2, 3], [3, 4]], 2), ([[1, 2], [7, 8], [4, 5]], 3)],
    )
    def test_find_longest_chain(self, pairs: List[List[int]], expected: int):
        actual = self.solution.find_longest_chain(pairs)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, expected",
        [
            (
                    create_treenode([1, 2, 3, 4, None, 2, 4, None, None, 4]),
                    [create_treenode([2, 4]), create_treenode([4])],
            ),
            (create_treenode([2, 1, 1]), [create_treenode([1])]),
            (
                    create_treenode([2, 2, 2, 3, None, 3, None]),
                    [create_treenode([3]), create_treenode([2, 3])],
            ),
        ],
    )
    def test_find_duplicate_subtrees(
            self, root: Optional[TreeNode], expected: List[Optional[TreeNode]]
    ):
        actual = self.solution.find_duplicate_subtrees(root)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, expected",
        [
            (create_treenode([1, 2]), [["", "1", ""], ["2", "", ""]]),
            (
                    create_treenode([1, 2, 3, None, 4]),
                    [
                        ["", "", "", "1", "", "", ""],
                        ["", "2", "", "", "", "3", ""],
                        ["", "", "4", "", "", "", ""],
                    ],
            ),
        ],
    )
    def test_print_tree(self, root: Optional[TreeNode], expected: List[List[str]]):
        actual = self.solution.print_tree(root)
        assert expected == actual

    @pytest.mark.parametrize(
        "arr, k, x, expected",
        [([1, 2, 3, 4, 5], 4, 3, [1, 2, 3, 4]), ([1, 2, 3, 4, 5], 4, -1, [1, 2, 3, 4])],
    )
    def test_find_closest_elements(
            self, arr: List[int], k: int, x: int, expected: List[int]
    ):
        actual = self.solution.find_closest_elements(arr, k, x)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, expected",
        [
            (create_treenode([1, 3, 2, 5, 3, None, 9]), 4),
            (create_treenode([1, 3, 2, 5, None, None, 9, 6, None, 7]), 7),
            (create_treenode([1, 3, 2, 5]), 2),
        ],
    )
    def test_width_of_binary_tree(self, root: Optional[TreeNode], expected: int):
        actual = self.solution.width_of_binary_tree(root)
        assert expected == actual

    @pytest.mark.parametrize(
        "n, k, expected",
        [(3, 1, [1, 2, 3]), (3, 2, [1, 3, 2])],
    )
    def test_construct_array(self, n: int, k: int, expected: List[int]):
        actual = self.solution.construct_array(n, k)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, low, high, expected",
        [
            (create_treenode([1, 0, 2]), 1, 2, create_treenode([1, None, 2])),
            (
                    create_treenode([3, 0, 4, None, 2, None, None, 1]),
                    1,
                    3,
                    create_treenode([3, 2, None, 1]),
            ),
        ],
    )
    def test_trim_bst(
            self,
            root: Optional[TreeNode],
            low: int,
            high: int,
            expected: Optional[TreeNode],
    ):
        actual = self.solution.trim_bst(root, low, high)
        assert expected == actual

    @pytest.mark.parametrize("num, expected", [(2736, 7236), (9973, 9973)])
    def test_maximum_swap(self, num: int, expected: int):
        actual = self.solution.maximum_swap(num)
        assert expected == actual

    @pytest.mark.parametrize(
        "n, presses, expected",
        [
            (1, 1, 2),
            (2, 1, 3),
            (3, 1, 4),
        ],
    )
    def test_flip_lights(self, n: int, presses: int, expected: int):
        actual = self.solution.flip_lights(n, presses)
        assert expected == actual

    @pytest.mark.parametrize(
        "root, expected",
        [
            (create_treenode([5, 4, 5, 1, 1, None, 5]), 2),
            (create_treenode([1, 4, 5, 4, 4, None, 5]), 2),
        ],
    )
    def test_longest_univalue_path(self, root: Optional[TreeNode], expected: int):
        actual = self.solution.longest_univalue_path(root)
        assert expected == actual

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            (1, 22, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]),
            (47, 85, [48, 55, 66, 77]),
        ],
    )
    def test_self_dividing_number(self, left: int, right: int, expected: List[int]):
        actual = self.solution.self_dividing_number(left, right)
        assert expected == actual

    @pytest.mark.parametrize(
        "letters, target, expected",
        [
            (["c", "f", "j"], "a", "c"),
            (["c", "f", "j"], "c", "f"),
            (["c", "f", "j"], "d", "f"),
        ],
    )
    def test_next_greatest_letter(self, letters: List[str], target: str, expected: str):
        actual = self.solution.next_greatest_letter(letters, target)
        assert expected == actual

    @pytest.mark.parametrize(
        "arr, expected",
        [([4, 3, 2, 1, 0], 1), ([1, 0, 2, 3, 4], 4)],
    )
    def test_max_chunks_to_sorted(self, arr: List[int], expected: int):
        actual = self.solution.max_chunks_to_sorted(arr)
        assert expected == actual

    @pytest.mark.parametrize(
        "n, k, expected",
        [(1, 1, 0), (2, 1, 0), (2, 2, 1)],
    )
    def test_kth_grammar(self, n: int, k: int, expected: int):
        actual = self.solution.kth_grammar(n, k)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected",
        [("a1b2", ["a1b2", "a1B2", "A1b2", "A1B2"]), ("3z4", ["3z4", "3Z4"])],
    )
    def test_letter_case_permutation(self, s: str, expected: List[str]):
        actual = self.solution.letter_case_permutation(s)
        actual.sort()
        expected.sort()
        assert expected == actual

    @pytest.mark.parametrize(
        "k, expected",
        [(0, 5), (5, 0), (3, 5)],
    )
    def test_preimage_size_fzf(self, k: int, expected: int):
        actual = self.solution.preimage_size_fzf(k)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums1, nums2, expected",
        [
            ([1, 3, 5, 4], [1, 2, 3, 7], 1),
            ([0, 3, 5, 8, 9], [2, 1, 4, 6, 9], 1),
        ],
    )
    def test_min_swap(self, nums1: List[int], nums2: List[int], expected: int):
        actual = self.solution.min_swap(nums1, nums2)
        assert expected == actual

    @pytest.mark.parametrize(
        "words, expected",
        [(["gin", "zen", "gig", "msg"], 2), (["a"], 1)],
    )
    def test_unique_morse_representations(self, words: List[str], expected: int):
        actual = self.solution.unique_morse_representations(words)
        assert expected == actual

    @pytest.mark.parametrize(
        "widths, s, expected",
        [
            (
                    [
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                    ],
                    "abcdefghijklmnopqrstuvwxyz",
                    [3, 60],
            ),
            (
                    [
                        4,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                        10,
                    ],
                    "bbbcccdddaaa",
                    [2, 4],
            ),
        ],
    )
    def test_number_of_lines(self, widths: List[int], s: str, expected: List[int]):
        actual = self.solution.number_of_lines(widths, s)
        assert expected == actual

    @pytest.mark.parametrize(
        "cpdomains, expected",
        [
            (
                    ["9001 discuss.leetcode.com"],
                    {"9001 leetcode.com", "9001 discuss.leetcode.com", "9001 com"},
            ),
            (
                    [
                        "900 google.mail.com",
                        "50 yahoo.com",
                        "1 intel.mail.com",
                        "5 wiki.org",
                    ],
                    {
                        "901 mail.com",
                        "50 yahoo.com",
                        "900 google.mail.com",
                        "5 wiki.org",
                        "5 org",
                        "1 intel.mail.com",
                        "951 com",
                    },
            ),
        ],
    )
    def test_subdomain_in_visits(self, cpdomains: List[str], expected: set[str]):
        actual = set(self.solution.subdomain_visits(cpdomains))
        assert expected == actual

    @pytest.mark.parametrize(
        "head, nums, expected",
        [
            (create_list_node([0, 1, 2, 3]), [0, 1, 3], 2),
            (create_list_node([0, 1, 2, 3, 4]), [0, 3, 1, 4], 2),
        ],
    )
    def test_num_components(
            self, head: Optional[ListNode], nums: List[int], expected: int
    ):
        actual = self.solution.num_components(head, nums)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected",
        [("ABC", 10), ("ABA", 8), ("LEETCODE", 92)],
    )
    def test_unique_letter_string(self, s: str, expected: int):
        actual = self.solution.unique_letter_string(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected",
        [("()", 1), ("(())", 2), ("()()", 2)],
    )
    def test_score_of_parentheses(self, s: str, expected: int):
        actual = self.solution.score_of_parentheses(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "quality, wage, k, expected",
        [
            ([10, 20, 5], [70, 50, 30], 2, 105.00000),
            ([3, 1, 10, 10, 1], [4, 8, 2, 2, 7], 3, 30.66667),
        ],
    )
    def test_min_cost_to_hire_worker(
            self, quality: List[int], wage: List[int], k: int, expected: float
    ):
        actual = self.solution.min_cost_to_hire_worker(quality, wage, k)
        actual = round(actual, 5)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, k, expected",
        [
            ([1], 1, 1),
            ([1, 2], 4, -1),
            ([2, -1, 2], 3, 3),
        ],
    )
    def test_min_shortest_subarray(self, nums: List[int], k: int, expected: int):
        actual = self.solution.shortest_subarray(nums, k)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums1, nums2, expected",
        [
            ([2, 7, 11, 15], [1, 10, 4, 11], [2, 11, 7, 15]),
            ([12, 24, 8, 32], [13, 25, 32, 11], [24, 32, 8, 12]),
        ],
    )
    def test_advantage_count(
            self, nums1: List[int], nums2: List[int], expected: List[int]
    ):
        actual = self.solution.advantage_count(nums1, nums2)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [([1, 2, 3, 4, 5, 6], [4, 5, 6]), ([1, 2, 3, 4, 5], [3, 4, 5])],
    )
    def test_middle_node(self, test_input: List[int], expected: List[int]):
        actual = list_node_to_list(
            self.solution.middle_node(create_list_node(test_input))
        )
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [([[1, 2], [3, 4]], 17), ([[2]], 5), ([[1, 0], [0, 2]], 8)],
    )
    def test_projection_area(self, test_input: List[List[int]], expected: int):
        actual = self.solution.projection_area(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
            ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
        ],
    )
    def test_sorted_squares(self, test_input: List[int], expected: List[int]):
        actual = self.solution.sorted_squares(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([4, 3, 10, 9, 8], [10, 9]),
            ([4, 4, 7, 6, 7], [7, 7, 6]),
            ([6], [6]),
        ],
    )
    def test_min_subsequence(self, test_input: List[int], expected: List[int]):
        actual = self.solution.min_subsequence(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ("a0b1c2", ["0a1b2c", "a0b1c2", "0a1b2c", "0c2a1b"]),
            ("leetcode", [""]),
            ("1229857369", [""]),
        ],
    )
    def test_reformat(self, test_input: str, expected: List[str]):
        actual = self.solution.reformat(test_input)
        assert actual in expected

    @pytest.mark.parametrize(
        "target, n, expected",
        [
            ([1, 3], 3, ["Push", "Push", "Pop", "Push"]),
            ([1, 2, 3], 3, ["Push", "Push", "Push"]),
            ([1, 2], 4, ["Push", "Push"]),
        ],
    )
    def test_build_array(self, target: List[int], n: int, expected: int):
        actual = self.solution.build_array(target, n)
        assert actual == expected

    @pytest.mark.parametrize(
        "start_time, end_time, query_time, expected",
        [
            ([1, 2, 3], [3, 2, 7], 4, 1),
            ([4], [4], 4, 1),
        ],
    )
    def test_busy_student(
            self, start_time: List[int], end_time: List[int], query_time: int, expected: int
    ):
        actual = self.solution.busy_student(start_time, end_time, query_time)
        assert actual == expected

    @pytest.mark.parametrize(
        "sentence, search_word, expected",
        [
            ("i love eating burger", "burg", 4),
            ("this problem is an easy problem", "pro", 2),
            ("i am tired", "you", -1),
        ],
    )
    def test_is_prefix_of_word(self, sentence: str, search_word: str, expected: int):
        actual = self.solution.is_prefix_of_word(sentence, search_word)
        assert actual == expected

    @pytest.mark.parametrize(
        "target, arr, expected",
        [
            ([1, 2, 3, 4], [2, 4, 1, 3], True),
            ([7], [7], True),
            ([3, 7, 9], [3, 7, 11], False),
        ],
    )
    def test_can_be_equal(self, target: List[int], arr: List[int], expected: bool):
        actual = self.solution.can_be_equal(target, arr)
        assert actual == expected

    @pytest.mark.parametrize(
        "nums, n, expected",
        [
            ([2, 5, 1, 3, 4, 7], 3, [2, 3, 5, 4, 1, 7]),
            ([1, 2, 3, 4, 4, 3, 2, 1], 4, [1, 4, 2, 3, 3, 2, 4, 1]),
            ([1, 1, 2, 2], 2, [1, 2, 1, 2]),
        ],
    )
    def test_shuffle(self, nums: List[int], n: int, expected: List[int]):
        actual = self.solution.shuffle(nums, n)
        assert actual == expected

    @pytest.mark.parametrize(
        "prices, expected",
        [
            ([8, 4, 6, 2, 3], [4, 2, 4, 2, 3]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
            ([10, 1, 1, 6], [9, 0, 1, 6]),
        ],
    )
    def test_final_prices(self, prices: List[int], expected: List[int]):
        actual = self.solution.final_prices(prices)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([3, 5, 1], True),
            ([1, 2, 4], False),
        ],
    )
    def test_can_make_arithmetic_progression(
            self, test_input: List[int], expected: bool
    ):
        actual = self.solution.can_make_arithmetic_progression(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "root, val, expected",
        [
            (
                    create_treenode([4, 1, 3, None, None, 2]),
                    5,
                    create_treenode([5, 4, None, 1, 3, None, None, 2]),
            ),
            (
                    create_treenode([5, 2, 4, None, 1]),
                    3,
                    create_treenode([5, 2, 4, None, 1, None, 3]),
            ),
            (
                    create_treenode([5, 2, 3, None, 1]),
                    4,
                    create_treenode([5, 2, 4, None, 1, 3]),
            ),
        ],
    )
    def test_insert_into_max_tree(
            self, root: Optional[TreeNode], val: int, expected: Optional[TreeNode]
    ):
        actual = self.solution.insert_into_max_tree(root, val)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ("lee(t(c)o)de)", "lee(t(c)o)de"),
            ("a)b(c)d", "ab(c)d"),
            ("))((", ""),
        ],
    )
    def test_min_remove_to_make_valid(self, test_input: str, expected: str):
        actual = self.solution.min_remove_to_make_valid(test_input)
        assert expected == actual

    @pytest.mark.parametrize(
        "mat, expected",
        [
            ([[1, 0, 0], [0, 0, 1], [1, 0, 0]], 1),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 3),
        ],
    )
    def test_num_special(self, mat: List[List[int]], expected: int):
        actual = self.solution.num_special(mat)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("  this   is  a sentence ", "this   is   a   sentence"),
            (" practice   makes   perfect", "practice   makes   perfect "),
        ],
    )
    def test_reorder_spaces(self, s: str, expected: str):
        actual = self.solution.reorder_spaces(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "logs, expected",
        [
            (["d1/", "d2/", "../", "d21/", "./"], 2),
            (["d1/", "d2/", "./", "d3/", "../", "d31/"], 3),
            (["d1/", "../", "../", "../"], 0),
        ],
    )
    def test_min_operations(self, logs: List[str], expected: int):
        actual = self.solution.min_operations(logs)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected", [([3, 5], 2), ([0, 0], -1), ([0, 4, 3, 0, 4], 3)]
    )
    def test_special_array(self, nums: List[int], expected: int):
        actual = self.solution.special_array(nums)
        assert expected == actual

    @pytest.mark.parametrize(
        "arr, expected",
        [
            ([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3], 2.00000),
            ([6, 2, 7, 5, 1, 2, 0, 3, 10, 2, 5, 0, 5, 5, 0, 8, 7, 6, 8, 0], 4.00000),
            (
                    [
                        6,
                        0,
                        7,
                        0,
                        7,
                        5,
                        7,
                        8,
                        3,
                        4,
                        0,
                        7,
                        8,
                        1,
                        6,
                        8,
                        1,
                        1,
                        2,
                        4,
                        8,
                        1,
                        9,
                        5,
                        4,
                        3,
                        8,
                        5,
                        10,
                        8,
                        6,
                        6,
                        1,
                        0,
                        6,
                        10,
                        8,
                        2,
                        3,
                        4,
                    ],
                    4.77778,
            ),
        ],
    )
    def test_trim_mean(self, arr: List[int], expected: float):
        actual = self.solution.trim_mean(arr)
        assert actual - expected <= 0.00001

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("aa", 0),
            ("abca", 2),
            ("cbzxy", -1),
        ],
    )
    def test_max_length_between_equal_characters(self, s: str, expected: int):
        actual = self.solution.max_length_between_equal_characters(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([1, 1, 2, 2, 2, 3], [3, 1, 1, 2, 2, 2]),
            ([2, 3, 1, 3, 2], [1, 3, 3, 2, 2]),
            ([-1, 1, -6, 4, 5, -6, 1, 4, 1], [5, -1, 4, 4, -6, -6, 1, 1, 1]),
        ],
    )
    def test_frequency_sort(self, nums: List[int], expected: List[int]):
        actual = self.solution.frequency_sort(nums)
        assert expected == actual

    @pytest.mark.parametrize(
        "number, expected",
        [
            ("1-23-45 6", "123-456"),
            ("123 4-567", "123-45-67"),
            ("123 4-5678", "123-456-78"),
        ],
    )
    def test_reformat_number(self, number: str, expected: str):
        actual = self.solution.reformat_number(number)
        assert expected == actual

    @pytest.mark.parametrize(
        "students, sandwiches, expected",
        [
            ([1, 1, 0, 0], [0, 1, 0, 1], 0),
            ([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1], 3),
        ],
    )
    def test_count_students(
            self, students: List[int], sandwiches: List[int], expected: int
    ):
        actual = self.solution.count_students(students, sandwiches)
        assert expected == actual

    @pytest.mark.parametrize(
        "word1, word2, expected",
        [
            ("abc", "pqr", "apbqcr"),
            ("ab", "pqrs", "apbqrs"),
            ("abcd", "pq", "apbqcd"),
        ],
    )
    def test_merge_alternately(self, word1: str, word2: str, expected: str):
        actual = self.solution.merge_alternately(word1, word2)
        assert expected == actual

    @pytest.mark.parametrize(
        "items, rule_key, rule_value, expected",
        [
            ([["phone", "blue", "pixel"], ["computer", "silver", "lenovo"], ["phone", "gold", "iphone"]], "color",
             "silver", 1),
            (
                    [["phone", "blue", "pixel"], ["computer", "silver", "phone"], ["phone", "gold", "iphone"]], "type",
                    "phone",
                    2),
        ],
    )
    def test_count_matches(self, items: List[List[str]], rule_key: str, rule_value: str, expected: int):
        actual = self.solution.count_matches(items, rule_key, rule_value)
        assert expected == actual

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("1001", False),
            ("110", True),
        ],
    )
    def test_check_ones_segment(self, s: str, expected: bool):
        actual = self.solution.check_ones_segment(s)
        assert expected == actual

    @pytest.mark.parametrize(
        "s1, s2, expected",
        [
            ("bank", "kanb", True),
            ("attack", "defend", False),
            ("kelb", "kelb", True),
        ],
    )
    def test_are_almost_equal(self, s1: str, s2: str, expected: bool):
        actual = self.solution.are_almost_equal(s1, s2)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([10, 20, 30, 5, 10, 50], 65),
            ([10, 20, 30, 40, 50], 150),
            ([12, 17, 15, 13, 10, 11, 12], 33),
        ],
    )
    def test_max_ascending_sum(self, nums: List[int], expected: int):
        actual = self.solution.max_ascending_sum(nums)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([-1, -2, -3, -4, 3, 2, 1], 1),
            ([1, 5, 0, 2, -3], 0),
            ([-1, 1, -1, 1, -1], -1),
        ],
    )
    def test_array_sign(self, nums: List[int], expected: int):
        actual = self.solution.array_sign(nums)
        assert expected == actual

    @pytest.mark.parametrize(
        "n, k, expected",
        [
            (5, 2, 3),
            (6, 5, 1),
        ],
    )
    def test_find_the_winner(self, n: int, k: int, expected: int):
        actual = self.solution.find_the_winner(n, k)
        assert expected == actual

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([2, 3, -1, 8, 4], 3),
            ([1, -1, 4], 2),
            ([2, 5], -1),
        ],
    )
    def test_pivot_index(self, test_input: List[int], expected: int):
        actual = self.solution.pivot_index(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, k, expected",
        [
            ([1, 2, 2, 1], 1, 4),
            ([1, 3], 3, 0),
            ([3, 2, 1, 5, 4], 2, 3),
        ],
    )
    def test_count_k_difference(self, test_input: List[int], k: int, expected: int):
        actual = self.solution.count_k_difference(test_input, k)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([7, 1, 5, 4], 4),
            ([9, 4, 3, 2], -1),
            ([1, 5, 2, 10], 9),
        ],
    )
    def test_maximum_difference(self, test_input: List[int], expected: int):
        actual = self.solution.maximum_difference(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "n, dislikes, expected",
        [
            (4, [[1, 2], [1, 3], [2, 4]], True),
            (3, [[1, 2], [1, 3], [2, 3]], False),
            (5, [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]], False),
        ],
    )
    def test_possible_bipartition(
            self, n: int, dislikes: List[List[int]], expected: bool
    ):
        actual = self.solution.possible_bipartition(n, dislikes)
        assert actual == expected

    @pytest.mark.parametrize(
        "fruits, expected",
        [
            ([1, 2, 1], 3),
            ([0, 1, 2, 2], 3),
            ([1, 2, 3, 2, 2], 4),
        ],
    )
    def test_total_fruit(self, fruits: List[int], expected: int):
        actual = self.solution.total_fruit(fruits)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([3, 1, 4, 2], [2, 4, 1, 3]),
            ([0], [0]),
        ],
    )
    def test_sort_array_by_parity(self, test_input: List[int], expected: List[int]):
        actual = self.solution.sort_array_by_parity(test_input)
        assert actual == expected

    @pytest.mark.parametrize(
        "arr, expected",
        [
            ([3, 1, 2, 4], 17),
            ([11, 81, 94, 43, 3], 444),
        ],
    )
    def test_sum_subarray_mins(self, arr: List[int], expected: int):
        actual = self.solution.sum_subarray_mins(arr)
        assert actual == expected

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([5, 0, 3, 8, 6], 3),
            ([1, 1, 1, 0, 6, 12], 4),
        ],
    )
    def test_partition_disjoint(self, nums: List[int], expected: int):
        actual = self.solution.partition_disjoint(nums)
        assert actual == expected

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("())", 1),
            ("(((", 3),
        ],
    )
    def test_min_add_to_make_valid(self, s: str, expected: int):
        actual = self.solution.min_add_to_make_valid(s)
        assert actual == expected

    @pytest.mark.parametrize(
        "arr, expected",
        [
            ([1, 0, 1, 0, 1], [0, 3]),
            ([1, 1, 0, 1, 1], [-1, -1]),
            ([1, 1, 0, 0, 1], [0, 2]),
        ],
    )
    def test_min_three_equal_parts(self, arr: List[int], expected: List[int]):
        actual = self.solution.three_equal_parts(arr)
        assert actual == expected

    @pytest.mark.parametrize(
        "grid, expected",
        [
            ([[0, 1], [1, 0]], 1),
            ([[0, 1, 0], [0, 0, 0], [0, 0, 1]], 2),
            ([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], 1),
        ],
    )
    def test_shortest_bridge(self, grid: List[List[int]], expected: int):
        actual = self.solution.shortest_bridge(grid)
        assert actual == expected

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("abc", 7),
            ("aba", 6),
            ("aaa", 3),
        ],
    )
    def test_distinct_subseq_ii(self, s: str, expected: int):
        actual = self.solution.distinct_subseq_ii(s)
        assert actual == expected

    @pytest.mark.parametrize(
        "s, expected",
        [
            ("IDID", [0, 4, 1, 3, 2]),
            ("III", [0, 1, 2, 3]),
            ("DDI", [3, 2, 0, 1]),
        ],
    )
    def test_di_string_match(self, s: str, expected: List[int]):
        actual = self.solution.di_string_match(s)
        assert actual == expected

    @pytest.mark.parametrize(
        "strs, expected",
        [
            (["cba", "daf", "ghi"], 1),
            (["a", "b"], 0),
            (["zyx", "wvu", "tsr"], 3),
        ],
    )
    def test_min_deletion_size(self, strs: List[str], expected: int):
        actual = self.solution.min_deletion_size(strs)
        assert actual == expected

    @pytest.mark.parametrize(
        "pushed, popped, expected",
        [
            ([1, 2, 3, 4, 5], [4, 5, 3, 2, 1], True),
            ([1, 2, 3, 4, 5], [4, 3, 5, 1, 2], False),
        ],
    )
    def test_validate_stack_sequences(
            self, pushed: List[int], popped: List[int], expected: bool
    ):
        actual = self.solution.validate_stack_sequences(pushed, popped)
        assert expected == actual

    @pytest.mark.parametrize(
        "nums, expected",
        [
            ([1, 2, 3, 3], 3),
            ([2, 1, 2, 5, 3, 2], 2),
            ([5, 1, 5, 2, 5, 3, 5, 4], 5),
        ],
    )
    def test_repeated_n_times(self, nums: List[int], expected: int):
        actual = self.solution.repeated_n_times(nums)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([3, 1], 2),
            ([2, 2, 2], 7),
            ([3, 2, 1, 5], 6),
        ],
    )
    def test_count_max_or_subsets(self, test_input: List[int], expected: int):
        actual = self.solution.count_max_or_subsets(test_input)
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
            self, test_input: str, queries: List[List[int]], expected: List[int]
    ):
        actual = self.solution.plates_between_candles(test_input, queries)
        assert actual == expected

    @pytest.mark.parametrize(
        "test_input, expected",
        [
            ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
            (
                    [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]],
                    [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]],
            ),
        ],
    )
    def test_rotate(self, test_input: List[List[int]], expected: List[List[int]]):
        self.solution.rotate_matrix(test_input)
        assert test_input == expected
