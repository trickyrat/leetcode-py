from typing import List

from src.leetcode.interview_solution import InterviewSolution
import pytest


class TestInterviewSolution:
    solution = InterviewSolution()

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

    @pytest.mark.parametrize(
        "s1, s2, expected",
        [
            ("waterbottle", "erbottlewat", True),
            ("aa", "aba", False),
        ],
    )
    def test_is_flipped_string(self, s1: str, s2: str, expected: bool):
        actual = self.solution.is_flipped_string(s1, s2)
        assert actual == expected

    @pytest.mark.parametrize(
        "k, expected",
        [(5, 9)],
    )
    def test_get_kth_magic_number(self, k: int, expected: int):
        actual = self.solution.get_kth_magic_number(k)
        assert actual == expected
