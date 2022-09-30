from leetcode.interview_solution import InterviewSolution
import pytest


class TestInterviewSolution:
    solution = InterviewSolution()

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
