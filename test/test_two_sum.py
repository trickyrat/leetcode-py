from typing import List
from solution import Solution

solution = Solution()


def test_input_valid_data():
    actual = solution.twoSum([2, 7, 11, 15], 9)
    expect = [0, 1]
    assert len(actual) == len(expect)
    for i in range(0, len(expect)):
        assert actual[i] == expect[i]
