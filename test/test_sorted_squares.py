from solution import Solution

solution = Solution()


def test_input_both_types_should_ok():
    actual = solution.sortedSquares([-4, -1, 0, 3, 10])
    expected = [0, 1, 9, 16, 100]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


def test_input_opposite_number_should_ok():
    actual = solution.sortedSquares([-7, -3, 2, 3, 11])
    expected = [4, 9, 9, 49, 121]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]
