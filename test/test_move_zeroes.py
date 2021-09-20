from solution import Solution


solution = Solution()


def test_only_zero():
    actual = [0]
    solution.moveZeroes(actual)
    expected = [0]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


def test_left_zero():
    actual = [0, 1, 0, 3, 12]
    solution.moveZeroes(actual)
    expected = [1, 3, 12, 0, 0]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]
