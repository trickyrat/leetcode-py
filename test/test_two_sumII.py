from solution import Solution

solution = Solution()


def test1():
    actual = solution.twoSumII([2, 7, 11, 15], 9)
    expected = [1, 2]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


def test2():
    actual = solution.twoSumII([2, 3, 4], 6)
    expected = [1, 3]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]


def test3():
    actual = solution.twoSumII([-1, 0], -1)
    expected = [1, 2]
    for i in range(0, len(expected)):
        assert actual[i] == expected[i]
