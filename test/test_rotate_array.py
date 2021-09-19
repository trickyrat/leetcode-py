from solution import Solution

solution = Solution()


def test_rotate1():
    actual = [1, 2, 3, 4, 5, 6, 7]
    solution.rotate(actual, 3)
    expected = [5, 6, 7, 1, 2, 3, 4]
    assert actual == expected


def test_rotate2():
    actual = [-1, -100, 3, 99]
    solution.rotate(actual, 2)
    expected = [3, 99, -1, -100]
    assert actual == expected
