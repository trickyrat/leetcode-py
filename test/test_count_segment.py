from solution import Solution

solution = Solution()


def test_input_normal_data():
    actual = solution.countSegment("Hello, my name is John")
    expected = 5
    assert expected == actual
