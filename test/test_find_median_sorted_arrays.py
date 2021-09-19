from solution import Solution

solution = Solution()


def test_input_two_same_length_arrays_should_ok():
    actual = solution.findMedianSortedArrays([1, 2, 3, 4], [5, 6, 7, 8])
    expect = 4.5
    assert actual == expect


def test_input_two_different_length_arrays_should_ok():
    actual = solution.findMedianSortedArrays([1, 3], [2])
    expect = 2.0
    assert actual == expect


def test_input_two_element_zero_arrays_should_ok():
    actual = solution.findMedianSortedArrays([0, 0], [0, 0])
    expect = 0.0
    assert actual == expect


def test_input_left_empty_array_should_ok():
    actual = solution.findMedianSortedArrays([], [1])
    expect = 1.0
    assert actual == expect


def test_input_right_empty_array_should_ok():
    actual = solution.findMedianSortedArrays([2], [])
    expect = 2.0
    assert actual == expect
