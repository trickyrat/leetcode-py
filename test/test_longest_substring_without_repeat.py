from solution import Solution

solution = Solution()


def test_input_normal_string_should_ok():
    actual = solution.longestSubstringWithoutRepeat("abcabcbb")
    expect = 3
    assert actual == expect


def test_input_same_chars_should_ok():
    actual = solution.longestSubstringWithoutRepeat("bbbbb")
    expect = 1
    assert actual == expect


def test_input_normal_string2_should_ok():
    actual = solution.longestSubstringWithoutRepeat("pwwkew")
    expect = 3
    assert actual == expect


def test_input_empty_string_should_ok():
    actual = solution.longestSubstringWithoutRepeat("")
    expect = 0
    assert actual == expect
