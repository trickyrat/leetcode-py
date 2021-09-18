from solution import Solution

solution = Solution()


def test_input_normal_string():
    actual = solution.longestSubstringWithoutRepeat("abcabcbb")
    expect = 3
    assert actual == expect


def test_input_same_chars():
    actual = solution.longestSubstringWithoutRepeat("bbbbb")
    expect = 1
    assert actual == expect


def test_input_normal_string2():
    actual = solution.longestSubstringWithoutRepeat("pwwkew")
    expect = 3
    assert actual == expect


def test_input_empty_string():
    actual = solution.longestSubstringWithoutRepeat("")
    expect = 0
    assert actual == expect
