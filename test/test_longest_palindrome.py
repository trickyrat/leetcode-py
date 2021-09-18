from solution import Solution


solution = Solution()


def test_valid1():
    actual = solution.longestPalindrome("babad")
    expected = "bab"
    assert actual == expected


def test_valid2():
    actual = solution.longestPalindrome("cbbd")
    expected = "bb"
    assert actual == expected


def test_input_single_char():
    actual = solution.longestPalindrome("a")
    expected = "a"
    assert actual == expected


def test_valid4():
    actual = solution.longestPalindrome("ac")
    expected = "a"
    assert actual == expected
