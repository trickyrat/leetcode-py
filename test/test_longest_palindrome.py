from solution import Solution


solution = Solution()


def test_valid1_should_ok():
    actual = solution.longestPalindrome("babad")
    expected = "bab"
    assert actual == expected


def test_valid2_should_ok():
    actual = solution.longestPalindrome("cbbd")
    expected = "bb"
    assert actual == expected


def test_input_single_char_should_ok():
    actual = solution.longestPalindrome("a")
    expected = "a"
    assert actual == expected


def test_valid4_should_ok():
    actual = solution.longestPalindrome("ac")
    expected = "a"
    assert actual == expected
