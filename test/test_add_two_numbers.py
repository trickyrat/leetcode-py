from solution import Solution, ListNode

solution = Solution()


def test_input_two_same_length_lists_should_ok():
    l1 = ListNode(2)
    l1.next = ListNode(4)
    l1.next.next = ListNode(3)

    l2 = ListNode(5)
    l2.next = ListNode(6)
    l2.next.next = ListNode(4)

    actual = solution.addTwoNumbers(l1, l2)

    expect = ListNode(7)
    expect.next = ListNode(0)
    expect.next.next = ListNode(8)

    assert actual.val == expect.val
    assert actual.next.val == expect.next.val
    assert actual.next.next.val == expect.next.next.val


def test_input_two_zero_same_length_lists_should_ok():
    l1 = ListNode(0)
    l2 = ListNode(0)
    actual = solution.addTwoNumbers(l1, l2)
    expect = ListNode(0)
    assert actual.val == expect.val


def test_input_two_lists_with_carry_should_ok():
    l1 = ListNode(9)
    l1.next = ListNode(9)
    l1.next.next = ListNode(9)
    l1.next.next.next = ListNode(9)
    l1.next.next.next.next = ListNode(9)
    l1.next.next.next.next.next = ListNode(9)
    l1.next.next.next.next.next.next = ListNode(9)

    l2 = ListNode(9)
    l2.next = ListNode(9)
    l2.next.next = ListNode(9)
    l2.next.next.next = ListNode(9)

    actual = solution.addTwoNumbers(l1, l2)

    expect = ListNode(8)
    expect.next = ListNode(9)
    expect.next.next = ListNode(9)
    expect.next.next.next = ListNode(9)
    expect.next.next.next.next = ListNode(0)
    expect.next.next.next.next.next = ListNode(0)
    expect.next.next.next.next.next.next = ListNode(0)
    expect.next.next.next.next.next.next.next = ListNode(1)

    assert actual.val == expect.val
    assert actual.next.val == expect.next.val
    assert actual.next.next.val == expect.next.next.val
    assert actual.next.next.next.val == expect.next.next.next.val
    assert actual.next.next.next.next.val == expect.next.next.next.next.val
    assert actual.next.next.next.next.next.val == expect.next.next.next.next.next.val
    assert (
        actual.next.next.next.next.next.next.val
        == expect.next.next.next.next.next.next.val
    )
    assert (
        actual.next.next.next.next.next.next.next.val
        == expect.next.next.next.next.next.next.next.val
    )
