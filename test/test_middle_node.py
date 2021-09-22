from solution import Solution
from utils import *

solution = Solution()


def test_input_even_data():
    head = initListNode([1, 2, 3, 4, 5, 6])
    actualNode = solution.middleNode(head)
    actual = printListNode(actualNode)
    expectedNode = initListNode([4, 5, 6])
    expected = printListNode((expectedNode))
    assert expected == actual


def test_input_odd_data():
    head = initListNode([1, 2, 3, 4, 5])
    actualNode = solution.middleNode(head)
    actual = printListNode(actualNode)
    expectedNode = initListNode([3, 4, 5])
    expected = printListNode((expectedNode))
    assert expected == actual
