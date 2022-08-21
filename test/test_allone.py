from leetcode.all_one import AllOne


def test_all_one():
    all_one = AllOne()
    all_one.inc("hello")
    all_one.inc("hello")
    actual_max_key1 = all_one.get_max_key()
    actual_min_key1 = all_one.get_min_key()

    assert actual_max_key1 == "hello"
    assert actual_min_key1 == "hello"

    all_one.inc("leet")
    actual_max_key2 = all_one.get_max_key()
    actual_min_key2 = all_one.get_min_key()

    assert actual_max_key2 == "hello"
    assert actual_min_key2 == "leet"
