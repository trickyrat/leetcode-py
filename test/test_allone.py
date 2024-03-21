from src.leetcode.all_one import AllOne


class TestAllOne:
    def test_all_one(self):
        all_one = AllOne()
        all_one.inc("hello")
        all_one.inc("hello")
        max_key = all_one.get_max_key()
        min_key = all_one.get_min_key()

        assert max_key == "hello"
        assert min_key == "hello"

        all_one.inc("leet")
        max_key = all_one.get_max_key()
        min_key = all_one.get_min_key()

        assert max_key == "hello"
        assert min_key == "leet"

        all_one.inc("leet")
        all_one.dec("hello")
        max_key = all_one.get_max_key()
        min_key = all_one.get_min_key()

        assert max_key == "leet"
        assert min_key == "hello"

