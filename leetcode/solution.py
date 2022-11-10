import collections
from _heapq import heappop, heappush
from itertools import pairwise, zip_longest, product
from bisect import bisect_right, bisect_left
from math import inf
from typing import List, Optional
import re

from collections import Counter, deque, defaultdict

from leetcode.automation import Automation
from leetcode.data_structures.list_node import ListNode
from leetcode.data_structures.tree_node import TreeNode
from leetcode.data_structures.node import Node


class Solution:
    def __init__(self) -> None:
        super().__init__()

    def two_sum(self, nums: List[int], target: int) -> List[int]:
        """1. Two sum"""
        size = len(nums)
        for i in range(size):
            for j in range(i + 1, size):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []

    def add_two_numbers(
            self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        """2. Add two Numbers"""
        carry = 0
        dummy_head: ListNode = ListNode(0, None)
        curr: ListNode = dummy_head
        while l1 or l2:
            sum = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next
            sum += carry
            curr.next = ListNode(sum % 10, None)
            curr = curr.next
            carry = sum // 10
            if carry != 0:
                curr.next = ListNode(carry, None)
        return dummy_head.next

    def longest_substring_without_repeat(self, s: str) -> int:
        """3. Longest substring without repeat"""
        size = len(s)
        i = ans = 0
        index = [0] * 128
        for j in range(0, size):
            i = max(index[ord(s[j])], i)
            ans = max(ans, j - i + 1)
            index[ord(s[j])] = j + 1
        return ans

    def find_median_sorted_arrays(self, nums1: List[int], nums2: List[int]) -> float:
        """4. Median of Two Sorted Arrays"""

        def get_kth_element(k) -> int:
            index1, index2 = 0, 0
            while True:
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])
                new_index1 = min(index1 + k // 2 - 1, m - 1)
                new_index2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[new_index1], nums2[new_index2]
                if pivot1 <= pivot2:
                    k -= new_index1 - index1 + 1
                    index1 = new_index1 + 1
                else:
                    k -= new_index2 - index2 + 1
                    index2 = new_index2 + 1

        m, n = len(nums1), len(nums2)
        total_length = m + n
        if total_length % 2 == 1:
            return get_kth_element((total_length + 1) // 2)
        else:
            return (
                           get_kth_element(total_length // 2)
                           + get_kth_element(total_length // 2 + 1)
                   ) / 2

    def longest_palindrome(self, s: str) -> str:
        """5. Longest Palindromic Substring"""
        n = len(s)
        if n < 2:
            return s
        max_len = 1
        begin = 0
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True
        for L in range(2, n + 1):
            for i in range(n):
                j = L + i - 1
                if j >= n:
                    break
                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                if dp[i][j] and j - i + 1 > max_len:
                    max_len = j - i + 1
                    begin = i
        return s[begin: begin + max_len]

    def z_convert(self, s: str, num_rows: int) -> str:
        """6. Zigzag Conversion"""
        n, r = len(s), num_rows
        if r == 1 or r >= n:
            return s
        t = r * 2 - 2
        ans = []
        for i in range(r):
            for j in range(0, n - i, t):
                ans.append(s[j + i])
                if 0 < i < r - 1 and j + t - i < n:
                    ans.append(s[j + t - i])
        return "".join(ans)

    def reverse_int(self, x: int) -> int:
        """7. Reverse Integer"""
        res = 0
        int_min, int_max = -(2 ** 31), 2 ** 31 - 1
        while x != 0:
            if res < int_min // 10 + 1 or res > int_max // 10:
                return 0
            digit = x % 10
            if x >= 0 or digit <= 0:
                pass
            else:
                digit -= 10
            x = (x - digit) // 10
            res = res * 10 + digit
        return res

    def my_atoi(self, s: str) -> int:
        """8. String to Integer (atoi)"""
        automation = Automation()
        for c in s:
            automation.get(c)
        return automation.sign * automation.res

    def is_palindrome(self, x: int) -> bool:
        """9. Palindrome Number"""
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        reverted_number = 0
        while x > reverted_number:
            reverted_number = reverted_number * 10 + x % 10
            x = x // 10
        return x == reverted_number or x == reverted_number // 10

    def is_match(self, s: str, p: str) -> bool:
        """10. Regular Expression Matching"""
        m, n = len(s), len(p)

        def matches(row: int, col: int) -> bool:
            if row == 0:
                return False
            if p[col - 1] == ".":
                return True
            return s[row - 1] == p[col - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == "*":
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]

    def max_area(self, height: List[int]) -> int:
        """11. Container With Most Water"""
        left, right = 0, len(height) - 1
        res = 0
        while left < right:
            area = min(height[left], height[right]) * (right - left)
            res = max(res, area)
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return res

    def int_to_roman(self, num: int) -> str:
        """12. Integer to Roman"""
        thousands = ["", "M", "MM", "MMM"]
        hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]

        return (
                thousands[num // 1000]
                + hundreds[num % 1000 // 100]
                + tens[num % 100 // 10]
                + ones[num % 10]
        )

    def roman_to_int(self, s: str) -> int:
        """13. Roman to Integer"""
        symbols = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        res = 0
        n = len(s)
        for i, ch in enumerate(s):
            value = symbols[ch]
            if i < n - 1 and value < symbols[s[i + 1]]:
                res -= value
            else:
                res += value
        return res

    def longest_common_prefix(self, strs: List[str]) -> str:
        """14. Longest Common Prefix"""
        if not strs:
            return ""
        length, count = len(strs[0]), len(strs)
        for i in range(length):
            c = strs[0][i]
            if any(i == len(strs[j]) or strs[j][i] != c for j in range(1, count)):
                return strs[0][:i]
        return strs[0]

    def remove_element(self, nums: List[int], val: int) -> int:
        """
        27 Remove Element
        """
        slow = 0
        for item in nums:
            if item != val:
                nums[slow] = item
                slow += 1
        return slow

    def search(self, nums: List[int], target: int) -> int:
        """33. Search in Rotated Sorted Array"""
        n = len(nums)
        if n == 0:
            return -1
        if n == 1:
            return 0 if nums[0] == target else -1
        left = 0
        right = n - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[n - 1]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    def search_insert(self, nums: List[int], target: int):
        """35. Search Insert Position"""
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left + 1 if nums[left] < target else left

    def group_anagrams(self, strs: List[str]) -> List[List[str]]:
        """49. Group Anagrams"""
        dic = collections.defaultdict(list)
        for item in strs:
            count = [0] * 26
            for ch in item:
                count[ord(ch) - ord("a")] += 1
            dic[tuple(count)].append(item)
        return list(dic.values())

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """56. Merge Intervals"""
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged

    def set_zeroes(self, matrix: List[List[int]]) -> None:
        """73. Set Matrix Zeroes"""
        rows, cols = len(matrix), len(matrix[0])
        col0 = 1
        for i in range(0, rows):
            if matrix[i][0] == 0:
                col0 = 0
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(rows - 1, -1, -1):
            for j in range(cols - 1, 0, -1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
            if col0 == 0:
                matrix[i][0] = 0

    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        """74. Search in 2D Matrix"""
        m = len(matrix)
        n = len(matrix[0])
        left = 0
        right = m * n - 1
        while left <= right:
            mid = left + (right - left) // 2
            x = matrix[mid // n][mid % n]
            if x > target:
                right = mid - 1
            elif x < target:
                left = mid + 1
            else:
                return True
        return False

    def min_depth(self, root: Optional[TreeNode]) -> int:
        """111. Minimum Depth of Binary Tree"""
        if not root:
            return 0
        q = deque([(root, 1)])
        while q:
            curr, depth = q.popleft()
            if not curr.left and not curr.right:
                return depth
            if curr.left:
                q.append((curr.left, depth + 1))
            if curr.right:
                q.append((curr.right, depth + 1))
        return 0

    def path_sum(self, root: Optional[TreeNode], target_num: int) -> List[List[int]]:
        """113. Path Sum II"""
        ret = list()
        path = list()

        def dfs(node: Optional[TreeNode], target: int):
            if not node:
                return
            path.append(node.val)
            target -= node.val
            if not node.left and not node.right and target == 0:
                ret.append(path[:])
            dfs(node.left, target)
            dfs(node.right, target)
            path.pop()

        dfs(root, target_num)
        return ret

    def two_sum_ii(self, numbers: List[int], target: int) -> List[int]:
        """167. Two Sum II - Input array is sorted"""
        left = 0
        right = len(numbers) - 1
        while left < right:
            total = numbers[left] + numbers[right]
            if total == target:
                return [left + 1, right + 1]
            elif total > target:
                right -= 1
            else:
                left += 1
        return [-1, -1]

    def rotate(self, nums: List[int], k: int) -> None:
        """189. Rotate Array"""

        # def reverse(nums: List[int], start: int, end: int):
        #     while start < end:
        #         nums[start], nums[end] = nums[end], nums[start]
        #         start += 1
        #         end -= 1

        # size = len(nums)
        # k %= size
        # reverse(nums, 0, size - 1)
        # reverse(nums, 0, k - 1)
        # reverse(nums, k, size - 1)
        def gcd(x: int, y: int) -> int:
            return gcd(y, x % y) if y else x

        n = len(nums)
        k = k % n
        count = gcd(k, n)
        for start in range(0, count):
            current = start
            prev = nums[start]
            while True:
                next = (current + k) % n
                nums[next], prev = prev, nums[next]
                current = next
                if start == current:
                    break

    def calculate(self, s: str) -> int:
        """224.Basic Calculator"""
        ops = [1]
        i = 0
        n = len(s)
        ret = 0
        sign = 1
        while i < n:
            if s[i] == " ":
                i += 1
            elif s[i] == "+":
                sign = ops[-1]
                i += 1
            elif s[i] == "-":
                sign = -ops[-1]
                i += 1
            elif s[i] == "(":
                ops.append(sign)
                i += 1
            elif s[i] == ")":
                ops.pop()
                i += 1
            else:
                num = 0
                while i < n and s[i].isdigit():
                    num = num * 10 + ord(s[i]) - ord("0")
                    i += 1
                ret += num * sign
        return ret

    def move_zeroes(self, nums: List[int]) -> None:
        """283 Move Zeroes"""
        n = len(nums)
        left = right = 0
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

    def find_duplicate(self, nums: List[int]) -> int:
        """287. Find the Duplicate Number"""
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        slow = 0
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow

    def search_v1(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        return -1

    def search_v2(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid
            elif nums[mid] > target:
                right = mid
            else:
                return mid
        return -1

    def count_numbers_with_unique_digits(self, n: int) -> int:
        """357. Count Numbers with Unique Digits"""
        if n == 0:
            return 1
        if n == 1:
            return 10
        res, cur = 10, 9
        for i in range(n - 1):
            cur *= 9 - i
            res += cur
        return res

    def get_sum(self, a: int, b: int) -> int:
        """371. Sum of Two Integers"""
        mask1 = 4294967296  # 2^32
        mask2 = 2147483648  # 2^31
        mask3 = 2147483647  # 2^31-1
        while b != 0:
            carry = ((a & b) << 1) % mask1
            a = (a ^ b) % mask1
            b = carry
        if a & mask2:  # negative
            return ~((a ^ mask2) ^ mask3)
        else:
            return a

    def lexical_order(self, n: int) -> List[int]:
        """386. Lexicographical Numbers"""
        ret = [0] * n
        num = 1
        for i in range(n):
            ret[i] = num
            if num * 10 <= n:
                num *= 10
            else:
                while num % 10 == 9 or num + 1 > n:
                    num //= 10
                num += 1
        return ret

    def valid_utf8(self, data: List[int]) -> bool:
        """393. UTF-8 Validation"""
        n = 0
        for i in range(0, len(data)):
            if n > 0:
                if data[i] >> 6 != 2:
                    return False
                n -= 1
            elif data[i] >> 7 == 0:
                n = 0
            elif data[i] >> 5 == 0b110:
                n = 1
            elif data[i] >> 4 == 0b1110:
                n = 2
            elif data[i] >> 5 == 0b11110:
                n = 3
            else:
                return False
        return n == 0

    def count_segment(self, s: str) -> int:
        """434. Number of Segments in a String"""
        segment_count = 0
        for i in range(0, len(s)):
            if (i == 0 or s[i - 1] == " ") and s[i] != " ":
                segment_count += 1
        return segment_count

    def find_substring_wraparound_string(self, p: str) -> int:
        """467. Unique Substrings in Wraparound String"""
        dp = defaultdict(int)
        k = 0
        for i, ch in enumerate(p):
            if i > 0 and (ord(ch) - ord(p[i - 1])) % 26 == 1:
                k += 1
            else:
                k = 1
            dp[ch] = max(dp[ch], k)
        return sum(dp.values())

    def valid_ip_address(self, ip: str) -> str:
        """468. Valid IP address"""

        def valid_ipv4(IP: str) -> str:
            chunks = IP.split('.')
            for chunk in chunks:
                length = len(chunk)
                if length == 0 or length > 3:
                    return "Neither"
                if chunk[0] == '0' and length != 1:
                    return "Neither"
                if not chunk.isnumeric():
                    return "Neither"
                if int(chunk) > 255:
                    return "Neither"
            return "IPv4"

        def valid_ipv6(IP: str) -> str:
            chunks = IP.split(':')
            hex_digits = "0123456789abcdefABCDEF"
            for chunk in chunks:
                length = len(chunk)
                if length == 0 or length > 4:
                    return "Neither"
                for c in chunk:
                    if hex_digits.find(c) == -1:
                        return "Neither"
            return "IPv6"

        if ip.count(".") == 3:
            return valid_ipv4(ip)
        elif ip.count(":") == 7:
            return valid_ipv6(ip)
        else:
            return "Neither"

    def magical_string(self, n: int) -> int:
        """481. Magical String"""
        if n < 4:
            return 1
        s = [""] * n
        s[:3] = "122"
        res = 1
        i, j = 2, 3
        while j < n:
            size = int(s[i])
            num = 3 - int(s[j - 1])
            while size and j < n:
                s[j] = str(num)
                if num == 1:
                    res += 1
                j += 1
                size -= 1
            i += 1
        return res

    def find_diagonal_order(self, matrix: List[List[int]]) -> List[int]:
        """498. Diagonal Traverse"""
        if matrix is None or len(matrix) == 0:
            return []
        n, m = len(matrix), len(matrix[0])
        row = col = 0
        direction = 1
        res = [0] * n * m
        r = 0
        while row < n and col < m:
            res[r] = matrix[row][col]
            r += 1
            new_row = row + (-1 if direction == 1 else 1)
            new_col = col + (1 if direction == 1 else -1)
            if new_row < 0 or new_row == n or new_col < 0 or new_col == m:
                if direction == 1:
                    row += 1 if col == m - 1 else 0
                    col += 1 if col < m - 1 else 0
                else:
                    col += 1 if row == n - 1 else 0
                    row += 1 if row < n - 1 else 0
                direction = 1 - direction
            else:
                row = new_row
                col = new_col
        return res

    def convert_to_base7(self, num: int) -> str:
        """504. Base 7"""
        if num == 0:
            return "0"
        negative = num < 0
        num = abs(num)
        digits = []
        while num:
            digits.append(str(num % 7))
            num //= 7
        if negative:
            digits.append("-")
        return "".join(reversed(digits))

    def find_lus_length(self, a: str, b: str) -> int:
        """521. Longest Uncommon Subsequence I"""
        return -1 if a == b else max(len(a), len(b))

    def complex_number_multiply(self, num1: str, num2: str) -> str:
        """537. Complex Number Multiply"""
        real1, imag1 = map(int, num1[:-1].split("+"))
        real2, imag2 = map(int, num2[:-1].split("+"))
        return f"{real1 * real2 - imag1 * imag2}+{real1 * imag2 + imag1 * real2}i"

    def find_circle_num(self, is_connected: List[List[int]]) -> int:
        """547. Number of Provinces"""

        def dfs(row: int) -> None:
            for column in range(provinces):
                if is_connected[row][column] == 1 and column not in visited:
                    visited.add(column)
                    dfs(column)

        provinces = len(is_connected)
        visited: set = set()
        circles = 0

        for i in range(provinces):
            if i not in visited:
                dfs(i)
                circles += 1
        return circles

    def optimal_division(self, nums: List[int]) -> str:
        """553. Optimal Division"""
        if len(nums) == 1:
            return str(nums[0])
        if len(nums) == 2:
            return str(nums[0]) + "/" + str(nums[1])
        return str(nums[0]) + "/(" + "/".join(map(str, nums[1:])) + ")"

    def preorder(self, root: Optional[Node]) -> List[int]:
        """589. N-ary Tree Preorder Traversal"""
        ans = []

        def dfs(node: Optional[Node]) -> None:
            if node is None:
                return None
            ans.append(node.val)
            for ch in node.children:
                dfs(ch)

        dfs(root)
        return ans

    def postorder(self, root: Optional[Node]) -> List[int]:
        """590. N-ary Tree Postorder Traversal"""
        ans = []

        def dfs(node: Optional[Node]):
            if node is None:
                return
            for ch in node.children:
                dfs(ch)
            ans.append(node.val)

        dfs(root)
        return ans

    def find_restaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        """599. Minimum Index Sum of Two Lists"""
        index = {s: i for i, s in enumerate(list1)}
        ans = []
        index_sum = inf
        for i, s in enumerate(list2):
            if s in index:
                j = index[s]
                if i + j < index_sum:
                    index_sum = i + j
                    ans = [s]
                elif i + j == index_sum:
                    ans.append(s)
        return ans

    def add_one_row(
            self, root: Optional[TreeNode], val: int, depth: int
    ) -> Optional[TreeNode]:
        """623. Add One Row to Tree"""
        if root is None:
            return
        if depth == 1:
            return TreeNode(val, root, None)
        if depth == 2:
            root.left = TreeNode(val, root.left, None)
            root.right = TreeNode(val, None, root.right)
        else:
            root.left = self.add_one_row(root.left, val, depth - 1)
            root.right = self.add_one_row(root.right, val, depth - 1)
        return root

    def find_longest_chain(self, pairs: List[List[int]]) -> int:
        """646. Maximum Length of Pair Chain"""
        curr, res = -inf, 0
        for x, y in sorted(pairs, key=lambda p: p[1]):
            if curr < x:
                curr = y
                res += 1
        return res

    def find_duplicate_subtrees(
            self, root: Optional[TreeNode]
    ) -> List[Optional[TreeNode]]:
        """652. Find Duplicate Subtrees"""

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            triple = (node.val, dfs(node.left), dfs(node.right))
            if triple in seen:
                (tree, index) = seen[triple]
                repeat.add(tree)
                return index
            else:
                nonlocal idx
                idx += 1
                seen[triple] = (node, idx)
                return idx

        seen = dict()
        repeat = set()
        idx = 0
        dfs(root)
        return list(repeat)

    def print_tree(self, root: Optional[TreeNode]) -> List[List[str]]:
        """655. Print Binary Tree"""

        def calculate_depth(root: Optional[TreeNode]) -> int:
            height = -1
            queue = [root]
            while queue:
                height += 1
                temp = queue
                queue = []
                for node in temp:
                    if node and node.left:
                        queue.append(node.left)
                    if node and node.right:
                        queue.append(node.right)
            return height

        height = calculate_depth(root)
        m = height + 1
        n = 2 ** m - 1
        res = [[""] * n for _ in range(m)]
        queue = deque([(root, 0, (n - 1) // 2)])
        while queue:
            node, row, column = queue.popleft()
            res[row][column] = str(node.val)
            if node and node.left:
                queue.append((node.left, row + 1, column - 2 ** (height - row - 1)))
            if node and node.right:
                queue.append((node.right, row + 1, column + 2 ** (height - row - 1)))
        return res

    def find_closest_elements(self, arr: List[int], k: int, x: int) -> List[int]:
        """658. Find K Closest Elements"""
        right = bisect_left(arr, x)
        left = right - 1
        n = len(arr)
        for _ in range(k):
            if left < 0:
                right += 1
            elif right >= n or x - arr[left] <= arr[right] - x:
                left -= 1
            else:
                right += 1
        return arr[left + 1: right]

    def width_of_binary_tree(self, root: Optional[TreeNode]) -> int:
        """662. Maximum Width of Binary Tree"""
        level_min = {}

        def dfs(node: Optional[TreeNode], depth: int, index: int) -> int:
            if node is None:
                return 0
            if depth not in level_min:
                level_min[depth] = index
            return max(
                index - level_min[depth] + 1,
                dfs(node.left, depth + 1, index * 2),
                dfs(node.right, depth + 1, index * 2 + 1),
            )

        return dfs(root, 1, 1)

    def construct_array(self, n: int, k: int) -> List[int]:
        """667. Beautiful Arrangement II"""
        res = list(range(1, n - k))
        i, j = n - k, n
        while i <= j:
            res.append(i)
            if i != j:
                res.append(j)
            i, j = i + 1, j - 1
        return res

    def trim_bst(
            self, root: Optional[TreeNode], low: int, high: int
    ) -> Optional[TreeNode]:
        """669. Trim a Binary Search Tree"""
        while root and (root.val < low or root.val > high):
            root = root.right if root.val < low else root.left
        if root is None:
            return None
        node = root
        while node.left:
            if node.left.val < low:
                node.left = node.left.right
            else:
                node = node.left
        node = root
        while node.right:
            if node.right.val > high:
                node.right = node.right.left
            else:
                node = node.right
        return root

    def maximum_swap(self, num: int) -> int:
        """670. Maximum Swap"""
        s = list(str(num))
        n = len(s)
        max_index = n - 1
        index1 = index2 = -1
        for i in range(n - 1, -1, -1):
            if s[i] > s[max_index]:
                max_index = i
            elif s[i] < s[max_index]:
                index1, index2 = i, max_index
        if index1 < 0:
            return num
        s[index1], s[index2] = s[index2], s[index1]
        return int("".join(s))

    def flip_lights(self, n: int, presses: int) -> int:
        """672. Bulb Switcher II"""
        seen = set()
        for i in range(2 ** 4):
            press_arr = [(i >> j) & 1 for j in range(4)]
            if sum(press_arr) % 2 == presses % 2 and sum(press_arr) <= presses:
                status = press_arr[0] ^ press_arr[1] ^ press_arr[3]
                if n >= 2:
                    status |= (press_arr[0] ^ press_arr[1]) << 1
                if n >= 3:
                    status |= (press_arr[0] ^ press_arr[2]) << 2
                if n >= 4:
                    status |= (press_arr[0] ^ press_arr[1] ^ press_arr[3]) << 3
                seen.add(status)
        return len(seen)

    def longest_univalue_path(self, root: Optional[TreeNode]) -> int:
        """687. Longest Univalue Path"""
        res = 0

        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            left1 = left + 1 if node.left and node.left.val == node.val else 0
            right1 = right + 1 if node.right and node.right.val == node.val else 0
            nonlocal res
            res = max(res, left1 + right1)
            return max(left1, right1)

        dfs(root)
        return res

    def self_dividing_number(self, left: int, right: int) -> List[int]:
        """728. Self Dividing Numbers"""

        def is_self_dividing(num: int) -> bool:
            x = num
            while x:
                x, d = divmod(x, 10)
                if d == 0 or num % d:
                    return False
            return True

        return [i for i in range(left, right + 1) if is_self_dividing(i)]

    def next_greatest_letter(self, letters: List[str], target: str) -> str:
        """744. Find The Smallest Letter Greater Than Target"""
        return (
            letters[bisect_right(letters, target)]
            if target < letters[-1]
            else letters[0]
        )

    def reach_number(self, target: int) -> int:
        """754. Reach a Number"""
        target = abs(target)
        res = 0
        while target > 0:
            res += 1
            target -= res
        return res if target % 2 == 0 else res + 1 + res % 2

    def order_of_largest_plus_sign(self, n: int, mines: List[List[int]]) -> int:
        """764. Largest Plus Sign"""
        dp = [[n] * n for _ in range(n)]
        banned = set(map(tuple, mines))

        for i in range(n):
            count = 0
            for j in range(n):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)

            count = 0
            for j in range(n - 1, -1, -1):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)

        for j in range(n):
            count = 0
            for i in range(n):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)

            count = 0
            for i in range(n - 1, -1, -1):
                count = 0 if (i, j) in banned else count + 1
                dp[i][j] = min(dp[i][j], count)

        return max(map(max, dp))

    def max_chunks_to_sorted(self, arr: List[int]) -> int:
        """769. Max Chunks To Make Sorted"""
        res = maximum = 0
        for i, x in enumerate(arr):
            maximum = max(maximum, x)
            res += maximum == i
        return res

    def kth_grammar(self, n: int, k: int) -> int:
        """779. K-th Symbol in Grammar"""
        k -= 1
        res = 0
        while k:
            k &= k - 1
            res ^= 1
        return res

    def letter_case_permutation(self, s: str) -> List[str]:
        """784. Letter Case Permutation"""
        res = []
        m = sum(c.isalpha() for c in s)
        for mask in range(1 << m):
            t, k = [], 0
            for c in s:
                if c.isalpha():
                    t.append(c.upper() if mask >> k & 1 else c.lower())
                    k += 1
                else:
                    t.append(c)
            res.append("".join(t))
        return res

    def preimage_size_fzf(self, k: int) -> int:
        """793. Preimage Size of Factorial Zeroes Function"""

        def zeta(n: int) -> int:
            res = 0
            while n:
                n //= 5
                res += n
            return res

        def nx(n: int):
            return bisect_left(range(5 * n), n, key=zeta)

        return nx(k + 1) - nx(k)

    def min_swap(self, nums1: List[int], nums2: List[int]) -> int:
        """801. Minimum Swaps To Make Sequences Increasing"""
        n = len(nums1)
        a, b = 0, 1
        for i in range(1, n):
            at, bt = a, b
            a = b = n
            if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
                a = min(a, at)
                b = min(b, bt + 1)
            if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
                a = min(a, bt)
                b = min(b, at + 1)
        return min(a, b)

    def unique_morse_representations(self, words: List[str]) -> int:
        """804. Unique Morse Code Words"""
        morse = [
            ".-",
            "-...",
            "-.-.",
            "-..",
            ".",
            "..-.",
            "--.",
            "....",
            "..",
            ".---",
            "-.-",
            ".-..",
            "--",
            "-.",
            "---",
            ".--.",
            "--.-",
            ".-.",
            "...",
            "-",
            "..-",
            "...-",
            ".--",
            "-..-",
            "-.--",
            "--..",
        ]
        return len(
            set("".join(morse[ord(ch) - ord("a")] for ch in word) for word in words)
        )

    def number_of_lines(self, widths: List[int], s: str) -> List[int]:
        """806. Number of Lines To Write String"""
        max_width = 100
        lines, width = 1, 0
        for c in s:
            need = widths[ord(c) - ord("a")]
            width += need
            if width > max_width:
                width = need
                lines += 1
        return [lines, width]

    def subdomain_visits(self, cpdomains: List[str]) -> List[str]:
        """811. Subdomain Visit Count"""
        count = Counter()
        for domain in cpdomains:
            c, s = domain.split()
            c = int(c)
            count[s] += c
            while "." in s:
                s = s[s.index(".") + 1:]
                count[s] += c
        return [f"{c} {s}" for s, c in count.items()]

    def ambiguous_coordinates(self, s: str) -> List[str]:
        """816. Ambiguous Coordinates"""

        def get_pos(s: str) -> List[str]:
            pos = []
            if s[0] != '0' or s == '0':
                pos.append(s)
            for p in range(1, len(s)):
                if p != 1 and s[0] == '0' or s[-1] == '0':
                    continue
                pos.append(s[:p] + '.' + s[p:])
            return pos

        n = len(s) - 2
        res = []
        s = s[1:len(s) - 1]
        for l in range(1, n):
            lt = get_pos(s[:l])
            if len(lt) == 0:
                continue
            rt = get_pos(s[l:])
            if len(rt) == 0:
                continue
            for i, j in product(lt, rt):
                res.append('(' + i + ', ' + j + ')')
        return res

    def num_components(self, head: Optional[ListNode], nums: List[int]) -> int:
        """817. Linked List Components"""
        nums_set = set(nums)
        in_set = False
        res = 0
        while head:
            if head.val not in nums_set:
                in_set = False
            elif not in_set:
                in_set = True
                res += 1
            head = head.next
        return res

    def unique_letter_string(self, s: str) -> int:
        """828. Count Unique Characters of All Substrings of a Given String"""
        index = collections.defaultdict(list)
        for i, c in enumerate(s):
            index[c].append(i)

        res = 0
        for arr in index.values():
            arr = [-1] + arr + [len(s)]
            for i in range(1, len(arr) - 1):
                res += (arr[i] - arr[i - 1]) * (arr[i + 1] - arr[i])
        return res

    def backspace_compare(self, s: str, t: str) -> bool:
        """844. Backspace String Compare"""
        i = len(s) - 1
        j = len(t) - 1
        skip_s = skip_t = 0
        while i >= 0 or j >= 0:
            while i >= 0:
                if s[i] == "#":
                    skip_s += 1
                    i -= 1
                elif skip_s > 0:
                    skip_s -= 1
                    i -= 1
                else:
                    break
            while j >= 0:
                if t[j] == "#":
                    skip_t += 1
                    j -= 1
                elif skip_t > 0:
                    skip_t -= 1
                    j -= 1
                else:
                    break
            if i >= 0 and j >= 0:
                if s[i] != t[j]:
                    return False
            else:
                if i >= 0 or j >= 0:
                    return False
            i -= 1
            j -= 1
        return True

    def score_of_parentheses(self, s: str) -> int:
        """856. Score of Parentheses"""
        res = bal = 0
        for i, c in enumerate(s):
            bal += 1 if c == "(" else -1
            if c == ")" and s[i - 1] == "(":
                res += 1 << bal
        return res

    def min_cost_to_hire_worker(
            self, quality: List[int], wage: List[int], k: int
    ) -> float:
        """857. Minimum Cost to Hire K Workers"""
        pairs = sorted(zip(quality, wage), key=lambda p: p[1] / p[0])
        res = inf
        total_quality = 0
        hire = []
        for q, w in pairs[: k - 1]:
            total_quality += q
            heappush(hire, -q)
        for q, w in pairs[k - 1:]:
            total_quality += q
            heappush(hire, -q)
            res = min(res, w / q * total_quality)
            total_quality += heappop(hire)
        return res

    def shortest_subarray(self, nums: List[int], k: int) -> int:
        """862. Shortest Subarray with Sum at Least K"""
        pre_sum_array = [0]
        res = len(nums) + 1
        for num in nums:
            pre_sum_array.append(pre_sum_array[-1] + num)

        q = deque()
        for i, curr_sum in enumerate(pre_sum_array):
            while q and curr_sum - pre_sum_array[q[0]] >= k:
                res = min(res, i - q.popleft())
            while q and pre_sum_array[q[-1]] >= curr_sum:
                q.pop()
            q.append(i)
        return res if res < len(nums) + 1 else -1

    def advantage_count(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """870. Advantage Shuffle"""
        n = len(nums1)
        index1, index2 = list(range(n)), list(range(n))
        index1.sort(key=lambda x: nums1[x])
        index2.sort(key=lambda x: nums2[x])
        res = [0] * n
        left, right = 0, n - 1
        for i in range(n):
            if nums1[index1[i]] > nums2[index2[left]]:
                res[index2[left]] = nums1[index1[i]]
                left += 1
            else:
                res[index2[right]] = nums1[index1[i]]
                right -= 1
        return res

    def middle_node(self, head: ListNode) -> ListNode:
        """876. Middle of the Linked List"""
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def projection_area(self, grid: List[List[int]]) -> int:
        """883. Projection Area of 3D Shapes"""
        xy_area = sum(v > 0 for row in grid for v in row)
        yz_area = sum(map(max, zip(*grid)))
        zx_area = sum(map(max, grid))
        return xy_area + yz_area + zx_area

    def possible_bipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        """886. Possible Bipartition"""
        group = [[] for _ in range(n)]
        for x, y in dislikes:
            group[x - 1].append(y - 1)
            group[y - 1].append(x - 1)
        color = [0] * n

        def dfs(x: int, c: int) -> bool:
            color[x] = c
            return all(color[y] != c and (color[y] or dfs(y, -c)) for y in group[x])

        return all(c or dfs(i, 1) for i, c in enumerate(color))

    def total_fruit(self, fruits: List[int]) -> int:
        """904. Fruit Into Baskets"""
        counter = Counter()
        left = res = 0
        for right, x in enumerate(fruits):
            counter[x] += 1
            while len(counter) > 2:
                counter[fruits[left]] -= 1
                if counter[fruits[left]] == 0:
                    counter.pop(fruits[left])
                left += 1
            res = max(res, right - left + 1)
        return res

    def sort_array_by_parity(self, nums: List[int]) -> List[int]:
        """905. Sort Array By Parity"""
        left, right = 0, len(nums) - 1
        while left < right:
            while left < right and nums[left] % 2 == 0:
                left += 1
            while left < right and nums[right] % 2 == 1:
                right -= 1
            if left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        return nums

    def sum_subarray_mins(self, arr: List[int]) -> int:
        """907. Sum of Subarray Minimums"""
        MOD = 10 ** 9 + 7
        n = len(arr)
        mono_stack = []
        dp = [0] * n
        res = 0
        for i, x in enumerate(arr):
            while mono_stack and arr[mono_stack[-1]] > x:
                mono_stack.pop()
            k = i - mono_stack[-1] if mono_stack else i + 1
            dp[i] = k * x + (dp[i - k] if mono_stack else 0)
            res = (res + dp[i]) % MOD
            mono_stack.append(i)
        return res

    def partition_disjoint(self, nums: List[int]) -> int:
        """915. Partition Array into Disjoint Intervals"""
        n = len(nums)
        curr_max = left_max = nums[0]
        index = 0
        for i in range(1, n - 1):
            curr_max = max(curr_max, nums[i])
            if nums[i] < left_max:
                left_max, index = curr_max, i
        return index + 1

    def min_add_to_make_valid(self, s: str) -> int:
        """921. Minimum Add to Make Parentheses Valid"""
        res = count = 0
        for c in s:
            if c == "(":
                count += 1
            elif count > 0:
                count -= 1
            else:
                res += 1
        return res + count

    def three_equal_parts(self, arr: List[int]) -> List[int]:
        """927. Three Equal Parts"""
        total = sum(arr)
        if total % 3:
            return [-1, -1]
        if total == 0:
            return [0, 2]

        partial = total // 3
        first = second = third = curr = 0
        for i, x in enumerate(arr):
            if x:
                if curr == 0:
                    first = i
                elif curr == partial:
                    second = i
                elif curr == 2 * partial:
                    third = i
                curr += 1
        n = len(arr)
        length = n - third
        if first + length <= second and second + length <= third:
            i = 0
            while third + i < n:
                if (
                        arr[first + i] != arr[second + i]
                        or arr[first + i] != arr[third + i]
                ):
                    return [-1, -1]
                i += 1
            return [first + length - 1, second + length]
        return [-1, -1]

    def shortest_bridge(self, grid: List[List[int]]) -> int:
        """934. Shortest Bridge"""
        n = len(grid)
        for i, row in enumerate(grid):
            for j, col in enumerate(row):
                if col != 1:
                    continue
                q = []

                def dfs(x: int, y: int) -> None:
                    grid[x][y] = -1
                    q.append((x, y))
                    for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                            dfs(nx, ny)

                dfs(i, j)
                step = 0
                while True:
                    temp = q
                    q = []
                    for x, y in temp:
                        for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                            if 0 <= nx < n and 0 <= ny < n:
                                if grid[nx][ny] == 1:
                                    return step
                                if grid[nx][ny] == 0:
                                    grid[nx][ny] = -1
                                    q.append((nx, ny))
                    step += 1
        return 0

    def distinct_subseq_ii(self, s: str) -> int:
        """940. Distinct Subsequences II"""
        mod = 10 ** 9 + 7
        alphas = [0] * 26
        res = 0
        for i, ch in enumerate(s):
            index = ord(s[i]) - ord("a")
            alphas[index], res = (res + 1) % mod, (res * 2 + 1 - alphas[index]) % mod
        return res

    def di_string_match(self, s: str) -> List[int]:
        """942. DI String Match"""
        n = len(s)
        lo, hi = 0, n
        perm = [0] * (n + 1)
        for i, ch in enumerate(s):
            if ch == "I":
                perm[i] = lo
                lo += 1
            else:
                perm[i] = hi
                hi -= 1
        perm[n] = lo
        return perm

    def min_deletion_size(self, strs: List[str]) -> int:
        """944. Delete Columns to Make Sorted"""
        return sum(any(x > y for x, y in pairwise(col)) for col in zip(*strs))

    def validate_stack_sequences(self, pushed: List[int], popped: List[int]) -> bool:
        """946. Validate Stack Sequences"""
        stack, j = [], 0
        for x in pushed:
            stack.append(x)
            while stack and stack[-1] == popped[j]:
                stack.pop()
                j += 1
        return len(stack) == 0

    def repeated_n_times(self, nums: List[int]) -> int:
        """961. N-Repeated Element in Size 2N Array"""
        found = set()
        for num in nums:
            if num in found:
                return num
            found.add(num)
        return -1

    def sorted_squares(self, nums: List[int]) -> List[int]:
        """977. Squares of a Sorted Array"""
        n = len(nums)
        ans = [0] * n
        i, j, pos = 0, n - 1, n - 1
        while i <= j:
            if nums[i] * nums[i] > nums[j] * nums[j]:
                ans[pos] = nums[i] * nums[i]
                i += 1
            else:
                ans[pos] = nums[j] * nums[j]
                j -= 1
            pos -= 1
        return ans

    def insert_into_max_tree(
            self, root: Optional[TreeNode], val: int
    ) -> Optional[TreeNode]:
        """998. Maximum Binary Tree II"""
        parent, curr = None, root
        while curr:
            if val > curr.val:
                if not parent:
                    return TreeNode(val, root, None)
                parent.right = TreeNode(val, curr, None)
                return root
            else:
                parent = curr
                curr = curr.right
        parent.right = TreeNode(val)
        return root

    def parse_bool_expr(self, expression: str) -> bool:
        """1106. Parsing A Boolean Expression"""
        stk = []
        for c in expression:
            if c == ',':
                continue
            if c != ")":
                stk.append(c)
                continue
            t = f = 0
            while stk[-1] != '(':
                if stk.pop() == 't':
                    t += 1
                else:
                    f += 1
            stk.pop()
            op = stk.pop()
            if op == '!':
                stk.append('t' if f == 1 else 'f')
            elif op == '&':
                stk.append('t' if f == 0 else 'f')
            elif op == '|':
                stk.append('t' if t else 'f')
        return stk[-1] == 't'

    def min_remove_to_make_valid(self, s: str) -> str:
        """1249 Minimum Remove to Make Valid Parentheses"""
        first_parse_chars = []
        balance = 0
        open_seen = 0
        for c in s:
            if c == "(":
                balance += 1
                open_seen += 1
            if c == ")":
                if balance == 0:
                    continue
                balance -= 1
            first_parse_chars.append(c)

        res = []
        open_to_keep = open_seen - balance
        for c in first_parse_chars:
            if c == "(":
                open_to_keep -= 1
                if open_to_keep < 0:
                    continue
            res.append(c)

        return "".join(res)

    def min_subsequence(self, nums: List[int]) -> List[int]:
        """1403. Minimum Subsequence in Non-Increasing Order"""
        nums.sort(reverse=True)
        total, curr = sum(nums), 0
        res = []
        for num in nums:
            curr += num
            res.append(num)
            if total - curr < curr:
                break
        return res

    def reformat(self, s: str) -> str:
        """1417. Reformat The String"""
        sum_digit = sum(c.isdigit() for c in s)
        sum_alpha = len(s) - sum_digit
        if abs(sum_digit - sum_alpha) > 1:
            return ""
        flag = sum_digit > sum_alpha
        t = list(s)
        j = 1
        for i in range(0, len(t), 2):
            if t[i].isdigit() != flag:
                while t[j].isdigit() != flag:
                    j += 2
                t[i], t[j] = t[j], t[i]
        return "".join(t)

    def build_array(self, target: List[int], n: int) -> List[str]:
        """1441. Build an Array With Stack Operations"""
        res = []
        prev = 0
        for num in target:
            for _ in range(num - prev - 1):
                res.append("Push")
                res.append("Pop")
            res.append("Push")
            prev = num
        return res

    def busy_student(
            self, start_time: List[int], end_time: List[int], query_time: int
    ) -> int:
        """1450. Number of Students Doing Homework at a Given Time"""
        return sum(s <= query_time <= e for s, e in zip(start_time, end_time))

    def is_prefix_of_word(self, sentence: str, search_word: str) -> int:
        """1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence"""
        i, index, n, = (
            0,
            1,
            len(sentence),
        )
        while i < n:
            start = i
            while i < n and sentence[i] != " ":
                i += 1
            end = i
            if sentence[start:end].startswith(search_word):
                return index
            index += 1
            i += 1
        return -1

    def can_be_equal(self, target: List[int], arr: List[int]) -> bool:
        """1460. Make Two Arrays Equal by Reversing Sub-arrays"""
        target.sort()
        arr.sort()
        return target == arr

    def shuffle(self, nums: List[int], n: int) -> List[int]:
        """1470. Shuffle the Array"""
        res = [0] * (2 * n)
        for i in range(n):
            res[2 * i] = nums[i]
            res[2 * i + 1] = nums[n + i]
        return res

    def final_prices(self, prices: List[int]) -> List[int]:
        """1475. Final Prices With a Special Discount in a Shop"""
        n = len(prices)
        res = [0] * n
        stack = [0]
        for i in range(n - 1, -1, -1):
            p = prices[i]
            while len(stack) > 1 and stack[-1] > p:
                stack.pop()
            res[i] = p - stack[-1]
            stack.append(p)
        return res

    def can_make_arithmetic_progression(self, arr: List[int]) -> bool:
        """1502. Can Make Arithmetic Progression From Sequence"""
        arr.sort()
        for i in range(1, len(arr) - 1):
            if arr[i] * 2 != arr[i - 1] + arr[i + 1]:
                return False
        return True

    def modify_string(self, s: str) -> str:
        """1576. Replace All ?'s to Avoid Consecutive Repeating Characters"""
        n = len(s)
        arr = list(s)
        for i in range(n):
            if arr[i] == "?":
                for ch in "abc":
                    if not (
                            i > 0 and arr[i - 1] == ch or i < n - 1 and arr[i + 1] == ch
                    ):
                        arr[i] = ch
                        break
        return "".join(arr)

    def num_special(self, mat: List[List[int]]) -> int:
        """1582. Special Positions in a Binary Matrix"""
        for i, row in enumerate(mat):
            count = sum(row) - (i == 0)
            if count:
                for j, x in enumerate(row):
                    if x == 1:
                        mat[0][j] += count
        return sum(x == 1 for x in mat[0])

    def reorder_spaces(self, text: str) -> str:
        """1592. Rearrange Spaces Between Words"""
        words = text.split()
        space = text.count(" ")
        if len(words) == 1:
            return words[0] + " " * space
        per_space, rest_space = divmod(space, len(words) - 1)
        return (" " * per_space).join(words) + " " * rest_space

    def min_operations(self, logs: List[str]) -> int:
        """1598. Crawler Log Folder"""
        depth = 0
        for _, log in enumerate(logs):
            if log == "./":
                continue
            elif log == "../":
                if depth > 0:
                    depth -= 1
            else:
                depth += 1
        return depth

    def special_array(self, nums: List[int]) -> int:
        """1608. Special Array With X Elements Greater Than or Equal X"""
        nums.sort(reverse=True)
        n = len(nums)
        for i in range(1, n + 1):
            if nums[i - 1] >= i and (i == n or nums[i] < i):
                return i
        return -1

    def trim_mean(self, arr: List[int]) -> float:
        """1619. Mean of Array After Removing Some Elements"""
        arr.sort()
        n = len(arr)
        return sum(arr[n // 20: -n // 20]) / (n * 0.9)

    def best_coordinate(self, towers: List[List[int]], radius: int) -> List[int]:
        """1620. Coordinate With Maximum Network Quality"""
        x_max = max(t[0] for t in towers)
        y_max = max(t[1] for t in towers)
        cx = cy = max_quality = 0
        for x in range(x_max + 1):
            for y in range(y_max + 1):
                quality = 0
                for tx, ty, q in towers:
                    d = (x - tx) ** 2 + (y - ty) ** 2
                    if d <= radius ** 2:
                        quality += int(q / (1 + d ** 0.5))
                if quality > max_quality:
                    cx, cy, max_quality = x, y, quality
        return [cx, cy]

    def max_length_between_equal_characters(self, s: str) -> int:
        """1624. Largest Substring Between Two Equal Characters"""
        res = -1
        dic = {}
        for i, ch in enumerate(s):
            if ch not in dic:
                dic[ch] = i
            else:
                res = max(res, i - dic[ch] - 1)
        return res

    def frequency_sort(self, nums: List[int]) -> List[int]:
        """1636. Sort Array by Increasing Frequency"""
        count = Counter(nums)
        nums.sort(key=lambda x: (count[x], -x))
        return nums

    def array_strings_are_equal(self, word1: List[str], word2: List[str]) -> bool:
        """1662. Check If Two String Arrays are Equivalent"""
        p1 = p2 = 0  # index of word
        i = j = 0  # index of character
        while p1 < len(word1) and p2 < len(word2):
            if word1[p1][i] != word2[p2][j]:
                return False
            i += 1
            if i == len(word1[p1]):
                p1 += 1
                i = 0
            j += 1
            if j == len(word2[p2]):
                p2 += 1
                j = 0
        return p1 == len(word1) and p2 == len(word2)

    def max_repeating(self, sequence: str, word: str) -> int:
        """1668. Maximum Repeating Substring"""
        m, n = len(sequence), len(word)
        if m < n:
            return 0
        fail = [-1] * n
        for i in range(1, n):
            j = fail[i - 1]
            while j != -1 and word[j + 1] != word[i]:
                j = fail[j]
            if word[j + 1] == word[i]:
                fail[i] = j + 1

        f = [0] * m
        j = -1
        for i in range(m):
            while j != -1 and word[j + 1] != sequence[i]:
                j = fail[j]
            if word[j + 1] == sequence[i]:
                j += 1
                if j == n - 1:
                    f[i] = (0 if i == n - 1 else f[i - n]) + 1
                    j = fail[j]
        return max(f)

    def interpret(self, command: str) -> str:
        """1678. Goal Parser Interpretation"""
        res = []
        for i, c in enumerate(command):
            if c == 'G':
                res.append(c)
            elif c == '(':
                res.append('o' if command[i + 1] == ')' else "al")
        return "".join(res)

    def count_consistent_strings(self, allowed: str, words: List[str]) -> int:
        """1684. Count the Number of Consistent Strings"""
        mask = 0
        for c in allowed:
            mask |= 1 << (ord(c) - ord('a'))
        res = 0
        for word in words:
            mask1 = 0
            for c in word:
                mask1 |= 1 << (ord(c) - ord('a'))
            if (mask | mask1) == mask:
                res += 1
        return res

    def reformat_number(self, number: str) -> str:
        """1694. Reformat Phone Number"""
        digits = []
        for ch in number:
            if ch.isdigit():
                digits.append(ch)
        n, pt = len(digits), 0
        res = []
        while n > 0:
            if n > 4:
                res.append("".join(digits[pt: pt + 3]))
                pt += 3
                n -= 3
            else:
                if n == 4:
                    res.append("".join(digits[pt: pt + 2]))
                    res.append("".join(digits[pt + 2: pt + 4]))
                else:
                    res.append("".join(digits[pt: pt + n]))
                break
        return "-".join(res)

    def count_students(self, students: List[int], sandwiches: List[int]) -> int:
        """1700. Number of Students Unable to Eat Lunch"""
        square = sum(students)
        circular = len(students) - square
        for sandwich in sandwiches:
            if sandwich == 1 and square:
                square -= 1
            elif sandwich == 0 and circular:
                circular -= 1
            else:
                break
        return square + circular

    def halves_are_alike(self, s: str) -> bool:
        """1704. Determine if String Halves Are Alike"""
        vowels = "aeiouAEIOU"
        a, b = s[:len(s) // 2], s[len(s) // 2:]
        return sum(c in vowels for c in a) == sum(c in vowels for c in b)

    def merge_alternately(self, word1: str, word2: str) -> str:
        """1768. Merge Strings Alternately"""
        res = []
        for i, j in zip_longest(word1, word2):
            if i:
                res.append(i)
            if j:
                res.append(j)
        return "".join(res)

    def count_matches(
            self, items: List[List[str]], rule_key: str, rule_value: str
    ) -> int:
        """1773. Count Items Matching a Rule"""
        key_index = {"type": 0, "color": 1, "name": 2}[rule_key]
        return sum(item[key_index] == rule_value for item in items)

    def check_ones_segment(self, s: str) -> bool:
        """1784. Check if Binary String Has at Most One Segment of Ones"""
        return "01" not in s

    def are_almost_equal(self, s1: str, s2: str) -> bool:
        """1790. Check if One String Swap Can Make Strings Equal"""
        i = j = -1
        for index, (x, y) in enumerate(zip(s1, s2)):
            if x != y:
                if i < 0:
                    i = index
                elif j < 0:
                    j = index
                else:
                    return False
        return i < 0 or j >= 0 and s1[i] == s2[j] and s1[j] == s2[i]

    def max_ascending_sum(self, nums: List[int]) -> int:
        """1800. Maximum Ascending Subarray Sum"""
        res = i = 0
        n = len(nums)
        while i < n:
            curr = nums[i]
            i += 1
            while i < n and nums[i] > nums[i - 1]:
                curr += nums[i]
                i += 1
            res = max(res, curr)
        return res

    def array_sign(self, nums: List[int]) -> int:
        """1822. Sign of the Product of an Array"""
        sign = 1
        for num in nums:
            if num == 0:
                return 0
            if num < 0:
                sign = -sign
        return sign

    def find_the_winner(self, n: int, k: int) -> int:
        """1823. Find the Winner of the Circular Game"""
        winner = 1
        for i in range(2, n + 1):
            winner = (winner + k - 1) % i + 1
        return winner

    def pivot_index(self, nums: List[int]) -> int:
        """1991. Find the Middle Index in Array"""
        total = sum(nums)
        sum_tmp = 0
        for i, num in enumerate(nums):
            if 2 * sum_tmp + num == total:
                return i
            sum_tmp += num
        return -1

    def count_k_difference(self, nums: List[int], k: int) -> int:
        """2006. Count Number of Pairs With Absolute Difference K"""
        res = 0
        cnt = Counter()
        for num in nums:
            res += cnt[num - k] + cnt[num + k]
            cnt[num] += 1
        return res

    def maximum_difference(self, nums: List[int]) -> int:
        """2016. Maximum Difference Between Increasing Elements"""
        n = len(nums)
        ans, pre_min = -1, nums[0]
        for i in range(1, n):
            if nums[i] > pre_min:
                ans = max(ans, nums[i] - pre_min)
            else:
                pre_min = nums[i]
        return ans

    def count_max_or_subsets(self, nums: List[int]) -> int:
        """2044. Count Number of Maximum Bitwise-OR Subsets"""
        max_or, cnt = 0, 0

        def dfs(pos: int, or_val: int) -> None:
            if pos == len(nums):
                nonlocal max_or, cnt
                if or_val > max_or:
                    max_or, cnt = or_val, 1
                elif or_val == max_or:
                    cnt += 1
                return
            dfs(pos + 1, or_val | nums[pos])
            dfs(pos + 1, or_val)

        dfs(0, 0)
        return cnt

    def plates_between_candles(self, s: str, queries: List[List[int]]) -> List[int]:
        """2055. Plates Between Candles"""
        n = len(s)
        pre_sum, total = [0] * n, 0
        left, l = [0] * n, -1
        for i, ch in enumerate(s):
            if ch == "*":
                total += 1
            else:
                l = i
            pre_sum[i] = total
            left[i] = l
        right, r = [0] * n, -1
        for i in range(n - 1, -1, -1):
            if s[i] == "|":
                r = i
            right[i] = r
        ans = [0] * len(queries)
        for i, (x, y) in enumerate(queries):
            x, y = right[x], left[y]
            if 0 <= x < y and y >= 0:
                ans[i] = pre_sum[y] - pre_sum[x]
        return ans
