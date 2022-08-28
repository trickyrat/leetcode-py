from itertools import pairwise
from bisect import bisect_right, bisect_left
from math import inf
from typing import List, Optional
import re

from collections import Counter, deque, defaultdict

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

    def add_two_numbers(self, l1: ListNode, l2: ListNode) -> ListNode:
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
                           get_kth_element(total_length // 2) + get_kth_element(total_length // 2 + 1)
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

    def min_depth(self, root: TreeNode) -> int:
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

    def path_sum(self, root: TreeNode, target_num: int) -> List[List[int]]:
        """113. Path Sum II"""
        ret = list()
        path = list()

        def dfs(node: TreeNode, target: int):
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
        ipv4_chunk = r"([0-9][1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
        ipv6_chunk = r"([0-9a-fA-F]{1,4})"
        ipv4_pattern = re.compile(r"^(" + ipv4_chunk + r"\.){3}" + ipv4_chunk + r"$")
        ipv6_pattern = re.compile(r"^(" + ipv6_chunk + r":){7}" + ipv6_chunk + r"$")
        if "." in ip:
            return "IPv4" if ipv4_pattern.match(ip) else "Neither"
        if ":" in ip:
            return "IPv6" if ipv6_pattern.match(ip) else "Neither"
        return "Neither"

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
                    row += (1 if col == m - 1 else 0)
                    col += (1 if col < m - 1 else 0)
                else:
                    col += (1 if row == n - 1 else 0)
                    row += (1 if row < n - 1 else 0)
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

    def preorder(self, root: Node) -> List[int]:
        """589. N-ary Tree Preorder Traversal"""
        ans = []

        def dfs(node: Node) -> None:
            if node is None:
                return None
            ans.append(node.val)
            for ch in node.children:
                dfs(ch)

        dfs(root)
        return ans

    def postorder(self, root: Node) -> List[int]:
        """590. N-ary Tree Postorder Traversal"""
        ans = []

        def dfs(node: Node):
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

    def add_one_row(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
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
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
            return height

        height = calculate_depth(root)
        m = height + 1
        n = 2 ** m - 1
        res = [[''] * n for _ in range(m)]
        queue = deque([(root, 0, (n - 1) // 2)])
        while queue:
            node, row, column = queue.popleft()
            res[row][column] = str(node.val)
            if node.left:
                queue.append((node.left, row + 1, column - 2 ** (height - row - 1)))
            if node.right:
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
        return arr[left + 1:right]

    def width_of_binary_tree(self, root: Optional[TreeNode]) -> int:
        """662. Maximum Width of Binary Tree"""
        level_min = {}

        def dfs(node: Optional[TreeNode], depth: int, index: int) -> int:
            if node is None:
                return 0
            if depth not in level_min:
                level_min[depth] = index
            return max(index - level_min[depth] + 1,
                       dfs(node.left, depth + 1, index * 2),
                       dfs(node.right, depth + 1, index * 2 + 1))

        return dfs(root, 1, 1)

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
        return letters[bisect_right(letters, target)] if target < letters[-1] else letters[0]

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

    def unique_morse_representations(self, words: List[str]) -> int:
        """804. Unique Morse Code Words"""
        morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.",
                 "....", "..", ".---", "-.-", ".-..", "--", "-.",
                 "---", ".--.", "--.-", ".-.", "...", "-", "..-",
                 "...-", ".--", "-..-", "-.--", "--.."]
        return len(set("".join(morse[ord(ch) - ord('a')] for ch in word) for word in words))

    def number_of_lines(self, widths: List[int], s: str) -> List[int]:
        """806. Number of Lines To Write String"""
        max_width = 100
        lines, width = 1, 0
        for c in s:
            need = widths[ord(c) - ord('a')]
            width += need
            if width > max_width:
                width = need
                lines += 1
        return [lines, width]

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

    def di_string_match(self, s: str) -> List[int]:
        """942. DI String Match"""
        n = len(s)
        lo, hi = 0, n
        perm = [0] * (n + 1)
        for i, ch in enumerate(s):
            if ch == 'I':
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

    def min_remove_to_make_valid(self, s: str) -> str:
        """1249 Minimum Remove to Make Valid Parentheses"""
        first_parse_chars = []
        balance = 0
        open_seen = 0
        for c in s:
            if c == '(':
                balance += 1
                open_seen += 1
            if c == ')':
                if balance == 0:
                    continue
                balance -= 1
            first_parse_chars.append(c)

        res = []
        open_to_keep = open_seen - balance
        for c in first_parse_chars:
            if c == '(':
                open_to_keep -= 1
                if open_to_keep < 0:
                    continue
            res.append(c)

        return "".join(res)

    def min_subsequence(self, nums: List[int]) -> List[int]:
        """1403. Minimum Subsequence in Non-Increasing Order"""
        nums.sort(reverse=True)
        total, curr = sum(nums), 0
        for i, num in enumerate(nums):
            curr += num
            if total - curr < curr:
                return nums[:i + 1]

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

    def busy_student(self, start_time: List[int], end_time: List[int], query_time: int) -> int:
        """1450. Number of Students Doing Homework at a Given Time"""
        return sum(s <= query_time <= e for s, e in zip(start_time, end_time))

    def is_prefix_of_word(self, sentence: str, search_word: str) -> int:
        """1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence"""
        i, index, n, = 0, 1, len(sentence)
        while i < n:
            start = i
            while i < n and sentence[i] != ' ':
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

    def rotate_matrix(self, matrix: List[List[int]]):
        """Interview 01.07 Rotate Matrix LCCI"""

        n = len(matrix)
        for i in range(n):
            for j in range(0, i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            matrix[i][:] = matrix[i][::-1]
