from typing import List
import re


class ListNode:
    """
    Definition fro singly-linked list.
    """

    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next


class Solution:
    def __init__(self) -> None:
        super().__init__()

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1.Two sum
        """
        size = len(nums)
        for i in range(size):
            for j in range(i + 1, size):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        2.Add two Numbers
        """
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

    def longestSubstringWithoutRepeat(self, s: str) -> int:
        """
        3.Longest substring without repeat
        """
        size = len(s)
        i = ans = 0
        index = [0] * 128
        for j in range(0, size):
            i = max(index[ord(s[j])], i)
            ans = max(ans, j - i + 1)
            index[ord(s[j])] = j + 1
        return ans

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        4. Median of Two Sorted Arrays
        """

        def getKthElement(k):
            index1, index2 = 0, 0
            while True:
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])
                newIndex1 = min(index1 + k // 2 - 1, m - 1)
                newIndex2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
                if pivot1 <= pivot2:
                    k -= newIndex1 - index1 + 1
                    index1 = newIndex1 + 1
                else:
                    k -= newIndex2 - index2 + 1
                    index2 = newIndex2 + 1

        m, n = len(nums1), len(nums2)
        totalLength = m + n
        if totalLength % 2 == 1:
            return getKthElement((totalLength + 1) // 2)
        else:
            return (
                getKthElement(totalLength // 2) + getKthElement(totalLength // 2 + 1)
            ) / 2

    def longestPalindrome(self, s: str) -> str:
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
        return s[begin : begin + max_len]

    def validIpAddress(self, IP: str) -> str:
        """
        Valid IP address
        """
        ipv4_chunk = r"([0-9][1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])"
        ipv6_chunk = r"([0-9a-fA-F]{1,4})"
        ipv4_pattern = re.compile(r"^(" + ipv4_chunk + r"\.){3}" + ipv4_chunk + r"$")
        ipv6_pattern = re.compile(r"^(" + ipv6_chunk + r"\:){7}" + ipv6_chunk + r"$")
        if "." in IP:
            return "IPv4" if ipv4_pattern.match(IP) else "Neither"
        if ":" in IP:
            return "IPv6" if ipv6_pattern.match(IP) else "Neither"
        return "Neither"

    def calculate(self, s: str) -> int:
        """
        simple calculator
        """
        stack = []
        operand = 0
        res = 0
        sign = 1
        for chr in s:
            if chr.isdigit():
                operand = (operand * 10) + int(chr)
            elif chr == "+":
                res += sign * operand
                sign = 1
                operand = 0
            elif chr == "-":
                res += sign * operand
                sign = -1
                operand = 0
            elif chr == "(":
                stack.append(res)
                stack.append(sign)
                sign = 1
                res = 0
            elif chr == ")":
                res += sign * operand
                res *= stack.pop()  # sign * res
                res += stack.pop()  # res + res
                operand = 0
        return res + sign * operand

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """
        Number of Provinces
        """

        def dfs(i: int):
            for j in range(provinces):
                if isConnected[i][j] == 1 and j not in visited:
                    visited.add(j)
                    dfs(j)

        provinces = len(isConnected)
        visited: set = set()
        circles = 0

        for i in range(provinces):
            if i not in visited:
                dfs(i)
                circles += 1
        return circles

    def rotate(self, nums: List[int], k: int) -> None:
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
        count: int = gcd(k, n)
        for start in range(0, count):
            current: int = start
            prev: int = nums[start]
            while True:
                next: int = (current + k) % n
                nums[next], prev = prev, nums[next]
                current = next
                if start == current:
                    break

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

    def search_insert(self, nums: List[int], target: int):
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left + 1 if nums[left] < target else left

    def removeElement(self, nums: List[int], val: int) -> int:
        slow = 0
        for item in nums:
            if item != val:
                nums[slow] = item
                slow += 1
        return slow

    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        977. Squares of a Sorted Array
        """
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
