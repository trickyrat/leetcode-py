from typing import List
import re


class ListNode:
    """
    Definition fro singly-linked list.
    """

    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next


class TreeNode:
    """
    Definition for a binary tree node.
    """

    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


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
        4 Median of Two Sorted Arrays
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
        """
        5.Longest Palindromic Substring
        """
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

    def removeElement(self, nums: List[int], val: int) -> int:
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
        """
        33 Search in Rotated Sorted Array
        """
        n = len(nums)
        if n == 0:
            return -1
        if n == 1:
            return 0 if nums[0] == target else -1
        l = 0
        r = n - 1
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target and target <= nums[n - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1

    def search_insert(self, nums: List[int], target: int):
        """
        35.Search Insert Position
        """
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left + 1 if nums[left] < target else left

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        74 Search in 2D Matrix
        """
        m = len(matrix)
        n = len(matrix[0])
        l = 0
        r = m * n - 1
        while l <= r:
            mid = l + (r - l) // 2
            x = matrix[mid // n][mid % n]
            if x > target:
                r = mid - 1
            elif x < target:
                l = mid + 1
            else:
                return True
        return False

    def twoSumII(self, numbers: List[int], target: int) -> List[int]:
        """
        167 Two Sum II - Input array is sorted
        """
        left = 0
        right = len(numbers) - 1
        while left < right:
            sum = numbers[left] + numbers[right]
            if sum == target:
                return [left + 1, right + 1]
            elif sum > target:
                right -= 1
            else:
                left += 1
        return [-1, -1]

    def rotate(self, nums: List[int], k: int) -> None:
        """
        189 Rotate Array
        """
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
        """
        224.Basic Calculator
        """
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

    def moveZeroes(self, nums: List[int]) -> None:
        """
        283 Move Zeroes
        """
        n = len(nums)
        left = right = 0
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1

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

    def getSum(self, a: int, b: int) -> int:
        """
        371.Sum of Two Integers
        """
        MASK1 = 4294967296  # 2^32
        MASK2 = 2147483648  # 2^31
        MASK3 = 2147483647  # 2^31-1
        while b != 0:
            carry = ((a & b) << 1) % MASK1
            a = (a ^ b) % MASK1
            b = carry
        if a & MASK2:  # 负数
            return ~((a ^ MASK2) ^ MASK3)
        else:
            return a

    def countSegment(self, s: str) -> int:
        """
        434 Number of Segments in a String
        """
        segmentCount = 0
        for i in range(0, len(s)):
            if (i == 0 or s[i - 1] == " ") and s[i] != " ":
                segmentCount += 1
        return segmentCount

    def validIpAddress(self, IP: str) -> str:
        """
        468.Valid IP address
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

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """
        547 Number of Provinces
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

    def backspaceCompare(self, s: str, t: str) -> bool:
        """ "
        844 Backspace String Compare
        """
        i = len(s) - 1
        j = len(t) - 1
        skipS = skipT = 0
        while i >= 0 or j >= 0:
            while i >= 0:
                if s[i] == "#":
                    skipS += 1
                    i -= 1
                elif skipS > 0:
                    skipS -= 1
                    i -= 1
                else:
                    break
            while j >= 0:
                if t[j] == "#":
                    skipT += 1
                    j -= 1
                elif skipT > 0:
                    skipT -= 1
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

    def middleNode(self, head: ListNode) -> ListNode:
        """
        876 Middle of the Linked List
        """
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        977 Squares of a Sorted Array
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

    def modifyString(self, s: str) -> str:
        """
        1576 替换所有的问号
        """
        n = len(s)
        arr = list(s)
        for i in range(n):
            if arr[i] == '?':
                for ch in "abc":
                    if not (i > 0 and arr[i - 1] == ch or i < n - 1 and arr[i + 1] == ch):
                        arr[i] = ch
                        break
        return ''.join(arr)
