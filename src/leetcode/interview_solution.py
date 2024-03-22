from typing import List


class InterviewSolution:
    def rotate_matrix(self, matrix: List[List[int]]):
        """Interview 01.07 Rotate Matrix LCCI"""

        n = len(matrix)
        for i in range(n):
            for j in range(0, i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        for i in range(n):
            matrix[i][:] = matrix[i][::-1]

    def is_flipped_string(self, s1: str, s2: str) -> bool:
        """01.09. String Rotation LCCI"""
        return len(s1) == len(s2) and s2 in (s1 + s1)

    def get_kth_magic_number(self, k: int) -> int:
        """17.09. Get Kth Magic Number LCCI"""
        dp = [0] * (k + 1)
        dp[1] = 1
        p3 = p5 = p7 = 1
        for i in range(2, k + 1):
            num3, num5, num7 = dp[p3] * 3, dp[p5] * 5, dp[p7] * 7
            dp[i] = min(num3, num5, num7)
            if dp[i] == num3:
                p3 += 1
            if dp[i] == num5:
                p5 += 1
            if dp[i] == num7:
                p7 += 1
        return dp[k]
