class InterviewSolution:
    def is_flipped_string(self, s1: str, s2: str) -> bool:
        return len(s1) == len(s2) and s2 in (s1 + s1)
