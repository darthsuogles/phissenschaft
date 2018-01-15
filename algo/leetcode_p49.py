from collections import Counter

class Solution:
    # @param strs, a list of strings
    # @return a list of strings
    def anagrams(self, strs):
        cnt = Counter([str(sorted(s)) for s in strs])
        return [s1 for s1 in strs if cnt[str(sorted(s1))] > 1]
        

if __name__ == "__main__":
    sol = Solution()
    print( sol.anagrams(["abc", "cba", "dca", "bbc", "cpp"]) )
