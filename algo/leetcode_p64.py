class Solution:
    # @param s, a string
    # @return a boolean
    def isNumber(self, s):        
        s = s.strip()
        if len(s) == 0:
            return False

        # Parser states
        idx = 0
        is_signed = False
        is_float = False
        is_sci = False

        if s[idx] in set(['+', '-']):
            is_signed = True
            idx += 1
            if len(s) == idx:
                return False
        if '.' == s[idx]:
            is_float = True
            idx += 1
            if len(s) == idx:
                return False

        # Now this must be a digit
        if s[idx] < '0' or s[idx] > '9':
            return False
        while idx < len(s):
            if '.' == s[idx]:
                if is_float or is_sci:
                    return False
                idx += 1
                is_float = True                                
                if len(s) == idx:
                    return True
                if 'e' == s[idx]:
                    continue
                if s[idx] < '0' or s[idx] > '9':
                    return False

            elif 'e' == s[idx]:
                if is_sci:
                    return False
                idx += 1
                is_sci = True
                if len(s) == idx:
                    return False
                if s[idx] in set(['+', '-']):
                    idx += 1
                    if len(s) == idx:
                        return False
                if s[idx] < '0' or s[idx] > '9':
                    return False

            # Now must be integers
            elif s[idx] < '0' or s[idx] > '9':
                return False

            idx += 1 # increment the counter
                    
        return True

if __name__ == "__main__":
    sol = Solution()

    assert not sol.isNumber("  ")
    assert sol.isNumber("0")
    assert sol.isNumber("  0.1 ")
    assert sol.isNumber("2.")
    assert sol.isNumber("2e10")
    assert not sol.isNumber("53k")
    assert sol.isNumber("+.8")
    assert not sol.isNumber("+ 1")
    assert not sol.isNumber(".1.")
    assert sol.isNumber("46.e3")
    assert sol.isNumber(" 005047e+6 ")
    assert not sol.isNumber("4e+")
    assert not sol.isNumber("6e6.5")
    assert sol.isNumber("-3")
    assert sol.isNumber("32.e-80123")
