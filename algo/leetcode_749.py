"""
Find shortest word matching license plates
"""

def shortestCompletingWord(licensePlate, words):
    wc = [0] * 26
    for ch in licensePlate:
        ch = ch.lower()
        if ch < 'a' or ch > 'z': continue
        wc[ord(ch) - ord('a')] += 1

    min_word = None
    for word in words:
        if min_word is not None and len(min_word) <= len(word):
            continue
        curr_wc = [0] * 26
        for ch in word:
            ch = ch.lower()
            if ch < 'a' or ch > 'z': continue
            curr_wc[ord(ch) - ord('a')] += 1
        for c0, c1 in zip(wc, curr_wc):
            if c0 > c1: break
        else:
            min_word = word

    return min_word


def TEST(licensePlate, words):
    print(shortestCompletingWord(licensePlate, words))


TEST("1s3 PSt", ["step", "steps", "stripe", "stepple"])
TEST("1s3 456", ["looks", "pest", "stew", "show"])
