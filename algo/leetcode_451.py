from collections import Counter

s = "aaabbbbcc"
res = ''.join(ch * cnt for (ch, cnt) in Counter(list(s)).most_common())



