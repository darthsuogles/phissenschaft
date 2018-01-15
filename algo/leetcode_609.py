""" Duplicate files
"""


def findDuplicate(paths):
    from collections import defaultdict
    import re
    patt = re.compile(r'(.+)\((.+)\)')
    fgrp = defaultdict(list)

    for fc in paths:
        elems = fc.split()
        fbase = elems[0]
        for el in elems[1:]:
            fnm, txt = patt.search(el).groups()
            fp = format('{}/{}').format(fbase, fnm)
            fgrp[txt].append(fp)

    
    return [grp for txt, grp in fgrp.items() if len(grp) > 1]


fls = ["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
print(findDuplicate(fls))
