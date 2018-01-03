''' Longest absolute file path
'''

def lengthLongestPath(input):
    if not input: return 0
    max_len = 0
    prefix_length = {0: 0}
    for line in input.splitlines():
        fname = line.lstrip('\t')
        level = len(line) - len(fname)
        if '.' in fname:
            _len = prefix_length[level] + len(fname)
            max_len = max(_len, max_len)
        else:
            # Add the trailing slash
            prefix_length[level + 1] = prefix_length[level] + len(fname) + 1

    return max_len

def lengthLongestPathBare(input):
    ''' Input is a string with tabs and newlines 
        It appears that we can use other legit python constructs
    '''
    if not input: return 0
    n = len(input)
    path_prefix = []
    curr_line = ''
    curr_level = 0
    is_curr_file = False
    max_len = 0
    
    def get_path_len(path_seq):
        return sum(map(len, path_seq)) + len(path_seq) - 1
        
    i = 0
    while i < n:
        ch = input[i]; j = i; i += 1

        if '\n' == ch:  # last line finished
            if curr_line:
                if not path_prefix:
                    path_prefix = [curr_line]
                else:
                    path_prefix = path_prefix[:curr_level]
                    path_prefix.append(curr_line)

                if is_curr_file:
                    _len = get_path_len(path_prefix)
                    max_len = max(max_len, _len)

            # reset parser states
            curr_line = ''
            is_curr_file = False            
            continue
        
        # Find the level
        if '\t' == ch: 
            k = j
            while j < n:
                if '\t' != input[j]: break
                j += 1

            curr_level = j - k
            i = j
            continue

        if '.' == ch:
            is_curr_file = True

        curr_line += ch
    
    if not curr_line or not is_curr_file:
        return max_len

    path_prefix.append(curr_line)
    _len = get_path_len(path_prefix)
    return max(max_len, _len)


def TEST(input):
    print(lengthLongestPath(input))


TEST("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext")
TEST("dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext")
TEST("dir\n    file.txt")
TEST("file name with  space.txt")
TEST("dir\n        file.txt")
