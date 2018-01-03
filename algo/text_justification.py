''' Pack words in a line with fixed width
'''

def textJustification(words, L):
    if not words: return []
    n = len(words)

    def fill_line(i, j, curr_line_width):
        n_spaces = L - curr_line_width
        m = j - i - 1        
        if 0 == m:
            w = words[i]
            return w + ' ' * n_spaces
        if j == n:  # last line
            return ' '.join(words[i:j]) + ' ' * n_spaces

        word_space_cnt = 1 + (n_spaces // m)
        i_max_spaces = i + (n_spaces % m)
        _line = ''
        for k in range(i, j):
            _line += words[k]
            if k + 1 == j:
                break
            _space = ' ' * word_space_cnt
            if k < i_max_spaces:
                _space += ' '
            _line += _space

        return _line

    i = 0; j = 1  # init & final positions of current line
    curr_line_width = len(words[0])
    lines = []
    while j < n:
        word = words[j]
        if curr_line_width + 1 + len(word) > L:
            line = fill_line(i, j, curr_line_width)
            curr_line_width = 0
            lines.append(line)
            #assert(len(line) == L)
            i = j

        if 0 == curr_line_width:
            curr_line_width = len(word)
        else:
            curr_line_width += 1 + len(word)
        j += 1


    line = fill_line(i, j, curr_line_width)    
    lines.append(line)
    #assert(len(line) == L)

    return lines


def TEST(words, L):
    print(textJustification(words, L))

TEST(["This", "is", "an", "example", "of", "text", "justification."], 16)
TEST(["two", "words."], 11)
