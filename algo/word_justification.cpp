#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;

// /* A rather concise solution is attached */
// vector<string> fullJustify(vector<string> &words, int L) {
//     vector<string> res;
//     for(int i = 0, k, l; i < words.size(); i += k) {
//         for(k = l = 0; i + k < words.size() and l + words[i+k].size() <= L - k; k++) {
//             l += words[i+k].size();
//         }
//         string tmp = words[i];
//         for(int j = 0; j < k - 1; j++) {
//             if(i + k >= words.size()) tmp += " ";
//             else tmp += string((L - l) / (k - 1) + (j < (L - l) % (k - 1)), ' ');
//             tmp += words[i+j+1];
//         }
//         tmp += string(L - tmp.size(), ' ');
//         res.push_back(tmp);
//     }
//     return res;
// }


class Solution {
public:
    vector<string> fullJustify(vector<string>& words, int maxWidth) {
        vector<string> res;
        if ( words.empty() ) return res;
                
        int n = words.size();
        int idx = 1, line_init = 0, line_size = words[0].size();
        for (; idx < n; ++idx) {
            string word = words[idx];
            if (line_size + 1 + word.size() > maxWidth) {
                // Current line reached full size
                string line(maxWidth, ' ');
                int num_words = idx - line_init;
                if (1 == num_words) {
                    int j = 0;
                    for (auto a: words[line_init]) 
                        line[j++] = a;
                } else {
                    int num_extra = maxWidth - line_size;  // extra spaces
                    int num_spaces = (num_words - 1 + num_extra) / (num_words - 1);
                    int r = num_extra % (num_words - 1);                    
                    int i = 0, j = 0;
                    for (; i < num_words && j < maxWidth; ++i) {
                        for (auto a: words[i + line_init])
                            line[j++] = a;
                        j += num_spaces + int(i < r);
                    }
                }
                res.push_back(line);
                line_init = idx;
                line_size = word.size();                
            } else {
                line_size += 1 + word.size();
            }
        }
        // Last line, all words pushed to left
        if (line_init < n) {
            string line(maxWidth, ' ');
            int i = 0, j = 0;
            int num_words = n - line_init;
            for (; i < num_words; ++i) {
                for (auto a: words[i + line_init]) 
                    line[j++] = a;
                ++j; // only one space
            }
            res.push_back(line);
        }        
        return res;
    }
};

void testCase(vector<string> words, const int maxWidth) {
    static Solution sol;
    auto res = sol.fullJustify(words, maxWidth);
    for (auto it = res.begin(); it != res.end(); ++it) {
        auto line = *it;
        int i = 0;
        int n = line.size();
        cout << n << ": ";
        for (; i < min(n, maxWidth); ++i)
            cout << line[i];
        cout << "$";
        for (; i < maxWidth; ++i)
            cout << " ";
        cout << "|";
        for (; i < n; ++i)
            cout << line[i];
        cout << endl;
    }
    cout << "-----------" << endl;
}

int main() {
    testCase({"What","must","be","shall","be."}, 5);
    testCase({"a", "b", "c", "d", "e"}, 3);
    testCase({"This", "is", "an", "example", "of", "text", "justification."}, 16);
    testCase({"Listen","to","many,","speak","to","a","few."}, 6);
    testCase({"When","I","was","just","a","little","girl","I","asked","my","mother","what","will","I","be","Will","I","be","pretty","Will","I","be","rich","Here's","what","she","said","to","me","Que","sera","sera","Whatever","will","be","will","be","The","future's","not","ours","to","see","Que","sera","sera","When","I","was","just","a","child","in","school","I","asked","my","teacher","what","should","I","try","Should","I","paint","pictures","Should","I","sing","songs","This","was","her","wise","reply","Que","sera","sera","Whatever","will","be","will","be","The","future's","not","ours","to","see","Que","sera","sera","When","I","grew","up","and","fell","in","love","I","asked","my","sweetheart","what","lies","ahead","Will","there","be","rainbows","day","after","day","Here's","what","my","sweetheart","said","Que","sera","sera","Whatever","will","be","will","be","The","future's","not","ours","to","see","Que","sera","sera","What","will","be,","will","be","Que","sera","sera..."}, 60);
}
