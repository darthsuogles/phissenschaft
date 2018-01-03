#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int last_pos_tbl[256];
	for (int i=0; i < 256; last_pos_tbl[i++] = -1);
	
        int curr_len = 0, max_len = 0;
        for (int i=0; i < s.size(); ++i)
        {
            int ch = int(s[i]);
            int last_pos = last_pos_tbl[ch];
            if (-1 == last_pos)
            {
                curr_len += 1;
                if ( curr_len > max_len )
                    max_len = curr_len;
            }
            else
	    {
                curr_len = i - last_pos;
		for (int j=0; j < 256; ++j)
		  if ( last_pos_tbl[j] < last_pos )
		    last_pos_tbl[j] = -1;
	    }
            last_pos_tbl[ch] = i;
        }
        return max_len;
    }
};

int main()
{
  Solution sol;
  vector<string> test_cases;
  test_cases.push_back("c");
  test_cases.push_back("abcabcbb");
  test_cases.push_back("bbbbbb");
  test_cases.push_back("tmmzuxt");

  for ( vector<string>::iterator iter = test_cases.begin();
	iter != test_cases.end(); ++iter )
    cout << *iter << " : " << sol.lengthOfLongestSubstring(*iter) << endl;
}
