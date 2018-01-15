#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <limits.h>

using namespace std;

class Solution {
  void lett_comb(string prefix, int pos, string &digits, vector<string> &res)
  {
    if ( digits.size() == pos )
      {
	if ( ! prefix.empty() )
	  res.push_back(prefix);
	return;
      }

    static char tbl_d2l[][4] = 
      {
	{'\0', '\0', '\0', '\0'},
	{'\0', '\0', '\0', '\0'},
	{'a', 'b', 'c', '\0'},
	{'d', 'e', 'f', '\0'},
	{'g', 'h', 'i', '\0'},
	{'j', 'k', 'l', '\0'},
	{'m', 'n', 'o', '\0'},
	{'p', 'q', 'r', 's'},
	{'t', 'u', 'v', '\0'},
	{'w', 'x', 'y', 'z'}
      };
    
    int dec = int(digits[pos] - '0');
    if ( dec < 2 ) // go directly to the next one
      lett_comb(prefix, pos + 1, digits, res);
    
    for (int i = 0; i < 4; ++i)
      {
	char ch = tbl_d2l[dec][i];
	if ( ch == '\0' ) break;

	string pref_nxt = prefix;
	pref_nxt.push_back(ch);
	lett_comb(pref_nxt, pos + 1, digits, res);
      }
  }

public:
  vector<string> letterCombinations(string digits)
  {
    vector<string> res;
    string prefix;
    lett_comb(prefix, 0, digits, res);
    return res;
  }
};

int main()
{
  Solution sol;

  vector<string> res =  sol.letterCombinations("123");
  for ( int i = 0; i < res.size(); ++i )
    cout << res[i] << endl;
}
