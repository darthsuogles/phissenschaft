#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
  string longestCommonPrefix(vector<string> &strs) {
    if ( strs.size() == 0 ) return "";
    
    string prefix;
    bool eq = true;
    int idx = 0;
        
    for ( ; ; ++idx )
      {
	if ( strs[0].size() == idx ) 
	  break;

	char ch = strs[0][idx];
	bool idx_full = false;
	for (int i = 1; i < strs.size(); ++i)
	  {
	    if ( strs[i].size() == idx )
	      {
		idx_full = true;
		eq = false;
		break;
	      }
                
	    if ( strs[i][idx] != ch )
	      {
		eq = false;
		break;
	      }
	  }
	if ( idx_full ) 
	  break;
	if ( ! eq )
	  break;
	prefix.push_back(ch);
      }
        
    return prefix;
  }
};

int main()
{
  Solution sol;
  vector<string> strs;
  strs.push_back("ab");
  strs.push_back("b");

  cout << sol.longestCommonPrefix(strs) << endl;
}
