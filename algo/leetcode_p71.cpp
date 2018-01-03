/**
 * LeetCode Problem 71
 *
 * Simplifying path
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
  string simplifyPath(string path)
  {
    if ( path.empty() ) return path;
    int len = path.size();
    
    vector<string> dirs;
    dirs.push_back("/"); // first one must be root
    int idx = 1;    
    for (; idx < len; ++idx)
      {
	int init = idx;
	for (; idx < len; ++idx)
	  if ( '/' == path[idx] ) break;
	if ( init == idx ) // empty path, skip
	  continue;

	string curr = path.substr(init, idx - init);
	if ( "." == curr ) // skip pwd
	  continue;
	if ( ".." == curr ) // remove the last guy
	  {
	    if ( ! dirs.empty() )
	      dirs.pop_back();
	  }
	else
	  dirs.push_back(curr);
      }

    if ( dirs.empty() ) return "/";

    auto it = dirs.begin();    
    string res = "/";    
    if ( "/" == *it ) ++it;      
    for (; it + 1 < dirs.end(); ++it)
      res += *it + "/";
    if ( it < dirs.end() )
      res += *it;

    return res;
  }
};

int main()
{
  Solution sol;
#define test_case(P) cout << sol.simplifyPath((P)) << endl;
  
  test_case("/home/");
  test_case("/a/./b/../../c/");
  test_case("/");
  test_case("///");
}
