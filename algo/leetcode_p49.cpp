/**
 * LeetCode Problem 49
 *
 * Anagrams
 */

#include <iostream>
//#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
  vector<string> anagrams(vector<string> &strs)
  {
    vector<string> res;
    if ( strs.size() < 2 ) return res;
    
    unordered_map< string, vector<string> > anagram_tbl;
    for (auto it = strs.begin(); it != strs.end(); ++it)
      {
	string hash_key = *it;
	sort(hash_key.begin(), hash_key.end());
	int cnt = anagram_tbl.count(hash_key);
	if ( cnt > 0 )
	  anagram_tbl[hash_key].push_back(*it);
	else
	  anagram_tbl[hash_key] = vector<string>(1, *it);
      }

    for (auto it = anagram_tbl.begin(); it != anagram_tbl.end(); ++it)
      {
	if ( it->second.size() <= 1 ) continue;
	res.insert(res.end(), it->second.begin(), it->second.end());
      }
    return res;
  }
};

int main()
{
  Solution sol;
  vector<string> strs;
  //strs.push_back("aabc");
  //strs.push_back("abca");
  //strs.push_back("aabcaa");
  strs.push_back("tea");
  strs.push_back("and");
  strs.push_back("ate");
  strs.push_back("eat");
  strs.push_back("dan");
  

  vector<string> res = sol.anagrams(strs);
  for (auto it = res.begin(); it != res.end(); ++it)
    cout << *it << endl;
}
