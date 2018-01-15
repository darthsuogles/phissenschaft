/**
 * LeetCode Problem 93
 *
 * https://leetcode.com/problems/restore-ip-addresses/
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace std;

class Solution {
private:
  vector<string> validPartitions(string s, int n) {
    vector<string> res;
    if ( 0 == n || s.empty() ) return res;
    if ( 1 == n ) {
      size_t len = s.size();
      if ( len > 3 ) return res;
      if ( '0' == s[0] && len > 1 ) return res;
      
      int val = 0;
      for ( int i = 0; i < len; ++i ) {
	val = val * 10 + (s[i] - '0');
      }
      if ( val <= 255 )
	res.push_back(s);
      return res;
    }

    int len = s.size();
    int look_ahead_len = ('0' == s[0]) ? 1 : 3;
    int val = 0;
    for (int i = 0; i < min(len, look_ahead_len); ++i) {
      char ch = s[i];
      val = val * 10 + (ch - '0');
      if ( 0 <= val && val <= 255 ) {
	auto tail_partitions = validPartitions(s.substr(i+1, len), n-1);
	for (auto it = tail_partitions.begin();
	     it != tail_partitions.end(); ++it) {
	  res.push_back(s.substr(0, i+1) + "." + *it);
	}
      }
    }
    return res;
  }
  
public:
  vector<string> restoreIpAddresses(string s) {
    return validPartitions(s, 4);
  }
};


int main() {
  Solution sol;
#define test_case(s) { \
    auto res = sol.restoreIpAddresses((s));		\
    for (auto it = res.begin(); it != res.end(); ++it)	\
      cout << *it << endl;				\
    cout << "-----------------" << endl;		\
  }							
  
  test_case("25525511135");
  test_case("0000");
  test_case("010010");
}
