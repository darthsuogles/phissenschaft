/**
 * LeetCode Problem 32
 *
 * Longest valid parenthesis
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <stack>
#include <algorithm>

using namespace std;

class Solution {
public:
  /**
   * Dynamic programming 
   */
  int longestValidParentheses_DP(string s)
  {
    int n = s.size();
    //vector<int> _arr((n * n + n) / 2, 0);
    //#define tbl(i, j) _arr[ (j) * ((j) - 1) / 2 + (i) ]

    vector<int> _arr(n * (n - 1) / 2, 0);
#define tbl(i, j)							\
    _arr[ ((n - (j) + (i) + 1) + (n - 1)) * ((j) - (i) - 1) / 2 + (i) ] 
    
    int max_val = -1;
    for (int i = 0; i+1 < n; ++i)
      {
	if ( (s[i] == '(') && (s[i+1] == ')') )
	  {
	    tbl(i, i+1) = 2;
	    max_val = max(max_val, 2);
	  }
	else
	  tbl(i, i+1)= -1;
      }
      
    for (int k = 3; k < n; k += 2)
      {
	for (int i = 0; i + k < n; ++i)
	  {
	    // Table entry i, i+k
	    // tbl(i, i+k) = max( \max_j(tbl(i,j) + tbl(j+1,i+k)), tbl(i+1, i+k-1) + (s[i] ?= s[i+k]) )
	    if ( s[i] == '(' && s[i+k] == ')' )
	      {
		int mid = tbl(i+1, i+k-1);
		if ( mid != -1 )
		  {
		    mid += 2;
		    tbl(i, i+k) = mid;
		    max_val = max(max_val, mid);
		    continue;
		  }
	      }
	    
	    int val = -1;
	    for (int j = i+1; j+1 < i+k; j += 2)
	      {
		int v1 = tbl(i, j);
		if ( v1 == -1 ) continue;		  
		int v2 = tbl(j+1, i+k);
		if ( v2 == -1 ) continue;
		val = max(val, v1 + v2);
	      }

	    tbl(i, i+k) = val;
	    max_val = max(max_val, val);
	  }
      }

    return max_val;
  }

  /**
   * Stack based solution, much faster
   */
  int longestValidParentheses(string s)
  {
    stack<int> stk;
    int cnt = 0;
    int max_cnt = 0;
    for (int i = 0; i < s.size(); ++i)
      {
	char ch = s[i];
	switch (ch)
	  {
	  case '(': /* if prev is also '(', cnt must be 0 now */
	    max_cnt = max(max_cnt, cnt);
	    stk.push(cnt);
	    cnt = 0;
	    break;

	  case ')':
	    if ( stk.empty() )
	      {
		stk.push(-1); // mark an un-matched
		cnt = 0;
	      }
	    else if ( stk.top() == -1 )
	      {
		cnt = 0;
		continue;
	      }
	    else
	      {
		cnt += stk.top() + 2;
		stk.pop();
		max_cnt = max(max_cnt, cnt);
	      }
	  }
      }
    return max_cnt;
  }
};

int main()
{
  Solution sol;
#define test_case(S, ref) {			\
    int res = sol.longestValidParentheses((S)); \
    cout << S << ": " << res << endl;		\
    assert(res == ref);				\
  }						\

  test_case("(()", 2);
  test_case(")(()()", 4);
  test_case(")(()()()", 6);
  test_case(")(()()))", 6);
  test_case(")()())", 4);
  string str = ")))))((()())()))()()(()(((((()()()())))()())(()())";
  test_case(str, 26);
  // cout << str.substr(24, 26) << endl;
}
