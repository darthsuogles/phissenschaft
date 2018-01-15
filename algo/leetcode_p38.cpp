/**
 * LeetCode problem 38
 *
 * Count and say
 */

#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
public:
  string countAndSay(int n)
  {
    int idx = 1;
    string seq = "1";
    for (; idx < n; ++idx)
      {
	string nxt = "";
	int len = seq.size();
	seq += '\0'; // add padding to simplify the loop
	int conseq = 1;
	for (int i = 0; i < len; ++i)
	  {
	    if ( seq[i] == seq[i+1] )
	      ++conseq;
	    else
	      {
		int a = conseq;
		string num;
		while ( a > 0 )
		  {
		    num += char((a % 10) + int('0'));
		    a /= 10;
		  }
		for (int k = 0; k < num.size()/2; ++k)
		  {
		    char tmp = num[k];
		    num[k] = num[n-1-k];
		    num[n-1-k] = tmp;
		  }
		nxt += num + seq[i];
		conseq = 1;
	      }
	  }
	seq = nxt;
      }
    return seq;
  }
};

int main()
{
  Solution sol;
  for (int i = 1; i < 10; ++i)
    cout << sol.countAndSay(i) << endl;
}
