/**
 * LeetCode Problem 187
 *
 * Repeated length-10 DNA sub-sequences
 */

#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

template <int LEN>
size_t seq_to_bitvec(const string &str)
{
  size_t res = 0;    
#pragma unroll 
  for (int i = 0; i < LEN; ++i)
    {
      int val;
      switch ( str[i] )
	{
	case 'A':
	  val = 0; break;
	case 'T':
	  val = 1; break;
	case 'C':
	  val = 2; break;
	case 'G':
	  val = 3; break;
	}
      res |= val;
      res <<= 2;
    }

  return res;
}

class Solution {
public:
  vector<string> findRepeatedDnaSequences(string s)
  {
    vector<string> res;
    if ( s.empty() ) return res;
        
    unordered_map<size_t, int> seq_dict;
    int len = s.size();
    const int sub_len = 10;
    for (int i = 0; i + sub_len <= len; ++i)
      {
	string curr = s.substr(i, sub_len);
	size_t hval = seq_to_bitvec<sub_len>(curr);
	if ( 2 == ++seq_dict[hval] ) // store the second copy only
	  res.push_back(curr);
      }

    return res;
  }
};

int main()
{
  Solution sol;

#define test_case(S) {						\
    string str = (S);						\
    vector<string> res = sol.findRepeatedDnaSequences(str);	\
    for (auto st = res.begin(); st != res.end(); ++st)		\
      cout << *st << endl;					\
    cout << "------------------" << endl; }			\

  test_case("AAAAAAAAAAAA");
  test_case("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT");
}
