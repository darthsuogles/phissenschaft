/**
 * LeetCode Problem 64
 *
 * Word justification
 */

#include <iostream>
#include <string>
#include <vector>
#include <cassert>

using namespace std;


class Solution {
public:
  vector<string> fullJustify(vector<string> &words, int L)
  {
    vector<string> res;
    if ( words.empty() ) return res;
    if ( 0 == L )
      {
	if ( words[0].empty() )
	  res.push_back("");
	return res;
      }
    if ( words[0].empty() )
      {
	res.push_back(string(L, ' '));
	return res;
      }

    auto wd = words.begin();
    while ( true )
      {
	vector<string> buf;
	int cnt = -1; // accounting for initial ghost space
	for (; wd != words.end(); ++wd )
	  {
	    if ( cnt + 1 + wd->size() > L )
	      break; // the line is full
	    buf.push_back(*wd);
	    cnt += 1 + wd->size();
	  }
	if ( wd != words.end() )
	  {
	    string line;
	    if ( buf.size() == 1 )
	      line = buf[0];
	    else if ( cnt == L ) // just fit
	      {
		auto it = buf.begin();
		for (; it+1 != buf.end(); ++it)
		  line += *it + " ";
		line += *it;
	      }
	    else
	      {
		int num_sp = buf.size() - 1;				  
		int tot_sp_len = num_sp + L - cnt;
		int sp_len = tot_sp_len / num_sp + 1;
		string space(sp_len, ' ');
		int idx = 0;
		for (; idx < (tot_sp_len % num_sp); ++idx)
		  line += buf[idx] + space;
		space.pop_back();
		for (; idx + 1 < buf.size(); ++idx)
		  line += buf[idx] + space;
		line += buf[idx];
	      }

	    while ( line.size() < L )
	      line += ' ';
	    res.push_back(line);
	    continue;
	  }
	else // the last line
	  {
	    string line;
	    auto it = buf.begin();
	    for (; it + 1 != buf.end(); ++it)
	      line += *it + " ";
	    line += *it;
	    while ( line.size() < L )
	      line += ' ';
	    res.push_back(line);
	    break;
	  }
      }

    return res;
  }
};

int main()
{
  Solution sol;

#define test_case(T, L) {						\
    vector<string> words;						\
    for (const char **cptr = (T); (*cptr)[0] != '\0'; words.push_back(*cptr++)); \
    vector<string> res = sol.fullJustify(words, (L));			\
    for (auto it = res.begin(); it != res.end(); ++it)			\
      cout << *it << "|" << endl;					\
    cout << "------------------------" << endl;				\
  }									\
  
  const char *A[] = {"This", "is", "an", "example", "of", "text", "justification.", "\0"};
  test_case(A, 16);

  const char *B[] = {"a", "\0"};
  test_case(B, 1);

  const char *C[] = {"a", "b", "c", "d", "e", "\0"};
  test_case(C, 3);

  const char *D[] = {"Listen","to","many,","speak","to","a","few.", "\0"};
  test_case(D, 6);
}
