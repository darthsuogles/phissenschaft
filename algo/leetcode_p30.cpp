/**
 * LeetCode Problem 30
 *
 * Substring with concatenation of all strings in a collection
 * Three solutions are provided.
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>

using namespace std;

class Solution {
  /**
   * The brute-force method (didn't pass the time limit)
   */
private:
  bool match_str_elems(string S, int pos, vector<string> &L, vector<bool> &is_used)
  {
    int K = L.size();
    int n = L[0].size(); // L is never empty
    int len = S.size();    

    if ( pos > len ) return false;

    int L_cnt = 0; // check if everything in L is used
    for (int i = 0; i < K; ++i)
      if (is_used[i]) ++L_cnt;
    if ( L_cnt == K )
      return true;

    if ( pos > len - n ) return false;

    // Search linearly over the string repo
    string curr = S.substr(pos, n);
    for (int i = 0; i < K; ++i) 
      {
	if ( is_used[i] ) continue;
	if ( curr != L[i] ) continue;

	is_used[i] = true;
	bool res =  match_str_elems(S, pos+n, L, is_used);
	is_used[i] = false;
	if ( res ) return true;
      }
    return false;
  }
  
public:
  vector<int> findSubstring_v0(string S, vector<string> &L)
  {
    vector<int> res;
    if ( L.empty() || S.empty() ) return res;

    int K = L.size(); 
    int n = L[0].size(); // all strings in L have the same size
    int len = S.size();

    // Only have to use searching indices at 0, 1, ..., n-2
    for (int init = 0; init < n - 1; ++init) 
      {
	for (int i = init; i < len; i += n)
	  {
	    vector<bool> is_used(K, false); // initialized to all zero
	    if ( match_str_elems(S, i, L, is_used) )
	      res.push_back(i); // just need one instance
	  }
      }

    return res;
  }

  /**
   * Using a map to expediate searching
   */  
private:
  /**
   * Returns the first match position after pos given (num_matched, vocab_tbl)
   */
  int match_at_pos(string &S, int pos, 
		   vector<string> &L,
		   int num_matched, /* number of matched substrings till before pos */
		   unordered_map<string, int> vocab_tbl)
  {
    int len = S.size();
    if ( pos > len ) return -1;    
    int K = L.size();
    int n = L[0].size();
    if ( num_matched == K ) return pos - K * n;
    if ( pos + (K - num_matched) * n > len ) return -1;
    
    int idx = pos, nm = num_matched;    
    for (; idx <= len - n; idx += n)
      {	
	string curr = S.substr(idx, n);
	if ( vocab_tbl[curr] > 0 )
	  {
	    --vocab_tbl[curr];
	    ++nm;
	    if ( nm == K )
	      return idx + n - K * n;
	  }
	else // not matched
	  {
	    for (; nm > 0; --nm)
	      {
		string foremost = S.substr(idx - nm * n, n);
		if ( foremost == curr )
		  break;
		++vocab_tbl[foremost];
	      }
	  }
      }

    return -1;
  }

public:
  vector<int> findSubstring_v1(string S, vector<string> &L)
  {
    vector<int> res;
    if ( S.empty() || L.empty() ) return res;
    int len = S.size();
    int K = L.size();
    int n = L[0].size();
    unordered_map<string, int> vocab_tbl;
    for ( vector<string>::iterator sit = L.begin();
	  sit != L.end(); ++sit )
      ++vocab_tbl[*sit];
    
    for (int init = 0; init < n; ++init)
      {
	int idx = init;
	while ( idx < len )
	  {
	    int pos_m = match_at_pos(S, idx, L, 0, vocab_tbl);
	    if ( -1 == pos_m ) break;
	    res.push_back(pos_m);
	    idx = pos_m + n; // start from the next word
	  }
      }
    
    return res;
  }

  /**
   * Method 3
   */
private:
  int search_str(string &curr, vector<string> &L, vector<bool> &is_used,
		 bool return_something = false)
  {
    std::pair< vector<string>::iterator, vector<string>::iterator > bounds;
    bounds = equal_range(L.begin(), L.end(), curr);
    
    vector<string>::iterator it = bounds.first;
    for (; it != bounds.second; ++it)
      if ( ! is_used[ it - L.begin() ] )
	break;

    if ( it != bounds.second )
      return it - L.begin();
    else if ( return_something )
      return bounds.first - L.begin();
    else 
      return -1;
  }
  
  int match_at_pos(string &S, int pos,
		   vector<string> &L, int num_matched,
		   vector<bool> is_used)
  {
    int len = S.size();
    if ( pos > len ) return -1;
    int K = L.size();
    int n = L[0].size();
    if ( num_matched == K ) return pos - n * K;
    if ( pos + (K - num_matched) * n > len ) return -1;

    int idx = pos, nm = num_matched;
    for (; idx <= len - n; idx += n)
      {
	string curr = S.substr(idx, n);
	int k = search_str(curr, L, is_used, false);
	if ( k != -1 ) // matched
	  {
	    is_used[k] = true;
	    ++nm;
	    if ( nm == K )
	      return idx + n - n * K;		
	  }
	else // not matched
	  {
	    for (; nm > 0; --nm)
	      { 
		string foremost = S.substr(idx - nm * n, n);
		if ( foremost == curr )
		  break;
		int k = search_str(foremost, L, is_used, true);
		is_used[k] = false;
	      }
	  }
      }
    return -1;
  }

public:
  vector<int> findSubstring_v2(string S, vector<string> &L)
  {
    vector<int> res;
    if ( S.empty() || L.empty() ) return res;

    int len = S.size();
    int K = L.size();
    int n = L[0].size();

    sort(L.begin(), L.end());
    
    for (int init = 0; init < n; ++init)
      {
	int idx = init;
	do
	  {
	    vector<bool> is_used(K, false);
	    int pos_m = match_at_pos(S, idx, L, 0, is_used);
	    if ( -1 == pos_m ) break;
	    res.push_back(pos_m);
	    idx = pos_m + n;
	  }
	while ( idx < len );
	
      }
    return res;
  }

public:
  vector<int> findSubstring(string S, vector<string> &L)
  {
    return findSubstring_v1(S, L);
    //return findSubstring_v2(S, L); // this is slower
  }
};

int main()
{
  Solution sol;
#define test_case(S, _Li) {				\
    string _S = (S);					\
    vector<string> _L;					\
    int K = sizeof(_Li) / sizeof(_Li[0]);		\
    for (int i = 0; i < K; ++i) _L.push_back(_Li[i]);	\
    vector<int> res = sol.findSubstring(_S, _L);	\
    for (int i = 0; i < res.size(); ++i)		\
      cout << res[i] << " ";				\
    if ( res.empty() ) cout << "none" << endl;		\
    else cout << endl;					\
  }							\
  

  const char *A[] = {"foo", "bar"};
  test_case("barfoothefoobarman", A);
  const char *A0[] = {"a"};
  test_case("a", A0);
  const char *A1[] = {"a", "a"};
  test_case("aaaa", A1);
  const char *A2[] = {"a", "b"};
  test_case("aaa", A2);
  const char *A3[] = {"ab", "ba"};
  test_case("abababab", A3);
  const char *A4[] = {"a","b","a"};
  test_case("abababab", A4);
}
