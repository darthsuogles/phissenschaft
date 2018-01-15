/**
 * LeetCode Problem 127
 *
 * Word ladder
 */

#include <string>
#include <unordered_set>
#include <queue>
#include <utility>
#include <iostream>

using namespace std;

class Solution {
public:
  int ladderLength(string start, string end, unordered_set<string> &dict)
  {
    if ( start.size() != end.size() )
      return 0;
    queue< pair<string, int> > word_queue;
    word_queue.push(pair<string, int>(start, 1));

    unordered_set<string> is_visited;    
    while ( ! word_queue.empty() )
      {
	auto wp = word_queue.front(); word_queue.pop();
	string curr = wp.first;
	if ( is_visited.count(curr) > 0 )
	  continue;
	is_visited.insert(curr);
	int dist = wp.second;
	if ( curr == end )
	  return dist;
            
	for (int i = 0; i < curr.size(); ++i)
	  {
	    for (char ch = 'a'; ch <= 'z'; ++ch)
	      {
		if ( curr[i] == ch ) continue;
		string next = curr;
		next[i] = ch;
		if ( next == end )
		  return dist+1;

		if ( dict.count(next) > 0 )
		  if ( is_visited.count(next) == 0 )
		    word_queue.push(pair<string, int>(next, dist+1));
	      }
	  }
      }
        
    return 0;
  }
};

int main()
{ 
  const char *A[] = {"hot","dot","dog","lot","log"};
  unordered_set<string> dict(A, A+5);

  Solution sol;
  cout << sol.ladderLength("hit", "cogtr", dict) << endl;
}
