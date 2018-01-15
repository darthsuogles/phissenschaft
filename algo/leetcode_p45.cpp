/**
 * LeetCode Problem 45
 *
 * Jump game II
 */

#include <iostream>
#include <queue>
#include <vector>

using namespace std;

class Solution {
public:
  int jump(int A[], int n)
  {
    if ( n <= 1 ) return 0;

    queue< std::pair<int, int> > q;
    q.push(std::pair<int, int>(0, 0));
    vector<bool> is_visited(n, false);
    is_visited[0] = true;
    
    while ( ! q.empty() )
      {
	std::pair<int, int> top = q.front();
	int idx = top.first;
	int depth = top.second;
	q.pop();
	int max_jump = A[idx];
	if ( idx + max_jump >= n-1 ) return depth + 1;

	for (int i = idx + 1; i <= idx + max_jump; ++i)
	  {
	    if ( is_visited[i] ) continue;
	    if ( i + A[i] >= n-1 ) return depth + 2;	    
	    q.push(std::pair<int, int>(i, depth + 1));
	    is_visited[i] = true;
	  }
      }
    return -1;
  }
};

int main()
{
  Solution sol;

  int A[] = {1,2};
  int len = sizeof(A) / sizeof(int);
  cout << sol.jump(A, len) << endl;
}
