/**
 * LeetCode Problem 18: 4 sums
 */
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
  vector<vector<int> > fourSum(vector<int> &num, int target)
  {
    int len = num.size();
    vector< vector<int> > res;
    if ( len < 4 ) return res;

    sort(num.begin(), num.end());
    
    for (int i = 0; i < len-3; ++i)
      {
	int a = num[i];
	for ( int j = i+1; j < len-2; ++j )
	  {
	    int b = num[j];

	    int tg = target - a - b;
	    int front = j+1;
	    int rear = len-1;

	    while ( front < rear )
	      {
		int sum = num[front] + num[rear];
		if ( sum < tg )
		  ++front;
		else if ( sum > tg )
		  --rear;
		else // match
		  {
		    int c = num[front], d = num[rear];
		    vector<int> quads(4, 0);
		    quads[0] = a;
		    quads[1] = b;
		    quads[2] = c;
		    quads[3] = d;
		    res.push_back(quads);

		    while ( front < rear && num[front] == c ) ++front;
		    while ( front < rear && num[rear] == d ) --rear;
		  }
	      }

	    while ( j+1 < len && num[j+1] == b ) ++j;
	  }

	while ( i+1 < len && num[i+1] == a ) ++i;
      }

    return res;
  }
};

int main()
{
  Solution sol;
  int A[] = {1, 0, -1, 0, -2, 2};
  int len = sizeof(A) / sizeof(int);
  vector<int> Av(A, A + len);
  vector< vector<int> > res = sol.fourSum(Av, 0);

  for (int i = 0; i < res.size(); ++i)
    {
      vector<int> &curr = res[i];
      for (int j = 0; j < curr.size(); ++j)
	cout << curr[j] << " ";
      cout << endl;
    }
}
