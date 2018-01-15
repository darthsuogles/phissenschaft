#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <limits.h>
#include <algorithm>

using namespace std;

class Solution {
public:
  int threeSumClosest(vector<int> &num, int target)
  {
    int min_dist = INT_MAX;
    int best_sum = INT_MAX;    

    sort(num.begin(), num.end());

    for (int i = 0; i < num.size(); ++i)
      {
	int tg = -num[i] + target;
	int front = i + 1;
	int rear = num.size() - 1;

	while (front < rear)
	  {
	    int sum = num[front] + num[rear];
	    if ( abs(sum - tg) < min_dist )
	      {
		min_dist = abs(sum - tg);
		best_sum = sum - tg + target;
		if ( 0 == min_dist )
		  return best_sum;
	      }

	    // Finding answer which start from number num[i]
	    if (sum < tg)
	      ++front;

	    else if (sum > tg)
	      --rear;
	  }

	// Processing duplicates of Number 1
	while (i + 1 < num.size() && num[i + 1] == num[i]) ++i;	  
      }
    
    return best_sum;
  }
};

int main()
{
  Solution sol;

  int A[] = {-1, 2, 1, -4};
  int len = sizeof(A) / sizeof(int);
  vector<int> Av(A, A + len);
  int target = 1;

  cout << sol.threeSumClosest(Av, target) << endl;

}
