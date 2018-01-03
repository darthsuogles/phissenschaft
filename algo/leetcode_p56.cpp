/**
 * LeetCode Problem 56
 *
 * Merge overlapping intervals
 */

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
struct Interval {
  int start;
  int end;
  Interval() : start(0), end(0) {}
  Interval(int s, int e) : start(s), end(e) {}
};

struct CmpIntervals {
  bool operator() (Interval &a, Interval &b) { return a.start < b.start; }
} cmp_intervals;

class Solution {
public:
  vector<Interval> merge(vector<Interval> &intervals)
  {
    vector<Interval> res;
    int n = intervals.size();
    if ( 0 == n ) return res;
    if ( 1 == n ) return intervals;
    
    sort(intervals.begin(), intervals.end(), cmp_intervals);
    int idx = 0;
    int p = intervals[0].start;
    int q = intervals[0].end;
    for (; idx < n; ++idx)
      {
	int pi = intervals[idx].start;
	int qi = intervals[idx].end;
	if ( pi > q ) // sum up the last interval
	  {
	    res.push_back(Interval(p, q));
	    p = pi;
	    q = qi;
	  }
	else if ( qi > q )
	  q = qi;
      }
    res.push_back(Interval(p, q));
    return res;
  }
};

int main()
{
  Solution sol;
  int A[][2] = {{1,3}, {2,6}, {8,10}, {15, 18}};
  int len = 4;
  vector<Interval> intervals;
  for (int i = 0; i < len; ++i)
    intervals.push_back(Interval(A[i][0], A[i][1]));

  vector<Interval> res = sol.merge(intervals);
  for (auto it = res.begin(); it != res.end(); ++it)
    cout << "[" << it->start << ", " << it->end << "] ";
  cout << endl;
}
