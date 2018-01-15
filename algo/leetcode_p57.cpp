/**
 * LeetCode Problem 57
 *
 * Insert interval
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
  vector<Interval> insert(vector<Interval> &intervals, Interval newInterval)
  {
    vector<Interval> res;
    int n = intervals.size();
    if ( 0 == n )
      {
	res.push_back(newInterval);
	return res;
      }

    int p = intervals[0].start;
    int q = intervals[0].end;
    int a = newInterval.start;
    int b = newInterval.end;

    if ( b < p ) // insert into the initial position
      {
	res.push_back(newInterval);
	res.insert(res.end(), intervals.begin(), intervals.end());
	return res;
      }
    
    int idx = 0;
    for (; idx < n; ++idx)
      {
	int pi = intervals[idx].start;
	int qi = intervals[idx].end;
	if ( qi < a )
	  res.push_back(intervals[idx]);
	else
	  break;
      }

    if ( n == idx )
      {
	res.push_back(newInterval);
	return res;
      }

    // Now: a .. qi, merge these two
    p = min(a, intervals[idx].start);
    q = b;
    for (; idx < n; ++idx)
      {
	int pi = intervals[idx].start;
	int qi = intervals[idx].end;
	if ( pi > q )
	  break;
	q = max(qi, q);
      }
    res.push_back(Interval(p, q));

    for (; idx < n; ++idx)
      res.push_back(intervals[idx]);
    
    return res;
  }
};

int main()
{
  Solution sol;

  int A[][2] = {{1,3}, {6,9}};
  int a[2] = {2,5};

#define test_case(A, B)  {					\
    int len = sizeof(A) / sizeof(int[2]);			\
    vector<Interval> intervals;					\
    for (int i = 0; i < len; ++i)				\
      intervals.push_back(Interval((A)[i][0], (A)[i][1]));	\
    Interval new_interval((B)[0], (B)[1]);			\
								\
    vector<Interval> res = sol.insert(intervals, new_interval); \
    for (auto it = res.begin(); it != res.end(); ++it)		\
      cout << "[" << it->start << ", " << it->end << "] ";	\
    cout << endl;						\
  }								\

  test_case(A, a);

  int B[][2] = {{1,2},{3,5},{6,7},{8,10},{12,16}};
  int b[2] = {4,9};
  test_case(B, b);

  int C[][2] = {{1,5}};
  int c[2] = {6,8};
  test_case(C, c);

  int D[][2] = {{1,5}};
  int d[2] = {0,0};
  test_case(D, d);

  int E[][2] = {{3,5}, {12,15}};
  int e[2] = {6,6};
  test_case(E, e);
}
