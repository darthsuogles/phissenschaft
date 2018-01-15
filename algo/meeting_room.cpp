#include <iostream>
#include <utility>
#include <vector>

using namespace std;

typedef vector< pair<int, int> > schedule_t;

/**
 * Merge two (sorted) schedules
 */
schedule_t merge(const schedule_t &s1, const schedule_t &s2) {
  if ( s1.empty() ) return s2;
  if ( s2.empty() ) return s1;

  // Initialization
  schedule_t res;
  int i = 0, j = 0;
  if ( s1[0].first < s2[0].first ) {
    res.push_back(s1[0]); ++i;
  } else {
    res.push_back(s2[0]); ++j;
  }

  // Insert into resulting schedule until one array is exhausted
  int m = s1.size(), n = s2.size();
  while (true) {
    int a, b;
    if ( j == n || s1[i].first < s2[j].first ) {
      a = s1[i].first;
      b = s1[i].second;
      ++i;
    } else if ( i == m || s1[i].first >= s2[j].first ) {
      a = s2[j].first;
      b = s2[j].second;
      ++j;
    }

    // Merge current time interval with the last 
    if ( a > res.back().second ) {
      res.push_back(make_pair(a, b));
    } else {
      a = res.back().first;
      b = max(b, res.back().second);
      res.pop_back();
      res.push_back(make_pair(a, b));
    }
    if ( i == m || j == n ) break;
  }

  // Insert the last elements
  for (; i < m; ++i) res.push_back(s1[i]);
  for (; j < n; ++j) res.push_back(s2[j]);

  return res;
}

schedule_t getFreeTimeSlots(vector<schedule_t> schedules) {
  // Merge all intervals
  schedule_t res;
  if ( schedules.empty() ) return res;
  for (auto it = schedules.begin(); it != schedules.end(); ++it) {
    res = merge(res, *it);
  }

  // Get free slots in-between
  schedule_t res_empty;
  for (int i = 0; i+1 < res.size(); ++i) {
    int a = res[i].second;
    int b = res[i+1].first;
    res_empty.push_back(make_pair(a, b));
  }
  return res_empty;
}

void print(schedule_t schd) {
  for (auto it = schd.begin(); it != schd.end(); ++it) {
    cout << "["
	 << it->first << ", "
	 << it->second << "] ";
  }
  cout << endl;
}

int main() {
  vector< schedule_t > schedules = {
    {{1,3}, {6,7}},
    {{2,4}},
    {{2,3}, {9,12}},
  };

  print(getFreeTimeSlots(schedules));
}
