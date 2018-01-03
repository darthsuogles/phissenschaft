/**
 * LeetCode Problem 210
 *
 * Course schedule
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <utility>

using namespace std;

class Solution {
public:
  vector<int> findOrder(int numCourses, vector< pair<int, int> >& prerequisites)
  {
    vector<int> res;
    if ( 0 == numCourses ) return res;

    vector<int> num_preqs(numCourses, 0);
    unordered_map< int, unordered_set<int> > graph_deps;
    for (auto uv = prerequisites.begin(); uv != prerequisites.end(); ++uv)
      {
	// Course u has a prerequisite v
	int u = uv->first, v = uv->second;
	if ( graph_deps[v].count(u) == 0 )
	  {
	    graph_deps[v].insert(u);
	    ++num_preqs[u];
	  }
      }

    int num_unlocked = 0;
    queue<int> cour_q;
    for (int cour = 0; cour < numCourses; ++cour)
      if ( 0 == num_preqs[cour] )
	{
	  cour_q.push(cour);
	  ++num_unlocked;
	}
    
    while ( ! cour_q.empty() )
      {
	int cour = cour_q.front(); cour_q.pop();
	res.push_back(cour);

	unordered_set<int> &curr_deps = graph_deps[cour];
	for (auto it = curr_deps.begin(); it != curr_deps.end(); ++it)
	  {
	    int next_cour = *it;
	    if ( --num_preqs[next_cour] == 0 )
	      {
		++num_unlocked;
		cour_q.push(next_cour);
	      }
	  }
      }

    if ( num_unlocked < numCourses )
      return vector<int>();
    return res;
  }
};

int main()
{
  Solution sol;
  
#define test_case(NC, D) {						\
    int num_cours = (NC);						\
    vector< pair<int, int> > cours_preqs;				\
    int sz = sizeof((D)) / sizeof((D)[0]);				\
    for (int i = 0; i < sz; ++i)					\
      cours_preqs.push_back( pair<int, int>((D)[i][0], (D)[i][1]) );	\
    vector<int> res = sol.findOrder(num_cours, cours_preqs);		\
    if ( res.empty() ) cout << "Empty";					\
    else								\
      for (int i = 0; i < res.size(); ++i) cout << res[i] << " ";	\
    cout << endl;	}						\

  int A[][2] = {{1,0}, {2,0}};
  test_case(3, A);

  int B[][2] = {{1,0}, {0,1}};
  test_case(2, B);

  int C[][2] = {{5,8},{3,5},{1,9},{4,5},{0,2},{1,9},{7,8},{4,9}};
  test_case(10, C);

  int D[][2] = {{1,0},{2,0},{3,1},{3,2}};
  test_case(4, D);
}
