/**
 * LeetCode Problem 207
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
  bool canFinish(int numCourses, vector< pair<int, int> >& prerequisites)
  {
    if ( 0 == numCourses ) return true;

    // Create a dependency graph
    unordered_map< int, unordered_set<int> > graph_deps;
    //vector< unordered_set<int> > graph_deps(numCourses);
    vector<int> num_preqs(numCourses, 0);
    for (auto pt = prerequisites.begin(); pt != prerequisites.end(); ++pt)
      {
	int cour = pt->first;
	int preq = pt->second;

	if ( graph_deps[preq].count(cour) == 0 )
	  {
	    graph_deps[preq].insert(cour);	
	    ++num_preqs[cour];
	  }
      }
    
    int num_unlocked = 0;
    queue<int> cour_q;
    for (int cour = 0; cour < numCourses; ++cour)
      {
	if ( num_preqs[cour] != 0 ) continue;
	++num_unlocked;
	cour_q.push(cour);
      }

    while ( ! cour_q.empty() )
      {
	int cour = cour_q.front(); cour_q.pop();
	unordered_set<int> &curr_deps = graph_deps[cour];
	for (auto ct = curr_deps.begin(); ct != curr_deps.end(); ++ct)
	  if( --num_preqs[*ct] < 1 )
	    {
	      ++num_unlocked;
	      cour_q.push(*ct);
	    }
      }

    return ( num_unlocked == numCourses );
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
    cout << sol.canFinish(num_cours, cours_preqs) << endl;	}	\

  int A[][2] = {{1,0}, {2,0}};
  test_case(3, A);

  int B[][2] = {{1,0}, {0,1}};
  test_case(2, B);

  int C[][2] = {{5,8},{3,5},{1,9},{4,5},{0,2},{1,9},{7,8},{4,9}};
  test_case(10, C);
}

