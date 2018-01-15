#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace std;

class Solution {    
public:
  /**
   * This is a direct solution, kinda brute-force
   */
  vector<vector<int> > threeSum_v0(vector<int> &num)
  {
    typedef set< pair<int, int> > pair_set_t;
    map<int, pair_set_t> hash_2sum;
    int len = num.size();
    
    vector< vector<int> > triples_3sum;
    for (int i = 0; i < len; ++i)
      {
	int a = num[i];
	if ( hash_2sum.count( -a ) > 0 )
	  {
	    map<int, pair_set_t>::iterator curr_ite = hash_2sum.find( -a );
	    pair_set_t &curr_set = curr_ite->second; // map<>::iter returns a pair
	    for ( pair_set_t::iterator iter = curr_set.begin();
		  iter != curr_set.end(); ++iter)
	      {
		int v1 = iter->first, v2 = iter->second;
		int res[3];
		if ( a < v1 )
		  {
		    res[0] = a; res[1] = v1; res[2] = v2;		    
		  }
		else if ( a < v2 )
		  {
		    res[0] = v1; res[1] = a; res[2] = v2;
		  }
		else
		  {
		    res[0] = v1; res[1] = v2; res[2] = a;
		  }
		triples_3sum.push_back( vector<int>(res, res+3) );
	      }
	    hash_2sum.erase( curr_ite ); // we only need one hit instance
	  }		       

	// Add pairs involving the j-th element 
	for (int j = 0; j < i; ++j)
	  {
	    int b = num[j];
	    if ( a > b )
	      {
		int _tmp = a; a = b; b = _tmp;
	      }

	    if ( hash_2sum.count( a+b ) > 0 )
	      hash_2sum[ a+b ].insert( pair<int, int>(a, b) );
	    else
	      {
		pair_set_t _s;
		_s.insert( pair<int, int>(a, b) );
		hash_2sum[ a+b ] = _s;
	      }
	  }
      }

    // Removing duplicates
    sort(triples_3sum.begin(), triples_3sum.end());
    vector< vector<int> >::iterator ite;
    ite = unique(triples_3sum.begin(), triples_3sum.end());
    triples_3sum.resize( distance(triples_3sum.begin(), ite) );

    return triples_3sum;
  }

  // Solution 2: solve the 2sum problem and do it for each element
  void twoSum(vector<int>::iterator it_fst,
	      vector<int>::iterator it_end,
	      set< pair<int, int> > &res,
	      int target)
  {
    set<int> hash_val;
    //set< pair<int, int> > res;
    
    for ( vector<int>::iterator it = it_fst; it != it_end; ++it )
      {
	int a = *it;
	if ( hash_val.count( target - a ) > 0 )
	  {
	    hash_val.erase( hash_val.find( target - a ) );
	    int b = target - a;
	    if ( a < b )
	      res.insert( pair<int, int>(a, b) );
	    else
	      res.insert( pair<int, int>(b, a) );	    
	  }

	hash_val.insert( a );
      }

    //return res;
  }

  vector< vector<int> > threeSum_v1(vector<int> &num)
  {
    set<int> already_used;
    vector< vector<int> > triples;
    for ( vector<int>::iterator it = num.begin(); it != num.end(); ++it )
      {
	int a = *it;
	if ( already_used.count(a) > 0 )
	  continue;
	else
	  already_used.insert(a);
	
	set< pair<int, int> > pairs;
	twoSum(it+1, num.end(), pairs, -a);
	set< pair<int, int> >::iterator pt = pairs.begin();
	for ( ; pt != pairs.end(); ++pt )
	  {
	    int v1 = pt->first, v2 = pt->second;
	    int res[3];
	    if ( a < v1 )
	      {
		res[0] = a; res[1] = v1; res[2] = v2;
	      }
	    else if ( a < v2 )
	      {
		res[0] = v1; res[1] = a; res[2] = v2;		
	      }
	    else
	      {
		res[0] = v1; res[1] = v2; res[2] = a;
	      }
	    triples.push_back( vector<int>(res, res+3) );
	  }
      }

    sort(triples.begin(), triples.end());
    vector< vector<int> >::iterator it_end = unique(triples.begin(), triples.end());
    triples.resize( distance(triples.begin(), it_end) );
    return triples;
  }

  vector<vector<int> > threeSum(vector<int> &num)
  {
    vector<vector<int> > res;

    sort(num.begin(), num.end());

    for (int i = 0; i < num.size(); ++i)
      {
	int target = -num[i];
	int front = i + 1;
	int rear = num.size() - 1;

	while (front < rear)
	  {

	    int sum = num[front] + num[rear];

	    // Finding answer which start from number num[i]
	    if (sum < target)
	      ++front;

	    else if (sum > target)
	      --rear;

	    else
	      {
		vector<int> triplet(3, 0);
		triplet[0] = num[i];
		triplet[1] = num[front];
		triplet[2] = num[rear];
		res.push_back(triplet);

		// Processing duplicates of Number 2
		// Rolling the front pointer to the next different number forwards
		while (front < rear && num[front] == triplet[1]) ++front;

		// Processing duplicates of Number 3
		// Rolling the rear pointer to the next different number backwards
		while (front < rear && num[rear] == triplet[2]) --rear;
	      }

	  }

	// Processing duplicates of Number 1
	while (i + 1 < num.size() && num[i + 1] == num[i]) ++i;	  
      }
    
    return res;
  }

};

int main()
{
  Solution sol;
  //int A[] = {-1, 0, 1, 2, -1, -4, 8, -3, 2, 9, 6};
  int A[] = {7,-1,14,-12,-8,7,2,-15,8,8,-8,-14,-4,-5,7,9,11,-4,-15,-6,1,-14,4,3,10,-5,2,1,6,11,2,-2,-5,-7,-6,2,-15,11,-6,8,-4,2,1,-1,4,-6,-15,1,5,-15,10,14,9,-8,-6,4,-6,11,12,-15,7,-1,-9,9,-1,0,-4,-1,-12,-2,14,-9,7,0,-3,-4,1,-2,12,14,-10,0,5,14,-1,14,3,8,10,-8,8,-5,-2,6,-11,12,13,-7,-12,8,6,-13,14,-2,-5,-11,1,3,-6};
  int len = sizeof(A) / sizeof(int);
  vector<int> Av(A, A + len);
  vector< vector<int> > res =  sol.threeSum(Av);
  for ( int i = 0; i < res.size(); ++i )
    {
      vector<int> triple = res[i];
      for ( int k = 0; k < 2; ++k )
	cout << triple[k] << ", ";
      cout << triple[2] << endl;
    }
}
