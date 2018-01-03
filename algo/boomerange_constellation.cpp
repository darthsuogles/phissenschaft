#include <iostream>
#include <unordered_map>
#include <vector>
#include <utility>

using namespace std;

int main() {
  int T; cin >> T;
  for (int idx = 1; idx <= T; ++idx) {
    int N; cin >> N;

    unordered_map< long,
		   unordered_map< int, unordered_map<int,
						     unsigned int> > > mapDistStarCnt;
    int cntBmrCnstl = 0;
    vector< pair<int, int> > listStars;
    
    while ( N-- > 0 ) {
      int x, y; cin >> x >> y;
      
      for (auto itStar = listStars.begin(); itStar != listStars.end(); ++itStar) {
	int x0 = itStar->first, y0 = itStar->second;
	long dist = (x - x0) * (x - x0); dist += (y - y0) * (y - y0);
	++mapDistStarCnt[dist][x][y];
	++mapDistStarCnt[dist][x0][y0];
      }
      listStars.push_back(make_pair(x, y));
    }
    for (auto itDist = mapDistStarCnt.begin(); itDist != mapDistStarCnt.end(); ++itDist) {
      auto mapStarXYCnt = itDist->second;
      for (auto itStarX = mapStarXYCnt.begin(); itStarX != mapStarXYCnt.end(); ++itStarX) {
	auto mapStarYCnt = itStarX->second;
	for (auto itStarY = mapStarYCnt.begin(); itStarY != mapStarYCnt.end(); ++itStarY) {
	  int nEdges = itStarY->second;
	  if ( nEdges >= 2 )
	    cntBmrCnstl += nEdges * (nEdges - 1) / 2;
	}
      }
    }
    
    cout << "Case #" << idx << ": " << cntBmrCnstl << endl;
  }
}
