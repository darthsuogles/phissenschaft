/**
 * Brick wall with least vertical line intersection
 */

#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

int leastBricks(vector< vector<int> > &wall) {
    if (wall.empty()) return 0;
    if (wall[0].empty()) return 0;

    unordered_map<int, int> fiss_count;
    int max_count = 0;
    for (auto &layer: wall) {
        int pref_sum = 0;
        for (auto it = layer.begin(); it + 1 != layer.end(); ++it) {
            pref_sum += *it;
            max_count = max(++fiss_count[pref_sum], max_count);
        }
    }
    return wall.size() - max_count;
}

int main() {
    vector< vector<int> > wall = {{1,2,2,1},
                                  {3,1,2},
                                  {1,3,2},
                                  {2,4},
                                  {3,1,2},
                                  {1,3,1,1}};
    cout << leastBricks(wall) << endl;
}
