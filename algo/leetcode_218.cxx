/**
 * The skyline problem
 */
#include <iostream>
#include <vector>
#include <utility>
#include <cassert>
#include <queue>
#include <algorithm>
#include <tuple>

using namespace std;

class Solution {
public:
    vector<pair<int, int>> getSkyline(vector<vector<int>> &buildings) {
        // Build a event sequence
        using event_t = tuple<int, int, int>;
        vector<event_t> events;
        for (auto lrh: buildings) {
            assert(lrh.size() == 3);
            int l = lrh[0], r = lrh[1], h = lrh[2];
            events.push_back(make_tuple(l, r, h));
            events.push_back(make_tuple(r, -1, h));
        }

        sort(events.begin(), events.end(),
             [](event_t e1, event_t e2) { return get<0>(e1) < get<0>(e2); });

        // Use pairs of <height, right>
        priority_queue<pair<int, int>> tall_overlay; // max-heap

        vector<pair<int, int>> landmarks;

        for (auto event: events) {
            int pos = get<0>(event);
            int where_ends = get<1>(event);
            int height = get<2>(event);

            while (!tall_overlay.empty()) {
                int top_where_ends = get<1>(tall_overlay.top());
                if (top_where_ends > pos) break;
                tall_overlay.pop();
            }

            if (-1 != where_ends) {
                tall_overlay.push(make_pair(height, where_ends));
            }
            int top_height = tall_overlay.empty() ? 0 : get<0>(tall_overlay.top());

            if (top_height > height)
                continue;
            auto mark = make_pair(pos, top_height);
            // Insert if we haven't collected any results
            if (landmarks.empty()) {
                landmarks.push_back(mark); continue;
            }
            // If previous landmark has an overlapping position, we know it is
            // gonna be shorter, thus remove it
            int prev_pos = get<0>(landmarks[landmarks.size() - 1]);
            if (prev_pos == pos)
                landmarks.pop_back();
            if (landmarks.empty()) {
                landmarks.push_back(mark); continue;
            }
            // If the previous landmark has the same height, then we
            // won't bother adding this new one
            int prev_height = get<1>(landmarks[landmarks.size() - 1]);
            if (prev_height != top_height)
                landmarks.push_back(mark);
        }
        return landmarks;
    }
};

Solution sol;

void TEST(vector<vector<int>> buildings) {
    cout << "--------" << endl;
    auto res = sol.getSkyline(buildings);
    for (auto landmark: res) {
        cout << landmark.first << " | " << landmark.second << endl;
    }
}

int main() {
    TEST({{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}});
    TEST({{0, 2, 3}, {2, 5, 3}});
}
