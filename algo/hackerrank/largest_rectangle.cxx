#include <iostream>
#include <vector>

using namespace std;

int main() {
    int N; cin >> N;
    vector<int> buildings(N + 2);
    for (int i = 1; i <= N; cin >> buildings[i++]);
    buildings[0] = -2; buildings[N + 1] = -1;

    vector<int> incr_seq(1, 0);  // increasing order
    int max_area = 0;
    for (int j = 1; j < buildings.size(); ++j) {
        int curr = buildings[j];
        while (true) {
            int k = incr_seq.back();
            if (buildings[k] < curr) break;
            incr_seq.pop_back();
            int i = incr_seq.back();
            int area = (j - i - 1) * buildings[k];
            max_area = max(area, max_area);
        }
        incr_seq.push_back(j);
    }
    cout << max_area << endl;
}
