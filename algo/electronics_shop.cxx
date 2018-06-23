#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int maybe_greatest_lowerbound(vector<int> &nums, int target) {
    // bisect right: we are looking for the greatest lowerbound
    int n = nums.size();
    if (0 == n) return -1;
    int i = 0, j = n - 1;
    while (i < j) {
        int k = i + (j - i) / 2;
        if (target < nums[k]) {
            j = k;
        } else {
            i = k + 1;
        }
    }
    return i - 1;
}

int main() {
    int money, num_keyboards, num_usb_drives;
    cin >> money >> num_keyboards >> num_usb_drives;
    vector<int> keyboard_prices(num_keyboards);
    for (int i = 0; i < num_keyboards; ++i) {
        cin >> keyboard_prices[i];
    }
    vector<int> usb_drive_prices(num_usb_drives);
    for (int i = 0; i < num_usb_drives; ++i) {
        cin >> usb_drive_prices[i];
    }
    sort(keyboard_prices.begin(), keyboard_prices.end());
    sort(usb_drive_prices.begin(), usb_drive_prices.end());

    int max_spending = -1;
    for (auto p: keyboard_prices) {
        int idx = maybe_greatest_lowerbound(usb_drive_prices, money - p);
        if (-1 == idx) continue;
        max_spending = max(p + usb_drive_prices[idx], max_spending);
    }
    cout << max_spending << endl;
}
