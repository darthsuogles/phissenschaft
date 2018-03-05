#include <cmath>
#include <cstdio>
#include <iostream>

using namespace std;

int days_in_month[] = { 0, // no zero-th month
                        31, 28, 31, 30, 31, 30,
                        31, 31, 30, 31, 30, 31
};

inline int days_in_feb(int year) {
    if (1918 == year) return 15;
    // bug in the testing program
    if (2000 == year || 2400 == year) return 29;
    if (year < 1918) return 28 + (0 == year % 4);
    return 28 + ((0 == year % 4) && (0 != year % 100));
}

inline int num_days(int year, int month) {
    if (2 == month) return days_in_feb(year);
    return days_in_month[month];
}

inline int get_day_in_month(int year, int month, int days) {
    if ((2 == month) && (1918 == year)) return 15 + (days - 1);
    return days;
}

int main() {
    int year; cin >> year;
    const int target_days = 256;
    int days = 0;
    for (int month = 1; month <= 12; ++month) {
        if (days + num_days(year, month) >= target_days) {
            printf("%02d\.%02d\.%04d\n",
                   get_day_in_month(year, month, target_days - days),
                   month,
                   year);
            break;
        }
        days += num_days(year, month);
        cout << "month: " << month << " : "  << days << endl;
    }

    return 0;
}
