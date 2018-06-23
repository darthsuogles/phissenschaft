#include <iostream>
#include <vector>

using namespace std;

inline bool is_leap_year(int year) {
    return (0 == year % 4) && (0 != year % 100);
}

struct Date {
    int day;
    int month;
    int year;

    int check_due(Date due) {
        if (due.year > year) return 0;
        if (due.year < year) return 10000;
        if (due.month > month) return 0;
        if (due.month < month) {
            return 500 * (month - due.month);
        }
        if (due.day < day) {
            return 15 * (day - due.day);
        }
        return 0;
    }

    friend istream& operator >>(istream &is, Date &date) {
        return is >> date.day >> date.month >> date.year;
    }

    friend ostream& operator <<(ostream &os, Date &date) {
        return os << date.day << " : " << date.month << " : " << date.year;
    }
};

int main() {
    Date returned, due;
    cin >> returned >> due;
    cout << returned.check_due(due) << endl;
}
