#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

vector<string> permutations(string s, int idx = 0) {
    vector<string> perms;
    if (s.empty() || (s.size() == idx))
        return perms;
    if ((idx + 1) == s.size()) {
        // Singleton string, all permutations are itself
        perms.push_back(string(1, s[idx]));
        return perms;
    }
        
    unordered_set<char> seen;    
    // All unique chars should play a leading role
    for (int i = idx; i < s.size(); ++i) {
        char ch = s[i];
        if (0 != seen.count(ch)) 
            continue; // 
        seen.insert(ch);
        s[i] = s[idx];
        // Not ordered since 
        auto sub_perms = permutations(s, idx + 1);
        for (auto sub_perm: sub_perms) {
            perms.push_back(ch + sub_perm);
        }
        s[i] = ch;
    }
    return perms;
}

int main() {
    auto perms = permutations("112");
    for (auto perm: perms) {
        cout << perm << endl;
    }
}
