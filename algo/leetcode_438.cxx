#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

vector<int> findAnagrams(string s, string p) {
	vector<int> res;
	if (s.empty() || p.empty()) return res;

	auto m = s.size(), n = p.size();
	unordered_map<char, int> tbl;
	for (auto ch: p) ++tbl[ch];
	int tot = n;
	
	for (int i = 0; i < m; ++i) {
		char ch = s[i];
		if (--tbl[ch] >= 0) {
			--tot;
			if (0 == tot) res.push_back(i - n + 1);
		}
		if (i + 1 >= n) { // incr the initial of this sub
			if (++tbl[s[i - n + 1]] > 0) ++tot;
		}
	}
	return res;
}

void TEST(string s, string p, vector<int> tgt) {
	auto res = findAnagrams(s, p);
	if (tgt.size() != res.size())
		cout << "ERROR" << endl;
	int i;
	for (i = 0; i < tgt.size(); ++i) {
		if (tgt[i] != res[i]) break;
	}
	if (i < tgt.size())
		cout << "ERROR" << endl;
	else
		cout << "OK" << endl;

	cout << "RESULT: ";
	for (auto a: res) cout << a << " ";
	cout << endl;
}

int main() {
	TEST("cbaebabacd", "abc", {0, 6});	
	TEST("abab", "ab", {0, 1, 2});	
}
