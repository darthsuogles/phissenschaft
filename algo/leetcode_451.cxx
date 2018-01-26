#include <iostream>
#include <unordered_map>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

typedef pair<char, int> elem_t;

struct cmp_t { bool operator() (const elem_t& lhs, const elem_t& rhs) {
	return lhs.second < rhs.second;
}};

typedef priority_queue<elem_t, vector<elem_t>, cmp_t> pq_t;

string frequencySort(string s) {
	if (s.empty()) return s;
	unordered_map<char, int> tbl;
	string res(s.size(), '\0');  // initialize the size
	for (auto it = s.begin(); it != s.end(); ++tbl[*it++]);	
	auto pq = pq_t(tbl.begin(), tbl.end());
	auto rt = res.begin();
	while (! pq.empty()) {
		auto curr = pq.top();
		char ch = curr.first;
		for (int i = 0; i < curr.second; ++i) *rt++ = ch;
		pq.pop();
	}
	return res;
}

int main() {
	string s = "aabbbbccc";
	cout << frequencySort(s) << endl;
}
