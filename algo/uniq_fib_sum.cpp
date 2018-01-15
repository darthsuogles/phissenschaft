#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

bool findUniqSetSum(int a, int idx,
		    vector<int> &seq_val,
		    vector<bool> &seq_chosen,
		    bool is_reinit = false) {
  if ( 0 == a ) return true;
  if ( a < 0 || idx < 0 ) return false;

  static unordered_map<int, unordered_set<int> > is_idx_visited;
  if ( is_reinit ) is_idx_visited.clear();
  if ( is_idx_visited.count(idx) > 0 ) {
    auto invalid_val_set = is_idx_visited[idx];
    if ( invalid_val_set.count(a) > 0 )
      return false;
  }
      
  seq_chosen[idx] = true;
  if ( findUniqSetSum(a - seq_val[idx], idx - 1, seq_val, seq_chosen) )
    return true;
  seq_chosen[idx] = false;
  if ( findUniqSetSum(a, idx - 1, seq_val, seq_chosen) )     
    return true;
  is_idx_visited[idx].insert(a);
  return false;
}

bool uniqFibSum(int a) {
  int f_prev = 1, f_curr = 1;
  vector<int> fib_seq = {1};
  for (; f_curr < a; ) {
    int f_next = f_prev + f_curr;
    fib_seq.push_back(f_next);
    f_prev = f_curr;
    f_curr = f_next;
  }
  if ( a == f_curr )
    return true;

  vector<bool> fib_chosen(fib_seq.size(), false);
  if ( findUniqSetSum(a, fib_seq.size() - 1, fib_seq, fib_chosen, true) ) {
    cout << "decompose " << a <<": ";
    int i = 0;
    for (; i < fib_seq.size(); ++i) {
      if ( fib_chosen[i] )
	cout << fib_seq[i] << " ";
    }
    cout << endl;
    return true;
  }
  return false;
}

int main() {
  cout << boolalpha;
  cout << uniqFibSum(4) << endl;
  cout << uniqFibSum(9) << endl;
  cout << uniqFibSum(1) << endl;
  cout << uniqFibSum(2) << endl;
  cout << uniqFibSum(11) << endl;
  cout << uniqFibSum(19) << endl;
  cout << uniqFibSum(3212) << endl;
  cout << uniqFibSum(10280) << endl;
  cout << uniqFibSum(19275) << endl;
}
