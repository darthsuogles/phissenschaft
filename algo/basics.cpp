#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <random>

using namespace std;

int main()
{
  // auto, type-safe threading, ...
  // http://www.cprogramming.com/c++11/c++11-ranged-for-loop.html
  vector<int> vec;
  vec.push_back(10);
  vec.push_back(20);

  for (int i: vec)
    {
      cout << i << "\t";
    }
  cout << endl;
  
  map<string, string> address_book;
  address_book["alice"] = "91213e";
  address_book["cheshire"] = "sss231ds1";
  for (auto address_entry: address_book) // no worries about the iterator type
    {
      cout << address_entry.first << "\t< " << address_entry.second << " > " << endl;
    }
  cout << endl;

  typedef mt19937 prng_t; // the Mersenne Twister with a popular choice of parameters
  uint32_t seed_val;
  prng_t rng;
  rng.seed(seed_val);

  uniform_int_distribution<uint32_t> uint_dist; // default, range = [0, MAX]
  uniform_int_distribution<uint32_t> uint_dist10(0, 10); // range = [0, 10]
  normal_distribution<double> normal_dist(0, 1);

  int cnt = 0;
  while ( cnt++ < 10 )
    {
      cout << uint_dist(rng) << " "
	   << uint_dist10(rng) << " "
	   << normal_dist(rng) << endl;
    }

  return 0;
}
