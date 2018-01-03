#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <random>

using namespace std;

template <typename T>
void get_all_permutations_aux(T buf[], const size_t pos, const size_t len)
{  
  // Base case
  if ( len == pos+1 )
    {
      for (int i=0; i<len; ++i)
	cout << buf[i] << " ";
      cout << endl;
      return;
    }

  for (int i=pos; i<len; ++i)
    {
      T tmp = buf[pos]; 
      buf[pos] = buf[i];
      buf[i] = tmp;
      get_all_permutations_aux(buf, pos+1, len);
      tmp = buf[pos];
      buf[pos] = buf[i];
      buf[i] = tmp;
    }
}

template <typename T>
void get_all_permutations(const T arr[], const size_t len)
{
  T *buf = new T[len];
  for (int i=0; i<len; buf[i] = arr[i], ++i);
  get_all_permutations_aux(buf, 0, len);
  delete [] buf;
}

template <typename T>
void random_shuffle(T buf[], const size_t len)
{
  int idx = len-1;
  random_device rdev;
  mt19937 gen( rdev() );
  // Ref: http://en.cppreference.com/w/cpp/numeric/random

  for (; idx > 0; --idx)
    {
      uniform_int_distribution<> dist(0, idx);
      int j = dist(gen);
      T tmp = buf[idx];
      buf[idx] = buf[j];
      buf[j] = tmp;
    }
}

int main(int argc, char **argv)
{
  //cout << "usage: " << argv[0] << " [options] " << endl;  

  // const size_t len = 4;
  // char arr[] = {'a', 'b', 'c', 'd'};
  // get_all_permutations(arr, len);

  int v[] = {1,2,3};
  //get_all_permutations(v, 3);

  random_shuffle(v, 3);
  for (int i=0; i<3; ++i)
    cout << v[i] << " ";
  cout << endl;

  return 0;
}

