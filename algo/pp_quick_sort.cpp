/**
 * Programming Pearls
 */

#include <iostream>
//#include <vector>
//#include <algorithm>

using namespace std;

void quick_sort(int A[], int p, int q)
{
  if ( q <= p ) return;
  
  int pivot = A[q];
  int i = p, j = q - 1;
  while ( true ) {
      while ( A[i] < pivot ) ++i;
      while ( A[j] > pivot && j > 0 ) --j;
      if ( i >= j ) break;
      int tmp = A[i]; A[i++] = A[j]; A[j--] = tmp;
  }
  int tmp = A[i]; A[i] = A[q]; A[q] = tmp;
  quick_sort(A, p, j);
  quick_sort(A, i+1, q);
}

int main()
{
  const int n = 1237;
  int A[n];
  for (int i = 0; i < n; ++i)
    A[i] = (i*17 + 31 + i*i) % 129;

  quick_sort(A, 0, n-1);
  bool is_sorted = true;
  for (int i = 0; i+1 < n; ++i)
    {
      //cout << A[i] << " ";
      if ( A[i] > A[i+1] )
	{
	  is_sorted = false;
	  break;
	}
    }
  if ( ! is_sorted )
    cout << "Failed" << endl;
  else
    cout << "Passed" << endl;
}
