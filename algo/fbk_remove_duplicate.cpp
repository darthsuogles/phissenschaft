/**
 * Facebook interview question
 */

#include <iostream>

using namespace std;

int remove_duplicate(int A[], int len)
{
  if ( len < 2 )
    return len;

  int j = 0; // first element is not a duplicate
  for (int i = 0; i + 1 < len; ++i)
    {
      if ( A[i] != A[i+1] )
	A[j++] = A[i];
    }
  A[j++] = A[len-1]; // the last is not a duplicate

  return j;
}

int main()
{
  int A[] = {1,1,2,3,4,4,4,5,6,7};
  int len = sizeof(A) / sizeof(int);

  int nlen = remove_duplicate(A, len);
  cout << nlen << endl;

  for (int i = 0; i < nlen; ++i)
    cout << A[i] << " ";
  cout << endl;
}
