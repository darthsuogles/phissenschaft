#include <iostream>
#include <vector>

using namespace std;

/**
 * Finding the maximum common substring of two strings A and B
 * by dynamic programming. 
 */
int lgcomm_substr(char A[], int m, char B[], int n)
{
  if ( m < n )
    return lgcomm_substr(B, n, A, m);

  vector<int> tbl(n);
  for (int k=0; k < n+1; tbl[k++] = 0);  
  int max_match = 0;
  int max_i = -1, max_j = -1;

  int prev = 0;
  for (int i=0; i < m; ++i)
    {
      prev = 0;
      for (int j=0; j < n; ++j)
	{
	  int val;
	  if ( A[i] == B[j] )
	    val = prev + 1;
	  else
	    val = 0;
	  if ( val > max_match )
	    {
	      max_match = val;
	      max_i = i;
	      max_j = j;
	      //printf("i: %d, j: %d\n", i+1, j+1);	      
	    }
	  prev = tbl[j];
	  tbl[j] = val;
	}
    }

  // Retract
  int k = 0;
  for (; k <= max_i && k <= max_j && A[max_i - k] == B[max_j - k]; ++k);
  while ( k > max_i || k > max_j ) --k; 

  // Print string A
  int ld_spaces = max_i - max_j;
  if ( ld_spaces < 0 )
    for (int i=0; i < -ld_spaces; ++i) cout << " ";
  for (int i=0; i < m; ++i) cout << A[i];
  cout << endl;

  // Print the bars indicating matches
  int mid_spaces = max(max_i, max_j) - max_match + 1;
  for (int s=0; s < mid_spaces; ++s) cout << " ";
  for (int s=0; s < max_match; ++s) cout << "|";
  cout << endl;

  // Print string B
  if ( ld_spaces > 0 )
    for (int j=0; j < ld_spaces; ++j) cout << " ";
  for (int j=0; j < n; ++j) cout << B[j];
  cout << endl;
  
  return max_match;
}

int main()
{
  char A[] = {'a', 'b', 'a', 'b', 'b', 'b'};
  char B[] = {'b', 'a', 'b', 'b', 'c'};
  int m = sizeof A;
  int n = sizeof B;
  cout << lgcomm_substr(A, m, B, n) << endl;

#define test_case(A, B)					\
  {							\
    int m = strlen((A)); int n = strlen((B));		\
    cout << lgcomm_substr((A), m, (B), n) << endl;	\
  }							\
  
  test_case("obama", "obama");
  test_case("obama", "osaama");
  test_case("abcsds", "csddsa");

  return 0;
}
