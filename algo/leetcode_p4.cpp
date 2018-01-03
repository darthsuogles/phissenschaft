#include <iostream>
#include <vector>

using namespace std;

class Solution {
    double findKthElem(int *A, int m, int *B, int n, int k) {
        if ( m > n ) // A is the shorter array
            return findKthElem(B, n, A, m, k);
        if ( 0 == m )
            return B[k-1];
        if ( 1 == k ) // return the smallest element
            return min(A[0], B[0]);

        // Let sa + sb = k, remove the smaller ones
        int sa = min(m, k / 2), sb = k - sa; 
        if ( A[sa-1] == B[sb-1] ) // the same median
            return A[sa-1];
        // A[0 ... (sa-1)] are amongst the smallest k elements
        else if ( A[sa-1] < B[sb-1] ) 
            return findKthElem(A+sa, m-sa, B, n, k-sa);
        // B[0 ... (sb-1)] are amongst the smallest k elements
        else
            return findKthElem(A, m, B+sb, n-sb, k-sb);        
    }
public:
    double findMedianSortedArrays(vector<int> &Av, vector<int> &Bv) {
        auto m = Av.size(), n = Bv.size();
        int *A = NULL; if (! Av.empty()) A = &Av[0];
        int *B = NULL; if (! Bv.empty()) B = &Bv[0];
        int tot = m + n, k = tot / 2;
        if ( tot % 2 )
            return findKthElem(A, m, B, n, k+1);
        else
            return (findKthElem(A, m, B, n, k) + findKthElem(A, m, B, n, k+1)) / 2;
    }
};

/**
 * Merge two sorted arrays A and B and store the result in Res
 */
vector<int> merge_arrays(vector<int> &A, vector<int> &B) {
    auto m = A.size(), n = B.size();
    vector<int> merged(m + n, 0);
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if ( A[i] < B[j])
            merged[k++] = A[i++];
        else
            merged[k++] = B[j++];
    }
    while ( i < m ) merged[k++] = A[i++];
    while ( j < n ) merged[k++] = B[j++];
    return merged;
}

int main()
{
    Solution sol;
    auto A = vector<int> {3};
    auto B = vector<int> {2,3};
  
    // test(NULL, 0, B, 2);
    // test(B, 2, B, 2);
    // test(A, 1, B, 2);
    // test_case((1,2,3), (3,4));
    auto C = merge_arrays(A, B);
    for (auto v: C) cout << v << " ";
    cout << endl;
}
