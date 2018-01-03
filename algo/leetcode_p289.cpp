/**
 * LeetCode Problem 289
 *
 * Game of Life, with inplace update
 * https://leetcode.com/problems/game-of-life/
 */

class Solution {
public:
    void gameOfLife(vector<vector<int>>& board) {
        int m = board.size();
        if ( 0 == m ) return;
        int n = board[0].size();
        if ( 0 == n ) return;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int cnt = 0;
                for (int ii = max(0, i-1); ii <= min(m-1, i+1); ++ii) {
                    for (int jj = max(0, j-1); jj <= min(n-1, j+1); ++jj) {
                        if ( ii == i && jj == j ) continue;
                        cnt += board[ii][jj] & 1;
                    }
                }
                int val = board[i][j];
                switch ( cnt ) {
                    case 2: 
                        if ( 0 == val) break;
                    case 3:
                        board[i][j] = 2 + val;
                        break;
                    default:
                        break;
                }
            }
        }
        
        for (int i = 0; i < m; ++i) 
            for (int j = 0; j < n; ++j)
                board[i][j] = (board[i][j] & 2) >> 1;
            
    }
};
