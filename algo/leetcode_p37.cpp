/**
 * LeetCode Problem 37
 *
 * Solve Sudoku (empty cell indicated by '.')
 */

#include <iostream>
#include <vector>

using namespace std;

typedef unsigned char byte;

class Solution {
public:
  /**
   * Brute force solution
   */
  void solveSudoku_v0(vector<vector<char> > &board)
  {
    int n = board.size(); // n = 9
    for (int i = 1; i < n; i += 3)
      for (int j = 1; j < n; j += 3)
	{
	  vector<bool> is_used_grid(n+1, false); // for this grid
	  int num_empty_cells = 0;
	  for (int hr = -1; hr <= 1; ++hr)
	    for (int vt = -1; vt <= 1; ++vt)
	      {
		char ch = board[i+hr][j+vt];
		if ( ch != '.' )
		  is_used_grid[int(ch - '0')] = true;
		else
		  ++num_empty_cells; // for empty cell
	      }

	  if ( 0 == num_empty_cells ) continue;

	  // Now get the unused cells
	  for (int hr = -1; hr <= 1; ++hr)
	    for (int vt = -1; vt <= 1; ++vt)
	      {
		char ch = board[i+hr][j+vt];
		if ( ch != '.' ) continue;

		// Find row and column constraints
		int row = i + hr, col = j + vt;
		vector<bool> is_used = is_used_grid;
		for (int k = 0; k < n; ++k)
		  {
		    char ch = board[row][k];
		    if ( '.' == ch ) continue;
		    is_used[int(ch - '0')] = true;
		  }
		for (int k = 0; k < n; ++k)
		  {
		    char ch = board[k][col];
		    if ( '.' == ch ) continue;
		    is_used[int(ch - '0')] = true;
		  }

		// Check all the available digits for the cell
		int num_avail = 0;
		for (int dec = 1; dec <= n; ++dec)
		  {
		    if ( is_used[dec] ) continue;
		    ++num_avail;
		    board[row][col] = char(dec + int('0'));
		    solveSudoku(board);
		    // Check if the problem is solved
		    int num_filled = 0;
		    for (int a = 0; a < n; ++a)
		      for (int b = 0; b < n; ++b)
			if ( board[a][b] != '.' ) ++num_filled;
		    if ( n * n == num_filled ) return;
		    // if ( num_filled > 77 )
		    //   cout << "board cells filled: " << num_filled << endl;
		    board[row][col] = '.';
		  }
		// If there is no way to assign a digit to the cell
		if ( 0 == num_avail ) return;
	      }
	}
  }

  /**
   * Method 2: prioritize w.r.t. number of constraints
   */
private:
  bool solve_sudoku(vector< vector<bool> > &row_constr,
		    vector< vector<bool> > &col_constr,
		    vector< vector<bool> > &grid_constr,
		    byte num_constr[9][9],
		    vector< vector<char> > &board,
		    int num_assigned)
  {
    const int n = 9;
    const int grid_size = 3;
#define grid_idx(I, J) ((I) / grid_size) * (n / grid_size) + ((J) / grid_size)

    if ( n * n == num_assigned ) return true;
    
    // Create the list of stuffs
    int max_cnt = -1;
    int row, col;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
	{
	  if ( board[i][j] != '.' ) continue;
	  if ( num_constr[i][j] > max_cnt )
	    {
	      max_cnt = num_constr[i][j];
	      row = i; col = j;
	    }
	}
    
    int grid = grid_idx(row, col);
    for (int dec = 1; dec <= n; ++dec)
      {
	if  ( row_constr[row][dec] ||
	      col_constr[col][dec] ||
	      grid_constr[grid][dec] )
	  continue;

	row_constr[row][dec] = true;
	col_constr[col][dec] = true;
	grid_constr[grid][dec] = true;

	// Initialize the constraints
	byte curr_num_constr[n][n];
	for (int i = 0; i < n; ++i)
	  for (int j = 0; j < n; ++j)
	    curr_num_constr[i][j] = num_constr[i][j];
	
#define update_constraints(a, b) {		\
	  int r = (a), c = (b);			\
	  int g = grid_idx(r, c);		\
	  int cnt = 0;				\
	  for ( int d = 1; d <= n; ++d )	\
	    {					\
	      if ( row_constr[r][d] ||		\
		   col_constr[c][d] ||		\
		   grid_constr[g][d] )		\
		++cnt;				\
	    }					\
	  curr_num_constr[r][c] = cnt;		\
	}					\
	
	// Update the row / col constraints count
	for ( int k = 0; k < n; ++k ) 
	  if ( k != col )
	    update_constraints(row, k);

	for ( int k = 0; k < n; ++k )
	  if ( k != row )
	    update_constraints(k, col);	    

	// Update the constraints on the grid
	int grid_x = 1 + (grid / grid_size) * grid_size;
	int grid_y = 1 + (grid % grid_size) * grid_size;
	for ( int hr = -1; hr <= 1; ++hr )
	  for ( int vt = -1; vt <= 1; ++vt )
	    if ( hr !=0 || vt != 0 )
	      update_constraints( grid_x + hr, grid_y + vt );

	board[row][col] = char(dec + int('0'));
	if ( solve_sudoku(row_constr, col_constr, grid_constr,
			  curr_num_constr, board, num_assigned + 1) )
	  return true;
	
	// Revert the changes made to the constraints and the board
	board[row][col] = '.';
	row_constr[row][dec] = false;
	col_constr[col][dec] = false;
	grid_constr[grid][dec] = false;
      }
    return false;
  }
    
public:
  void solveSudoku(vector<vector<char> > &board)
  {
    const int n = 9;
    const int grid_size = 3;
    
    vector< vector<bool> > row_constr(n, vector<bool>(n+1, false));
    vector< vector<bool> > col_constr(n, vector<bool>(n+1, false));
    vector< vector<bool> > grid_constr(n, vector<bool>(n+1, false));
    byte num_constr[n][n];
    
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
	{
	  char ch = board[i][j];
	  if ( ch == '.' ) continue;
	  int dec = int(ch - '0');
	  row_constr[i][dec] = true;
	  col_constr[j][dec] = true;
	  grid_constr[ grid_idx(i, j) ][dec] = true;
	}

    // Initialize count of constraints
    int num_assigned = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
	{
	  char ch = board[i][j];
	  if ( ch != '.' )
	    ++num_assigned;
	  else
	    {
	      int grid = grid_idx(i, j);
	      int cnt = 0;
	      for (int dec = 1; dec <= n; ++dec)
		if ( row_constr[i][dec] || col_constr[j][dec] || grid_constr[grid][dec] )
		  ++cnt;
	      num_constr[i][j] = cnt;
	    }
	}
    
    solve_sudoku(row_constr, col_constr, grid_constr, num_constr, board, num_assigned);
  }

};

int main()
{
  // const char *puzzle_board[9] = {"53.678912",
  // 				 "67.195..8",
  // 				 "1983.25.7",
  // 				 "85.76.4.3",
  // 				 "426.53.91",
  // 				 "71392485.",
  // 				 "96.5.72.4",
  // 				 "287.19635",
  // 				 "345286179"};

  const char *puzzle_board[9] = {"..9748...",
  				 "7........",
  				 ".2.1.9...",
  				 "..7...24.",
  				 ".64.1.59.",
  				 ".98...3..",
  				 "...8.3.2.",
  				 "........6",
  				 "...2759.."};
  
  Solution sol;
  vector< vector<char> > board(9);
  for (int i = 0; i < 9; ++i)
    {
      board[i].resize(9);      
      for (int j = 0; j < 9; ++j)
	cout << (board[i][j] = puzzle_board[i][j]) << " ";
      cout << endl;
    }
  sol.solveSudoku(board);
  cout << "\n--------------------------\n" << endl;
  for (int i = 0; i < 9; ++i)
    {
      for (int j = 0; j < 9; ++j)
	cout << board[i][j] << " ";
      cout << endl;
    }

}
