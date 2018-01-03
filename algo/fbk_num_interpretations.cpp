/**
 * Facebook interview question
 */

#include <iostream>
#include <string>
#include <vector>

using namespace std;

// The recursive solution
int num_interpretations(string codes)
{
  if ( codes.empty() ) return 0;
  int len = codes.size();
  
  int cnt = 0;
  
  // First digit  
  char ch = codes[0];
  if ( ch >= '1' && ch <= '9' )
    {
      if ( 1 == len ) return 1;

      // Right shift by one
      string next_codes = codes.substr(1, len-1);
      cnt = num_interpretations(next_codes);
    }
  if ( 1 == len )
    return cnt;
      
  // First two digits
  int val = 0;
  for (int i = 0; i < 2; ++i)
    {
      char ch = codes[i];
      if ( ch <= '0' || ch > '9' )
	return cnt;
      val = val * 10 + int(ch - '0');
    }

  if ( val > 0 && val <= 26 )
    {
      if ( 2 == len )
	return cnt + 1;
      
      // Right shift by two
      string next_codes = codes.substr(2, len-2);
      cnt += num_interpretations(next_codes);
    }

  return cnt;
}

int num_interpretations_nonrec(string codes)
{
  if ( codes.empty() ) return 0;

  int len = codes.size();  
  //vector<int> tbl(len, 0);
  int tbl[2] = {0};
  for (int i = len-1; i >= 0; --i)
    {
      int cnt = 0;
      int val = 0;

      char ch = codes[i];	    
      if ( '1' <= ch && ch <= '9' )
	{
	  val = int(ch - '0');
	  if ( i + 1 >= len )
	    cnt = 1;
	  else
	    {
	      //cnt = tbl[i+1];
	      cnt = tbl[0];
	      ch = codes[i+1];
	      if ( '1' <= ch && ch <= '9' )
		{
		  val = val * 10 + int(ch - '0');
		  if ( 1 <= val && val <= 26 )
		    {
		      if ( i + 2 < len )
			cnt += tbl[1];
			//cnt += tbl[i+2];
		      else
			cnt += 1;
		    }
		}
	    }
	}
      
      //tbl[i] = cnt;
      tbl[1] = tbl[0];
      tbl[0] = cnt;
    }

  return tbl[0];
}

int main()
{
  cout << num_interpretations_nonrec("112") << endl;
  cout << num_interpretations_nonrec("306") << endl;
}
