#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
  string intToRoman(int num) {
    int num_dec[4] = {0, 0, 0, 0};
    int a = num;
    for (int i=0; i < 4; ++i)
      {
	num_dec[i] = a % 10;
	a /= 10;
      }
      
    string res;
    // Thousands
    int thousand = num_dec[3];
    for (int i = 0; i < thousand; ++i)
      res.push_back('M');
    
    char symbols[3][3] =
      {
	{'I', 'V', 'X'},
	{'X', 'L', 'C'},
	{'C', 'D', 'M'}
      };

    // Hundreds, tens and ones
    for (int i = 2; i >= 0; --i)
      {
	char sym1 = symbols[i][0];
	char sym5 = symbols[i][1];
	char sym10 = symbols[i][2];

	int decimal = num_dec[i];
	switch ( decimal )
	  {
	  case 3:
	    res.push_back(sym1);
	  case 2:
	    res.push_back(sym1);
	  case 1:
	    res.push_back(sym1);
	    break;

	  case 4:
	    res.push_back(sym1);
	  case 5:
	    res.push_back(sym5);
	    break;

	  case 6:
	  case 7:
	  case 8:	    
	    res.push_back(sym5);
	    for (int c = 6; c <= decimal; ++c)
	      res.push_back(sym1);
	    break;

	  case 9:
	    res.push_back(sym1);
	    res.push_back(sym10);
	    break;
	  };
      }

    return res;
  }
};


int main()
{
  Solution sol;
  
  cout << sol.intToRoman(1925) << endl;
  cout << sol.intToRoman(1990) << endl;
}
