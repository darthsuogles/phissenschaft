/**
 * Connected components in a 2D image
 */

#include <iostream>
#include <vector>

using namespace std;

void paint_canvas(int i, int j, int label,
		  vector< vector<char> > &image,
		  vector< vector<char> > &canvas)
{
  int m = image.size();
  int n = image[0].size();  
  if ( i < 0 || i >= m || j < 0 || j >= n ) return;
  if ( image[i][j] != '1' || canvas[i][j] != '0' ) return;
  
  canvas[i][j] = char(label + int('0'));
  if ( i > 0 )
    paint_canvas(i-1, j, label, image, canvas);
  if ( i < m-1 )
    paint_canvas(i+1, j, label, image, canvas);
  if ( j > 0 )
    paint_canvas(i, j-1, label, image, canvas);
  if ( j < n-1 )
    paint_canvas(i, j+1, label, image, canvas);
}

void image_conn_comp(vector< vector<char> > &image)
{
  if ( image.empty() ) return;

  int label = 1;   
  int m = image.size();
  int n = image[0].size();
  vector< vector<char> > canvas(m, vector<char>(n, '0'));  
  
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      if ( '1' == image[i][j] && '0' == canvas[i][j] )
	paint_canvas(i, j, label++, image, canvas);
  
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      image[i][j] = canvas[i][j];
}

int main()
{
  const int m = 4;
  const int n = 10;
  const char *IMG[m] = {
    "0000011000",
    "1100110000",
    "1110011000",
    "0001100000"
  };
  
  vector< vector<char> > image(m, vector<char>(n));
  cout << image.size() << endl;
  cout << image[0].size() << endl;
  
  for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
	cout << (image[i][j] = IMG[i][j]) << " ";
      cout << endl;
    }

  image_conn_comp(image);
  cout << "\n---------------------------\n" << endl;
  for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
	cout << image[i][j] << " ";
      cout << endl;
    }
}
