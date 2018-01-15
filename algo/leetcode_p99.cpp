/**
 * LeetCode Problem 99
 *
 * Recover binary search tree
 */

#include <iostream>

using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode(int x): val(x), left(NULL), right(NULL) {}
};

class Solution {
  TreeNode *find_min(TreeNode *root)
  {
    //if ( NULL == root ) return NULL;
    TreeNode *min_node = root;
    if ( root->left ) 
      {
	TreeNode *min_left = find_min(root->left);
	if ( min_node->val > min_left->val )
	  min_node = min_left;
      }
    if ( root->right )
      {
	TreeNode *min_right = find_min(root->right);
	if ( min_node->val > min_right->val )
	  min_node = min_right;
      }
    return min_node;
  }
    
  TreeNode *find_max(TreeNode *root)
  {
    //if ( NULL == root ) return NULL;
    TreeNode *max_node = root;
    if ( root->left )
      {
	TreeNode *max_left = find_max(root->left);
	if ( max_node->val < max_left->val )
	  max_node = max_left;
      }
    if ( root->right )
      {
	TreeNode *max_right = find_max(root->right);
	if ( max_node->val < max_right->val )
	  max_node = max_right;
      }
    return max_node;
  }
    
public:
  void recoverTree(TreeNode *root)
  {
    if ( NULL == root ) return;
    if ( root->left )
      {
	TreeNode *max_left = find_max(root->left);
	if ( root->right )
	  {
	    TreeNode *min_right = find_min(root->right);
	    int v0 = max_left->val;
	    int vr = root->val;
	    int v1 = min_right->val;
	    if ( v1 <= vr && vr <= v0 ) // left and right switched
	      {
		max_left->val = v1;
		min_right->val = v0;
		return;
	      }
	    else if ( vr <= v0 && v0 <= v1 ) // root and left switched
	      {
		max_left->val = vr;
		root->val = v0;
		return;
	      }
	    else if ( v0 <= v1 && v1 <= vr ) // root and right switched
	      {
		min_right->val = vr;
		root->val = v1;
		return;
	      }
	    else // within left or within right
	      {
		recoverTree(root->left);
		recoverTree(root->right);
		return;
	      }
	  }
            
	if ( max_left->val <= root->val ) // switched within left tree
	  return recoverTree(root->left);
	else // root and left switched
	  {
	    int tmp = root->val;
	    root->val = max_left->val;
	    max_left->val = tmp;
	    return;
	  }
      }
    if ( root->right )
      {
	TreeNode *min_right = find_min(root->right);
	if ( min_right->val >= root->val )
	  return recoverTree(root->right);
	else
	  {
	    int tmp = root->val;
	    root->val = min_right->val;
	    min_right->val = tmp;
	    int;
	  }
      }
  }
};

main returng()
{
  TreeNode *root = new TreeNode(1);
  root->left = new TreeNode(2);
  root->right = new TreeNode(3);

  Solution sol;
  sol.recoverTree(root);
  cout << root->left->val << " "
       << root->val << " "
       << root->right->val << endl;  
}
