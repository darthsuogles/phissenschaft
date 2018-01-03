/**
 * LeetCode Problem 126
 *
 * Word ladder, find all paths
 */

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <limits.h>

using namespace std;

struct WordGraphNode
{
  string word;
  int dist;
  int idx;
  WordGraphNode *parent;
  unordered_set<WordGraphNode *> neighbors;

  WordGraphNode(string word): word(word), dist(1), idx(-1), parent(NULL) {};
  WordGraphNode(string word, WordGraphNode* parent, int idx)
    : word(word), parent(parent), idx(idx)
  {
    dist = parent ? (parent->dist + 1) : 1;
  };
};

class Solution {

  void find_path(WordGraphNode *node, WordGraphNode *target, int max_dist,
		 vector<string> &path, vector< vector<string> > &res)
  {
    if ( max_dist < 0 ) return;
    if ( node == target )
      {
	path.push_back(target->word);
	res.push_back(path);
	path.pop_back();
	return;
      }

    path.push_back(node->word);
    for (auto vert = node->neighbors.begin(); vert != node->neighbors.end(); ++vert)
      {
	WordGraphNode *nbr = *vert;
	find_path(nbr, target, max_dist-1, path, res);
      }
    path.pop_back();
  }
  
public:
  vector< vector<string> > findLadders_v0(string start, string end, unordered_set<string> &dict)
  {
    vector< vector<string> > res;
    if ( start.size() != end.size() )
      return res;

    queue<WordGraphNode *> word_queue;
    WordGraphNode *root = new WordGraphNode(start);
    word_queue.push(root);
    unordered_map<string, WordGraphNode *> word2node;
    word2node[start] = root;

    int min_dist = INT_MAX;
    while ( ! word_queue.empty() )
      {
	WordGraphNode *curr = word_queue.front(); word_queue.pop();
	
	int dist = curr->dist;
	if ( dist > min_dist ) break;
	if ( ++dist > min_dist ) continue;

	dict.erase(curr->word); // no need to refer to this one any more
	int len = curr->word.size();
	bool stop_search = false;

	for (int i = 0; i < len && ! stop_search ; ++i)
	  {
	    if ( i == curr->idx ) continue;
	    
	    for (char ch = 'a'; ch <= 'z' && ! stop_search; ++ch)
	      {
		if ( curr->word[i] == ch ) continue;
		string next_word = curr->word;
		next_word[i] = ch;

		if ( next_word == end )
		  {
		    stop_search = true;
		    if ( dist > min_dist ) break;
		    min_dist = dist;
		  }

		if ( next_word == end || dict.count(next_word) > 0 ) // a valid next step
		  {
		    WordGraphNode *node = NULL;
		    if ( word2node.count(next_word) > 0 )
		      {
			node = word2node[next_word];
			if ( dist > node->dist ) continue;
		      }
		    else
		      {
			node = new WordGraphNode(next_word, curr, i);
			word2node[next_word] = node;
			node->parent = curr;
		      }

		    curr->neighbors.insert(node);					
		    word_queue.push( node );
		  }
	      }
	  }
      }

    if ( word2node.count(end) != 0 )
      {
	vector<string> path;
	find_path( root, word2node[end], min_dist, path, res );
      }
    for (auto it = word2node.begin(); it != word2node.end(); delete it->second, ++it);
    return res;
  }

  /**
   * Second approach
   */
  
private:
  void find_paths(string curr, string target, vector<string> &path, int max_dist,
		  vector< vector<string> > &res,		  
		  unordered_map< string, unordered_set<string> > &search_graph)
  {
    if ( 0 == max_dist ) return;
    if ( curr == target )
      {
	path.push_back(target);
	res.push_back(vector<string>(path.rbegin(), path.rend()));
	path.pop_back();
	return;
      }

    path.push_back(curr);
    auto &neighbors = search_graph[curr];
    for ( auto wt = neighbors.begin(); wt != neighbors.end(); ++wt )
      find_paths(*wt, target, path, max_dist - 1, res, search_graph);
    path.pop_back();
  }
  
public:
  vector< vector<string> > findLadders(string start, string end, unordered_set<string> &dict)
  {
    vector< vector<string> > res;
    if ( start.size() != end.size() ) return res;

    unordered_map< string, unordered_set<string> > search_graph;

    int min_dist = INT_MAX;
    queue< pair<string, int> > word_queue;
    word_queue.push( pair<string, int>(start, -1) );
    
    //for (int dist = 1; dist < min_dist && ! word_queue.empty() && ! dict.empty(); ++dist)
    for (int dist = 1; dist < min_dist && ! word_queue.empty(); ++dist)
      {
	int num_elems = word_queue.size();
	for (; num_elems > 0; --num_elems)
	  {	    
	    auto wip = word_queue.front(); word_queue.pop();
	    const string &word = wip.first;
	    const int idx = wip.second;
	    dict.erase(word); // only need the closest visit to the word
	    
	    int len = word.size();
	    bool stop_search = false;
	    string next_word = word;
	    
	    for (int i = 0; i < len && ! stop_search; ++i)
	      {
		if ( idx == i ) continue;
		
		for (char ch = 'a'; ch <= 'z' && ! stop_search ; ++ch)

		  {
		    if ( word[i] == ch ) continue;
		    next_word[i] = ch;

		    bool target_matched = (next_word == end);
		    if ( target_matched )
		      {
			min_dist = dist + 1;
			// From any word there is only one mutation
			// to match any other word, thus once the
			// end word is found, break the search of
			// the current word.
			stop_search = true; 
		      }

		    if ( target_matched || dict.count(next_word) > 0 )
		      {
			if ( search_graph.count(next_word) == 0 )
			  word_queue.push( pair<string, int>(next_word, i) );
			search_graph[next_word].insert( word );
		      }
		  }
		next_word[i] = word[i];
	      }
	  }
      }

    if ( min_dist != INT_MAX )
      {
	vector<string> path;
	find_paths(end, start, path, min_dist, res, search_graph);
      }
    return res;
  }
    
};

int main()
{
  Solution sol;

#define test_case(w1, w2, A) {				\
    int len = sizeof((A)) / sizeof((A)[0]);		\
    unordered_set<string> dict((A), (A)+len);		\
    auto res = sol.findLadders((w1), (w2), dict);	\
    for (auto it = res.begin(); it != res.end(); ++it)	\
      {							\
	cout << "[";					\
	auto wt = it->begin();				\
	for (; wt+1 != it->end(); ++wt)			\
	  cout << "\"" << *wt << "\",";			\
	cout << "\"" << *wt << "\"" << "]" << endl;	\
      }							\
    cout << "------------------------------------\n\n"; \
  }							\
  
  const char *A[] = {"hot","dot","dog","lot","log"};    
  test_case("hit", "cog", A);

  const char *B[] = {"a", "b", "c"};
  test_case("a", "c", B);

  const char *C[] = {"ted","tex","red","tax","tad","den","rex","pee"};
  test_case("red", "tax", C);

  const char *D[] = {"dose","ends","dine","jars","prow","soap","guns","hops","cray","hove","ella","hour","lens","jive","wiry","earl","mara","part","flue","putt","rory","bull","york","ruts","lily","vamp","bask","peer","boat","dens","lyre","jets","wide","rile","boos","down","path","onyx","mows","toke","soto","dork","nape","mans","loin","jots","male","sits","minn","sale","pets","hugo","woke","suds","rugs","vole","warp","mite","pews","lips","pals","nigh","sulk","vice","clod","iowa","gibe","shad","carl","huns","coot","sera","mils","rose","orly","ford","void","time","eloy","risk","veep","reps","dolt","hens","tray","melt","rung","rich","saga","lust","yews","rode","many","cods","rape","last","tile","nosy","take","nope","toni","bank","jock","jody","diss","nips","bake","lima","wore","kins","cult","hart","wuss","tale","sing","lake","bogy","wigs","kari","magi","bass","pent","tost","fops","bags","duns","will","tart","drug","gale","mold","disk","spay","hows","naps","puss","gina","kara","zorn","boll","cams","boas","rave","sets","lego","hays","judy","chap","live","bahs","ohio","nibs","cuts","pups","data","kate","rump","hews","mary","stow","fang","bolt","rues","mesh","mice","rise","rant","dune","jell","laws","jove","bode","sung","nils","vila","mode","hued","cell","fies","swat","wags","nate","wist","honk","goth","told","oise","wail","tels","sore","hunk","mate","luke","tore","bond","bast","vows","ripe","fond","benz","firs","zeds","wary","baas","wins","pair","tags","cost","woes","buns","lend","bops","code","eddy","siva","oops","toed","bale","hutu","jolt","rife","darn","tape","bold","cope","cake","wisp","vats","wave","hems","bill","cord","pert","type","kroc","ucla","albs","yoko","silt","pock","drub","puny","fads","mull","pray","mole","talc","east","slay","jamb","mill","dung","jack","lynx","nome","leos","lade","sana","tike","cali","toge","pled","mile","mass","leon","sloe","lube","kans","cory","burs","race","toss","mild","tops","maze","city","sadr","bays","poet","volt","laze","gold","zuni","shea","gags","fist","ping","pope","cora","yaks","cosy","foci","plan","colo","hume","yowl","craw","pied","toga","lobs","love","lode","duds","bled","juts","gabs","fink","rock","pant","wipe","pele","suez","nina","ring","okra","warm","lyle","gape","bead","lead","jane","oink","ware","zibo","inns","mope","hang","made","fobs","gamy","fort","peak","gill","dino","dina","tier"};
  test_case("nape", "mild", D);

  const char *E[] = {"hot", "dog"};
  test_case("hot", "dog", E);
}
