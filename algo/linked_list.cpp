#include <cstdio>
#include <cstdlib>

template <class T>
struct Node
{
  T id;
};

template <class T>
class LinkedList
{
protected:
  Node<T>* head;
  Node<T>* tail;
  unsigned int count;

public:
  LinkedList() 
  {
    printf("LinkedList\n");
    head = NULL;
    tail = NULL;
    count = 0;
  }
};

template <class T>
class EnhancedLinkedList : public LinkedList<T>
{
public:
  EnhancedLinkedList(): LinkedList<T>() 
  {
    printf("EnhancedLinkedList\n");
  }

  Node<T>* getHead()
  {
    // Without using the keyword "this", the following line generates an error 
    // Ref: http://gcc.gnu.org/onlinedocs/gcc/Name-lookup.html
    return this->head;
  }
};


int main()
{
  EnhancedLinkedList<int> list;
  list.getHead();

  return 0;
}
