#include <iostream>

using namespace std;

template <int p, int i>
class IsPrime {
public:
    enum { prim = ((p % i) && IsPrime<p, i - 1>::prim) };
};

// Use "\C-c r r" to send buffer to CINT/ROOT
int x = 1 + 2 + 3;
cout << x << endl;
cout << x + x + 1 << endl;
   
template <int p>
class IsPrime<p, 1> {
public:
    enum { prim = 1 };
};

template <int i>
class PrimePrint {
public:
    PrimePrint<i - 1> a;
    enum { prim = IsPrime<i, i - 1>::prim };
    void f() {
        a.f();
        if (prim) 
            cout << "prime number: " << i << endl;
    }
};

template<>
class PrimePrint<1> {
public:
    enum { prim = 0 };
    void f() {}
};

#ifndef LAST
#define LAST 128
#endif

int main() {
    PrimePrint<LAST> a;
    a.f();
}
