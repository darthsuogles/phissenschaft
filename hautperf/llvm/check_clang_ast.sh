#!/bin/bash

# Clang by default is a frontend for many tools; -Xclang is used to pass
# options directly to the C++ frontend.
# Ref: http://clang.llvm.org/docs/IntroductionToTheClangAST.html
clang -Xclang -ast-dump -fsyntax-only \
      -xc - << _C_SRC_EOF_
int f(int x) {
  int y = (x / 42);
  return y;
}
_C_SRC_EOF_
