# Try symbolic differentiation
from abc import ABC, abstractmethod, abstractproperty

class DiffOP(ABC):
    @property
    def _need_parenthesis(self): return False
    @abstractmethod
    def diff_expr(self): pass

class Const(DiffOP):
    def __init__(self, c): self.c = c
    def __str__(self): return "const({})".format(self.c)
    def diff_expr(self): return "0"

class Var(DiffOP):
    def __init__(self, name): self.name = name
    def __str__(self): return self.name
    def diff_expr(self): return self.name

class Add(DiffOP):
    def __init__(self, x, y): self.x = x; self.y = y
    @property
    def _need_parenthesis(self): return True
    def __str__(self): return "{} + {}".format(self.x, self.y)
    def diff_expr(self): return "{} + {}".format(self.x.diff_expr(), self.y.diff_expr())

class Exp(DiffOP):
    def __init__(self, x): self.x = x
    def __str__(self): return "exp({})".format(self.x)
    def diff_expr(self):
        in_expr = self.x.diff_expr()
        if self.x._need_parenthesis():
            in_expr = "({})".format(in_expr)
        return "{} * {}".format(self, in_expr)


y = Exp(Exp(Add(Exp(Add(Exp(Var("x")), Var("y"))), Const(3))))
y.diff_expr()
