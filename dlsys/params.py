"""
Metaprogramming
"""
from collections import namedtuple

class ParamsRegistry(type):
    def __init__(cls, name, bases, namespace):
        super(ParamsRegistry, cls).__init__(name, bases, namespace)
        if not hasattr(cls, 'registry'):
            cls.registry = set()
        cls.registry.add(cls)
        cls.registry -= set(bases)

    def __iter__(cls):
        return iter(cls.registry)

    def __str__(cls):
        if cls in cls.registry:
            return ':param: {}'.format(cls.__name__)
        return cls.__name__ + ": " + ", ".join([sc.__name__ for sc in cls])


class Params(object, metaclass=ParamsRegistry):
    pass


kls = type('HasOptimizer', (Params,), dict(
    to_str=lambda self: 'str repr {}'.format(type(self))
))
print(kls)
print(type(kls))
obj = kls()
print(obj.to_str())
