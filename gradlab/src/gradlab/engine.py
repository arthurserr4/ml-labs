import math


class Value:
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0                # nothing has been sent back yet
        self._prev = set(_children)    # the Values I was made from
        self._backward = lambda: None  # an input has nothing to pass back

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other))

        def _backward():
            # the local derivative of a + b is 1 for both sides
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    
    def tanh(self):
        out = Value(math.tanh(self.data),(self,))
        t = out.data

        def _backward():
            self.grad += (1-t**2) * out.grad

            
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
