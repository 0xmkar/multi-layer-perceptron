# I wanna make the value class
# vars - data, grad, label, operation, backward func, prev
# Ops - mult, div, add, activation_func

class Value:

    def __init__(self, data: float, label: str = "") -> None:
        self.data = data
        self.grad = 0
        self.label = label
        self._backwards = None
        self.op = None
        self.prev = None

    def __add__(self, other):
        if isinstance(other, (int, float)) : other = Value(other)
        out = Value(self.data + other.data)
        out.prev = (self, other)
        out.label = "+"
        out.op = "+"

        def _backwards():
            self.grad += out.grad
            other.grad += out.grad

        out._backwards = _backwards
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)) : other = Value(other)
        out = Value(self.data * other.data)
        out.prev = (self, other)
        out.label = "*"
        out.op = "*"

        def _backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backwards = _backwards
        return out

    def __rmul__(self, other):
        return self * other

    def backwards(self):
        topo: list[Value] = []
        visited: set[int] = set()

        def build(v: Value) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            if v.prev:
                for child in v.prev:
                    build(child)
            topo.append(v)

        build(self)
        for v in topo:
            v.grad = 0
        self.grad = 1.0
        for v in reversed(topo):
            if v._backwards:
                v._backwards()

    def __repr__(self):
        return f"Value(data={self.data}, label='{self.label}')"
