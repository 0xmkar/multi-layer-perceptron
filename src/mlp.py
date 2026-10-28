from data_type import Value
import random
from typing import Iterator

class Neuron:
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def activation_function(self, x: Value) -> Value:
        return x.tanh()

    def __call__(self, x: list[Value]) -> Value:
        z = self.b
        for wi, xi in zip(self.w, x):
            z = z + wi * xi
        return self.activation_function(z)
    
    def call_without_activation(self, x: list[Value]) -> Value:
        z = self.b
        for wi, xi in zip(self.w, x):
            z = z + wi * xi
        return z
    
    def __repr__(self) -> str:
        return f"Neuron(weights={self.w}, bias={self.b})"
    
    def parameters(self) -> list[Value]:
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in: int, n_out: int) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
    
    def __call__(self, x: list[Value]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]
    
    def parameters(self) -> list[Value]:
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer(neurons={len(self.neurons)}, input-per-neuron=({self.n_in})"

class multi_layer_perceptron:
    def __init__(self, n_in: int, n_hidden: list[int], n_out: int) -> None:
        sz = [n_in] + n_hidden + [n_out]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz)-1)]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = layer.__call__(x)
        return x

    def __repr__(self) -> str:
        return f"MLP(layers={len(self.layers)})"

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def loss_function(self, y_pred: list[Value], y_true: list[Value]) -> Value:
        return sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_pred)))
