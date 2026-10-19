# we need to give different activation functions to the perceptron's output neuron
# so that it can output different values 

# MLP of -> 2 inputs per neuron in the first layer -> each being number to be added
# will experiment with different number of hidden layers and neurons per layer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import multi_layer_perceptron
from data_type import Value
from visualise import show_mlp

a = multi_layer_perceptron(2, [3, 4, 5, 3, 10, 5], 2)
b = a([Value(1), Value(2)])
b[0].backwards()
show_mlp(a)