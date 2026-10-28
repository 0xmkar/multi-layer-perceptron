import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import multi_layer_perceptron, Neuron
from data_type import Value
from visualise import show_mlp

# MLP of -> 3 inputs per neuron in the first layer -> each being number to be added, 
# one being the operator
# will experiment with different number of hidden layers and neurons per layer

# Operators:
# 0: +
# 1: -
# 2: *
# 3: /

perceptron = multi_layer_perceptron(3, [3, 4, 5, 6, 8], 1)


# training loop