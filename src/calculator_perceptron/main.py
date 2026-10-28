# we need to give different activation functions to the perceptron's output neuron
# so that it can output different values 

# MLP of -> 2 inputs per neuron in the first layer -> each being number to be added
# will experiment with different number of hidden layers and neurons per layer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import multi_layer_perceptron, Neuron
from data_type import Value
from visualise import show_mlp

# Percecptron
a = multi_layer_perceptron(2, [3, 4, 5, 6, 8], 5)
inputs = [Value(1), Value(2)]
b = a(inputs)
print(b)

# Backward pass - to get the gradient
for i in range(len(b)):
    b[i].backwards()

show_mlp(a, inputs)

# Final Neuron to get output
final_neuron = Neuron(5)
final_output = final_neuron.call_without_activation(b)
print("Final output: ", final_output)