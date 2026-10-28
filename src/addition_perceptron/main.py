# we need to give different activation functions to the perceptron's output neuron
# so that it can output different values 

# MLP of -> 2 inputs per neuron in the first layer -> each being number to be added
# will experiment with different number of hidden layers and neurons per layer

import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp import Neuron
from data_type import Value
from src.visualise import show_graph

# Final neuron for addition
final_neuron = Neuron(2)
best_loss = float('inf')
best_params = None
BATCH_SIZE = 32
LR = 0.001

# training loop
for i in range(1000):
    batch_losses: list[Value] = []
    for _ in range(BATCH_SIZE):
        inputs = [Value(random.uniform(0, 10)), Value(random.uniform(0, 10))]
        final_output = final_neuron.call_without_activation(inputs)
        target = Value(inputs[0].data + inputs[1].data)
        batch_losses.append((final_output - target) ** 2)

    loss = batch_losses[0]
    for sample_loss in batch_losses[1:]:
        loss = loss + sample_loss
    loss = loss * (1.0 / BATCH_SIZE)

    loss.backwards()
    if loss.data < best_loss:
        best_loss = loss.data
        best_params = [p.data for p in final_neuron.parameters()]
    for p in final_neuron.parameters():
        p.data -= LR * p.grad
    if i % 100 == 0:
        print(f"Step {i} - Avg batch loss: {loss.data:.6f}")
    if i == 999:
        print(f"Best loss: {best_loss:.6f}")
        print(f"Best parameters: {best_params}")

# load best parameters
for p, best_p in zip(final_neuron.parameters(), best_params):
    p.data = best_p

# One-sample graph (batch-averaged loss ties 32 graphs together and blows up graphviz/PIL)
viz_inputs = [Value(3.0), Value(4.0)]
viz_output = final_neuron.call_without_activation(viz_inputs)
viz_loss = (viz_output - Value(viz_inputs[0].data + viz_inputs[1].data)) ** 2
viz_loss.backwards()
show_graph(viz_loss)

# test the model on multiple inputs and print the output
inputs = [[Value(5), Value(5)], [Value(1), Value(2)], [Value(3), Value(4)], [Value(6), Value(7)], [Value(8), Value(9)]]
for input in inputs:
    final_output = final_neuron.call_without_activation(input)
    print(f"Final test output: {final_output.data}")