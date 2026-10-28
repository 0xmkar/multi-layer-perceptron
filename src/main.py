import random
from data_type import Value
from visualise import show_graph
from mlp import Neuron

# a = multi_layer_perceptron(3, [4, 5], 1)
# show_mlp(a)
# b = a([Value(1), Value(2), Value(3)])
# b[0].backwards()
# # for li, layer in enumerate(a.layers):
# #     print(f"Layer {li} ({layer.n_in} → {layer.n_out})")
# #     for ni, neuron in enumerate(layer.neurons):
# #         print(f"  [{ni}] w={[round(w.data, 4) for w in neuron.w]}  b={neuron.b.data:.4f}")

# a = multi_layer_perceptron(3, [4, 5], 1)
# inputs = [Value(1), Value(2), Value(3)]
# target = [Value(-2)]
# lr = 0.05

# for step in range(50):
#     b = a(inputs)
#     loss = a.loss_function(b, target)
#     loss.backwards()

#     for p in a.parameters():
#         p.data -= lr * p.grad

#     if step % 10 == 0:
#         print(f"step {step} - loss {loss.data:.6f}, output {b[0].data:.6f}")


# Learning how to give more data for training
# saving model
# loading model

