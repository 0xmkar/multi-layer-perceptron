from data_type import Value
from visualise import show_graph
from mlp import Neuron, Layer, multi_layer_perceptron

a = multi_layer_perceptron(3, [4, 5], 1)
b = a([Value(1), Value(2), Value(3)])
b[0].backwards()
for li, layer in enumerate(a.layers):
    print(f"Layer {li} ({layer.n_in} → {layer.n_out})")
    for ni, neuron in enumerate(layer.neurons):
        print(f"  [{ni}] w={[round(w.data, 4) for w in neuron.w]}  b={neuron.b.data:.4f}")

# Example output:
# Layer 0 (3 → 4)
#   [0] w=[-0.0692, 0.5331, -0.6226]  b=0.3199
#   [1] w=[0.4223, -0.6063, -0.6167]  b=-0.9355
#   [2] w=[-0.3673, 0.695, -0.0142]  b=0.8908
#   [3] w=[-0.4708, 0.6079, 0.452]  b=0.2087
# Layer 1 (4 → 5)
#   [0] w=[-0.0896, -0.2674, -0.549, -0.8291]  b=-0.5135
#   [1] w=[0.0188, -0.6568, 0.1958, -0.9419]  b=-0.8909
#   [2] w=[0.4461, -0.5585, 0.6267, -0.6367]  b=0.0272
#   [3] w=[-0.4873, -0.5931, -0.4259, 0.696]  b=-0.1512
#   [4] w=[-0.6156, 0.0417, 0.29, 0.8675]  b=0.8271
# Layer 2 (5 → 1)
#   [0] w=[-0.4905, 0.6657, -0.7443, 0.1084, -0.8063]  b=0.1556



# 1. Add a loss function
# 2. Tweak weights and biases so we can alter the ouput
# 3. Visualise MLP with weights and biases
