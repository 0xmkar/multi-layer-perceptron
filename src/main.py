from data_type import Value
from visualise import show_graph

x = Value(1)
y = Value(11)

a = x + y
b = a * y
c = b * b
d = c + b
e = d + c

e.backward()

print(e.data, e.grad)
print(b.data, b.grad)
print(x.grad, y.grad)

# Visualise
show_graph(e)