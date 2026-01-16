from wrapper import Tensor, LinearLayer

print("--- MicroTensor High-Level Test ---")

A = Tensor(2, 3, data=[1, 2, 3, 4, 5, 6])
B = Tensor(3, 2, data=[1, 0, 0, 1, 1, 1])

print("Tensor A: ", A)
print("Tensor B: ", B)

C = A.matmul(B)
print("A * B = ", C)

D = Tensor(1, 4, data=[-2, 4, -7, 3])
print("Before ReLU: ", D)
D.relu()
print("After ReLU: ", D)

print("--- AI Inference Test ---")

input_data = Tensor(1, 3, data=[0.5, -0.2, 0.9])
print("Input:", input_data)

layer = LinearLayer(3, 2)
print("Weights:", layer.weights)

output = layer.forward(input_data)

output.add(layer.bias)

output.relu()

print("Final Prediction:", output)