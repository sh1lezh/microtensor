# import ctypes

# lib = ctypes.CDLL('./libtensor.so')

# lib.create_tensor.restype = ctypes.c_void_p
# lib.create_tensor.argtypes = [ctypes.c_int, ctypes.c_int]

# lib.get_value.restype = ctypes.c_float
# lib.get_value.argtypes = [ctypes.c_void_p, ctypes.c_int]

# lib.set_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

# print("--- Starting MicroTensor ---")

# tensor_ptr = lib.create_tensor(2, 3)
# print(f"Tensor created at memory address: {tensor_ptr}")

# print("Setting index 0 to 5.5...")
# lib.set_value(tensor_ptr, 0, 5.5)

# val = lib.get_value(tensor_ptr, 0)
# print(f"Value read back from C: {val}")

# print("--- Success ---")

# lib.matmul.restype = ctypes.c_void_p
# lib.matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

# print("\n--- Testing Matrix Multiplication ---")

# a_ptr = lib.create_tensor(2, 3)

# data_a = [1, 2, 3, 4, 5, 6]
# for i, val in enumerate(data_a): 
#     lib.set_value(a_ptr, i, val)

# b_ptr = lib.create_tensor(3, 2)
# data_b = [1, 0, 0, 1, 1, 1]
# for i, val in enumerate(data_b):
#     lib.set_value(b_ptr, i, val)

# print("Multiplying inside C engine...")
# result_ptr = lib.matmul(a_ptr, b_ptr)

# val1 = lib.get_value(result_ptr, 0)
# val2 = lib.get_value(result_ptr, 1)

# print(f"Result[0,0] = {val1} (Expected 4.0)")
# print(f"Result[0,1] = {val2} (Expected 5.0)")

# lib.relu.argtypes = [ctypes.c_void_p]

# print("\n Testing ReLU Activation")

# t_ptr = lib.create_tensor(1, 4)

# lib.set_value(t_ptr, 0, -2)
# lib.set_value(t_ptr, 1, 5)
# lib.set_value(t_ptr, 2, -10)
# lib.set_value(t_ptr, 3, 3)

# print("Before ReLU: [-2, 5, -10, 3]")

# lib.relu(t_ptr)

# val0 = lib.get_value(t_ptr, 0)
# val1 = lib.get_value(t_ptr, 1)
# val2 = lib.get_value(t_ptr, 2)
# val3 = lib.get_value(t_ptr, 3)

# print(f"After ReLU: [{val0}, {val1}, {val2}, {val3}]")

# if val0 == 0 and val2 == 0:
#     print("SUCCESS: Negative numbers were killed")
# else:
#     print("FAILURE: Something went wrong.")

from wrapper import Tensor, LinearLayer

print("--- MicroTensor High-Level Test ---")

# A = Tensor(2, 3, data=[1, 2, 3, 4, 5, 6])
# B = Tensor(3, 2, data=[1, 0, 0, 1, 1, 1])

# print("Tensor A: ", A)
# print("Tensor B: ", B)

# C = A.matmul(B)
# print("A * B = ", C)

# D = Tensor(1, 4, data=[-2, 4, -7, 3])
# print("Before ReLU: ", D)
# D.relu()
# print("After ReLU: ", D)

print("--- AI Inference Test ---")

input_data = Tensor(1, 3, data=[0.5, -0.2, 0.9])
print("Input:", input_data)

layer = LinearLayer(3, 2)
print("Weights:", layer.weights)

output = layer.forward(input_data)

output.add(layer.bias)

output.relu()

print("Final Prediction:", output)