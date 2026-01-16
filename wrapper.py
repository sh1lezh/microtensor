import ctypes
import random

_lib = ctypes.CDLL('./libtensor.so')

_lib.create_tensor.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.create_tensor.restype = ctypes.c_void_p

_lib.get_value.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.get_value.restype = ctypes.c_float

_lib.set_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]

_lib.matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.matmul.restype = ctypes.c_void_p

_lib.relu.argtypes = [ctypes.c_void_p]

_lib.add_tensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

class Tensor: 
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols

        self.c_ptr = _lib.create_tensor(rows, cols)

        if data: 
            if len(data) != rows * cols: 
                raise ValueError("Data size doesn't match dimension!")
            
            for i, val in enumerate(data):
                _lib.set_value(self.c_ptr, i, float(val))

    def __repr__(self):
        vals = []
        for i in range(self.rows * self.cols): 
            vals.append(_lib.get_value(self.c_ptr, i))
        return f"Tensor({self.rows}x{self.cols}: {vals})"
    
    def matmul(self, other):
        result_ptr = _lib.matmul(self.c_ptr, other.c_ptr)

        new_t = Tensor(self.rows, other.cols)

        new_t.c_ptr = result_ptr
        return new_t
    
    def relu(self):
        _lib.relu(self.c_ptr)
        return self
    
    def add(self, other):
        _lib.add_tensor(self.c_ptr, other.c_ptr)
        return self
    
class LinearLayer:
    def __init__(self, input_size, output_size):
        weights_data = [random.uniform(-1, 1) for _ in range(input_size * output_size)]
        self.weights = Tensor(input_size, output_size, data=weights_data)

        bias_data = [random.uniform(-0.1, 0.1) for _ in range(output_size)]
        self.bias = Tensor(1, output_size, data=bias_data)

    def forward(self, input_tensor):
        hidden = input_tensor.matmul(self.weights)
        return hidden