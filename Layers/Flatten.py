import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.batch_size = None

    def forward(self, input_tensor):
        self.batch_size, *self.input_shape = input_tensor.shape
        self.input_shape = tuple(self.input_shape)
        return np.reshape(input_tensor, (self.batch_size, np.prod(self.input_shape)))

    def backward(self, error_tensor):

        return np.reshape(error_tensor, (self.batch_size, *self.input_shape))
