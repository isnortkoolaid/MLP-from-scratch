import numpy as np
from utils import relu, relu_derivative, softmax, cross_entropy_loss
class mlp:
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, output_size = 10):
        self.w1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.w2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.w3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        self.learning_rate = 0.01
    def forward(self, x):
        self.z1 = np.matmul(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.matmul(self.a1, self.w2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.matmul(self.a2, self.w3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3
    def backward(self, x, dout):
        dz2 = np.matmul(dout, self.w3.T) * relu_derivative(self.z2)
        dw3 = np.matmul(self.a2.T, dout)
        db3 = np.sum(dout, axis=0, keepdims=True)
        dz1 = np.matmul(dz2, self.w2.T) * relu_derivative(self.z1)
        dw2 = np.matmul(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dw1 = np.matmul(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    