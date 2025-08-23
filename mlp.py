import numpy as np
from utils import relu, relu_derivative, softmax, cross_entropy_loss
class mlp:
    def __init__(self, input_size=784, hidden_size=512, output_size = 10):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = 0.005
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    def backward(self, x, dout):
        d2 = np.dot(dout, self.w2.T) * relu_derivative(self.z1)
        dw2 = np.dot(self.a1.T, dout)
        db2 = np.sum(dout, axis=0, keepdims=True)
        dw1 = np.dot(x.T, d2)
        db1 = np.sum(d2, axis=0, keepdims=True)
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    