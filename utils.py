import numpy as np
def relu(input):
    return np.maximum(0, input)
def relu_derivative(input):
    return (input > 0).astype(np.float64)
def softmax(input):
    exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
    return exp_input / np.sum(exp_input, axis=1, keepdims=True)
def cross_entropy_loss(predictions, target, epsilon=1e-12):
    m = target.shape[0]
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    log_likelihood = -np.log(predictions[range(m), target])
    loss = np.sum(log_likelihood) / m
    return loss


