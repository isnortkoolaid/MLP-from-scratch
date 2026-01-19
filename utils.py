import numpy as np
def relu(input):
    return np.maximum(0, input)
def relu_derivative(input):
    return (input > 0).astype(np.float64)
def softmax(input):
    exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
    return exp_input / np.sum(exp_input, axis=1, keepdims=True)
def cross_entropy_loss(predictions, targets):
    sample_count = predictions.shape[0]
    #print(targets)
    clipped_preds = np.clip(predictions, 1e-15, 1 - 1e-15)
    #print(clipped_preds)
    log_likelihood = sum(-np.log(clipped_preds) * targets)
    loss = np.sum(log_likelihood) / sample_count
    return loss

