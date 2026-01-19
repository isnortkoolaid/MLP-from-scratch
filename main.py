import numpy as np
from mlp import mlp
from utils import cross_entropy_loss
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.data = mnist.data / 255.0
train_data, test_data, train_target, test_target = train_test_split(
    mnist.data, mnist.target, test_size=0.2, random_state=42)
encoder = OneHotEncoder(sparse_output=False)
train_target_oh = encoder.fit_transform(train_target.reshape(-1, 1))
test_target_oh = encoder.transform(test_target.reshape(-1, 1))
print(train_data.shape)
print(train_target.shape)
print(mnist.data[2])
print(mnist.target[2])
model = mlp()
epochs = 10
batch_size = 64
for epoch in range(epochs):
    for i in range(0, train_data.shape[0], batch_size): 
        x = train_data[i:i+batch_size]
        y = train_target_oh[i:i+batch_size]
        #print(y)
        predictions = model.forward(x)
        loss = cross_entropy_loss(predictions, y)
        dout = predictions - y
        model.backward(x, dout)
        if (i) % 1600 == 0:
            print(f"Epoch {epoch}, Sample {i+1}, Loss: {loss:.8f}")
correct = 0
for i in range(test_data.shape[0]):
    x = test_data[i].reshape(1, -1)
    y = int(test_target[i])
    predictions = model.forward(x)
    if np.argmax(predictions) == y:
        correct += 1
accuracy = correct / test_data.shape[0]
print(f"Test Accuracy: {accuracy * 100:.8f}%")