#MLP
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:33:49 2026

@author: nkarmokar
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (z > 0).astype(float)

def softmax(Z):
    z_shift = Z - np.max(Z, axis = 1, keepdims = True)
    expZ = np.exp(Z_shift)
    return expZ/(np.sum(expZ, axis = 1, keepdims = True))

def cross_entropy(pred, y):
    m = y.shape[0]
    return -np.sum(y*np.log(pred + 1e-8))/m

mnist = fetch_openml("mnist_784", version = 1)

X = mnist.data.astype(np.float32)/255
y = mnist.target.astype(int)

y_onehot = np.zeros((y.size, 10))
y_onehot[np.arrange(y.size), y] = 1

X_train, y_train, X_test, y_test = train_test_split(X, y_onehot, test_size = 0.2, random_state = 42)

input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
b1 = np.random.randn(1, hidden_size)

W1 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/input_size)
b1 = np.random.randn(1, output_size)

learning_rate = 0.01
epochs = 10
batch_size = 128

for epoch in ranges(epochs):
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    y_train = y_train[permutation]
   
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
       
        m = X_batch.shape[0]
       
       
        Z1 = np.dot(X_batch, W1) +b1
        A1 = relu(Z1)
       
        Z2 = np.dot(A1, W2) +b2
        A2 = softmax(Z2)
       
        loss = cross_entropy(A2, y_batch)
       
        dZ2 = A2 - y_batch
       
        dW2 = np.dot(A1.T, dZ2)/m
        db2 = np.sum(dZ2, axis = 0, keepdims = True)/m
        dA1 = np.dot(dZ2, W2.T)
       
        dZ1 = dA1*relu_derivative(Z1)        
        dW1 = np.dot(x_batch.T, dZ1)/m
        db1 = np.sum(dZ1, axis = 0, keepdims = True)/m
       
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2
       
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
       
    print("epochs", epoch+1, "Loss:", round(loss, 4))
   
   
Z1 = np.dot(X_test, W1) +b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = softmax(Z2)

predictions = np.argmax(A2, axis = 1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)

print("Test Accuracy:", round(accuracy * 100, 2), "%")
       