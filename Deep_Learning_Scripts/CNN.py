#CNN
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 09:25:06 2026

@author: nkarmokar
"""
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy(pred, y):
    m = y.shape[0]
    return -np.sum(y * np.log(pred + 1e-8)) / m

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
       
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size)*0.1
       
    def forward(self, input):
        self.input = input
        batch_size, depth, height, width = input.shape
       
        out_h = height - self.filter_size + 1
        out_w = width - self.filter_size + 1
       
        self.output = np.zeros((batch_size, self.num_filters, out_h, out_w))
       
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        region = input[b, :, i:i+self.filter_size, j:j+self.filter_size]
                        self.output[b, f, i, j] = np.sum(region*self.filters[f])
                       
        return self.output
   
    def backward(self, d_out, lr):
        d_filters = np.zeros_like(self.filters)
        batch_size, depth, height, width = input.shape
       
        out_h = height - self.filter_size + 1
        out_w = width - self.filter_size + 1
       
        self.output = np.zeros((batch_size, self.num_filters, out_h, out_w))
       
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        region = input[b, :, i:i+self.filter_size, j:j+self.filter_size]
                        d_filters[f] += d_out[b,f,i,j]*region
        self.filters -= lr*d_filters/batch_size
       
class MaxPool:
    def __init__(self, size):
        self.size = size
       
    def forward(self, input):
        self.input = input
        batch, depth, height, width = input.shape
       
        out_h = height // self.size
        out_w = width // self.size
       
        self.output = np.zeros((batch, depth, out_h, out_w))
       
        for b in range(batch):
            for d in range(depth):
                for i in range(out_h):
                    for j in range(out_w):    
                        region = input[b, d, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size]
                        self.output[b, d, i, j] = np.max(region)
                       
        return self.output
   
    def backward(self, d_out):
        d_input = np.zeros_like(self.input)
        batch, depth, out_h, out_w = d_out.shape
       
        for b in range(batch):
            for d in range(depth):
                for i in range(out_h):
                    for j in range(out_w):
                        region = input[b, d, i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size]
                       
                        max_val = np.max(region)
                        for x in range(self.size):
                            for y in range(self.size):
                                if region[x, y] == max_val:
                                    d_input[b, d, i*self.size+x, j*self.size+y] = d_out[b, d, i, j]
        return d_input
   
   
class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)*0.1
        self.b = np.zeros((1, output_size))
       
    def forward(self, d_out, lr):
        self.input = x
        return np.dot(self.input*self.W) + self.b
   
    def backward(self, d_out, lr):
        m = self.input.shape[0]
       
        dW = np.dot(self.input.T, d_out)/m
        db = np.sum(d_out, axis = 0, keepdims = True)/m
        d_input = np.dot(d_out, self.W.T)
       
        self.W -= lr*dW
        self.b -= lr*db
       
        return d_input
   
     


print("Loading MNIST")
mnist = fetch_openml("mnist_784", version = 1)

X = mnist.data.astype(np.float32)/255
y = mnist.target.astype(int)

X = X[:5000]
y = y[:5000]

X = X.reshape(-1, 1, 28, 28)

y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size = 0.2, random_state=42)

conv = ConvLayer(num_filters = 8, filter_size = 3, input_depth = 1)
pool = MaxPool(size = 2)
fc1 = Dense(8*13*13, 128)
fc2 = Dense(128, 10)

lr = 0.01
epochs = 5
batch_size = 32

for epoch in range(epochs):
    print("Epoch: ", epoch+1)
   
    for u in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
       
        out = conv.forward(X_batch)
        out = relu(out)
        out = pool.forward(out)
       
        out = out.reshape(out.shape[0], -1)
        out = fc1.forward(out)
        out = relu(out)
       
        out = fc2.forward(out)
        probs = softmax(out)
       
        loss = cross_entropy(probs, y_batch)
       
        d_out = probs - y_batch
        d_out = fc2.backward(d_out, lr)
       
        d_out = d_out*relu_derivative(fc1.input)
        d_out = fc1.backward(d_out, lr)
       
        d_out = d_out.reshape(-1, 8, 13, 13)
        d_out = pool.backward(d_out)
       
        conv.backward(d_out, lr)
       
       
       
       
       
       
       
       
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST images are 28x28 grayscale
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = models.Sequential()

# 1st Convolutional Layer
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# 2nd Convolutional Layer
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

# Flatten before dense layers
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# -------------------------------
# 5. Evaluate on test data
# -------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


