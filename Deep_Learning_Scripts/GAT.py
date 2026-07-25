#GAT
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:30:02 2026

@author: nkarmokar
"""

import numpy as np

# ======================================
# Utility Functions
# ======================================

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.2):
    return np.where(x > 0, 1, alpha)

def softmax(x, axis=1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy(pred, y):
    m = y.shape[0]
    return -np.sum(y * np.log(pred + 1e-8)) / m


# ======================================
# Toy Graph
# ======================================

A = np.array([
    [1,1,0,0],
    [1,1,1,1],
    [0,1,1,0],
    [0,1,0,1]
], dtype=float)  # self-loops included

# Node features (4 nodes, 3 features)
X = np.array([
    [1,0,1],
    [0,1,0],
    [1,1,0],
    [0,0,1]
], dtype=float)

# Labels
y = np.array([0,1,0,1])
num_classes = 2

y_onehot = np.zeros((y.size, num_classes))
y_onehot[np.arange(y.size), y] = 1


# ======================================
# GAT Class
# ======================================

class GAT:

    def __init__(self, input_dim, hidden_dim, output_dim):

        # Weight matrix
        self.W = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)

        # Attention vector
        self.a = np.random.randn(2 * hidden_dim, 1) * 0.01

        # Output layer
        self.W_out = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)

    def forward(self, X, A):

        self.X = X
        self.A = A

        N = X.shape[0]

        # Linear transformation
        self.H = X.dot(self.W)  # (N, hidden_dim)

        # Compute attention scores
        self.e = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    concat = np.concatenate([self.H[i], self.H[j]])
                    self.e[i, j] = leaky_relu(concat.dot(self.a).item())

        # Mask non-neighbors
        self.e_masked = np.where(A == 1, self.e, -9e15)

        # Attention coefficients
        self.alpha = softmax(self.e_masked, axis=1)

        # Aggregate
        self.H_prime = self.alpha.dot(self.H)

        # Output layer
        self.Z = self.H_prime.dot(self.W_out)
        self.out = softmax(self.Z)

        return self.out

    def backward(self, y, lr=0.01):

        m = y.shape[0]

        # Output gradient
        dZ = self.out - y
        dW_out = self.H_prime.T.dot(dZ) / m

        dH_prime = dZ.dot(self.W_out.T)

        # Backprop through attention aggregation
        dAlpha = dH_prime.dot(self.H.T)
        dH = self.alpha.T.dot(dH_prime)

        # (For simplicity we skip full attention gradient derivation)
        # This implementation updates only W_out and W

        dW = self.X.T.dot(dH) / m

        # Update
        self.W_out -= lr * dW_out
        self.W -= lr * dW


# ======================================
# Training
# ======================================

gat = GAT(input_dim=3, hidden_dim=5, output_dim=2)

epochs = 200
lr = 0.01

for epoch in range(epochs):

    output = gat.forward(X, A)
    loss = cross_entropy(output, y_onehot)

    gat.backward(y_onehot, lr)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ======================================
# Evaluation
# ======================================

pred = np.argmax(gat.forward(X, A), axis=1)
accuracy = np.mean(pred == y)

print("Final Accuracy:", accuracy)


#GAT using build-in function
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

# Load dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# ----------------------------
# GAT Model
# ----------------------------
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim, heads=8, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create model
model = GAT(dataset.num_features, 8, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# ----------------------------
# Training
# ----------------------------
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())

print("GAT Test Accuracy:", acc)