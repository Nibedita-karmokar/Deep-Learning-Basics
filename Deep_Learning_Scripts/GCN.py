#GNN
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 17:17:19 2026

@author: nkarmokar
"""



import numpy as np

# =====================================
# Utility Functions
# =====================================

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(pred, y):
    m = y.shape[0]
    return -np.sum(y * np.log(pred + 1e-8)) / m


# =====================================
# Create Toy Graph
# =====================================

# Adjacency matrix (4 nodes)
A = np.array([
    [0,1,0,0],
    [1,0,1,1],
    [0,1,0,0],
    [0,1,0,0]
], dtype=float)

# Add self-loops
A_hat = A + np.eye(A.shape[0])

# Degree matrix
D = np.diag(np.sum(A_hat, axis=1))

# Normalized adjacency
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
A_norm = D_inv_sqrt.dot(A_hat).dot(D_inv_sqrt)

# Node features (4 nodes, 3 features each)
X = np.array([
    [1,0,1],
    [0,1,0],
    [1,1,0],
    [0,0,1]
], dtype=float)

# Labels (2 classes)
y = np.array([0,1,0,1])
num_classes = 2

# One-hot encoding
y_onehot = np.zeros((y.size, num_classes))
y_onehot[np.arange(y.size), y] = 1


# =====================================
# GCN Class
# =====================================

class GCN:

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2/input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2/hidden_dim)

    def forward(self, X, A_norm):

        self.X = X
        self.A = A_norm

        # Layer 1
        self.Z1 = A_norm.dot(X).dot(self.W1)
        self.H1 = relu(self.Z1)

        # Layer 2
        self.Z2 = A_norm.dot(self.H1).dot(self.W2)
        self.out = softmax(self.Z2)

        return self.out

    def backward(self, y, lr=0.01):

        m = y.shape[0]

        # Output layer gradient
        dZ2 = self.out - y
        dW2 = self.H1.T.dot(self.A.T).dot(dZ2) / m

        # Backprop to hidden layer
        dH1 = self.A.dot(dZ2).dot(self.W2.T)
        dZ1 = dH1 * relu_derivative(self.Z1)
        dW1 = self.X.T.dot(self.A.T).dot(dZ1) / m

        # Update weights
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2


# =====================================
# Training
# =====================================

gcn = GCN(input_dim=3, hidden_dim=5, output_dim=2)

epochs = 200
learning_rate = 0.05

for epoch in range(epochs):

    output = gcn.forward(X, A_norm)
    loss = cross_entropy(output, y_onehot)

    gcn.backward(y_onehot, learning_rate)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# =====================================
# Evaluation
# =====================================

predictions = np.argmax(gcn.forward(X, A_norm), axis=1)
accuracy = np.mean(predictions == y)

print("Final Accuracy:", accuracy)

#GCN using built-in function:
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Load dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# ----------------------------
# GCN Model
# ----------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create model
model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

print("GCN Test Accuracy:", acc)