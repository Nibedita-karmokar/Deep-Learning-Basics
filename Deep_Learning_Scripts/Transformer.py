#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:09:42 2026

@author: nkarmokar
"""

import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim)/np.sqrt(in_dim)
        self.b = np.zeros((1, output_dim))
       
        def forward(self, x):
            return np.dot(x, self.W) + self.b
       
class LayerNorm:
    def __init__(self, dim, eps = 1e-6):
        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))
        self.eps = eps
       
    def forward(self, x):
        mean = np.mean(x, axis = -1, keepdims = True)
        var  = np.var(x, axis = -1, keepdims = True)
        norm = (x - mean)/np.sqrt(var + self.eps)
        return self.gamma*norm + self.beta
   
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return exp/np.sum(exp, axis = -1, keepdims = True)

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
   
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos/np.power(10000, (2*i)/d_model)    
            PE[pos, i] = np.sin(angle)
           
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(angle)
               
    return PE

def scaled_dot_product(Q, K, V):
    d_k = Q.shape[-1]
   
    scores = np.matmul(Q, K.transpose(0, 2, 1))/np.sqrt(d_k)
    weights = softmax(scores)
    output = np.matmul(weights, V)
   
    return output

class FeedForward:
    def __init__(self, d_model, hidden_dim):
        self.fc1 = Linear(d_model, hidden_dim)
        self.fc2 = Linear(hidden_dim, d_model)
       
    def forward(self, x):
        return self.fc2.forward(relu(self.fc1.forward(x)))

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
       
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
       
        self.Wo = Linear(d_model, d_model)
       
    def split_heads(self, x):
        batch, seq_len, dim = x.shape 
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
       
        return x.transpose(0, 2, 1, 3)
   
    def combine_heads(self, x):
        batch, heads, seq_len, dim = x.shape
        x = x.transpose(0, 2, 1, 3)
       
        return x.reshape(batch, seq_len, heads*dim)
   
    def forward(self, x):
        Q = self.Wq.forward(x)
        K = self.Wk.forward(x)
        V = self.Wv.forward(x)
       
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
       
        attention = scaled_dot_product(Q, K, V)
        combined = self.combine_heads(attention)      
        output = self.Wo.forward(combined)
       
        return output
       
   
class EncoderBlock:
    def __init__(self, d_model, num_heads, hidden_dim):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, hidden_dim)
       
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
       
    def forward(self, x):
        attn = self.mha.forward(x)       
        x = self.norm1.forward(x + attn)
        ff = self.ffn.forward(x)
        x = self.norm2.forward(x+ff)
       
        return x   
   
class TransformerEncoder:
    def __init__(self, num_layers, d_model, num_heads, hidden_dim):
        self.layers = []
       
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(d_model, num_heads, hidden_dim))
           
    def forward(self, x):
        for layer in range(num_layers):
            x = layer.forward(x)  
            
    return x

batch_size = 2
seq_len = 5
d_model = 32

x = np.random.randn(batch_size, seq_len, d_model)

x += positional_encoding(seq_len, d_model)

model = TrnasformerEncoder(num_layers = 2, d_model = 32, num_heads = 4, hidden_dim = 32)

output = model.forward(x)
   
   
