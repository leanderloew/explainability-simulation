
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
from torch.nn import init
import math
from torch.nn import functional as F
import os 

class multi_attention(torch.nn.Module):
    def __init__(self, input_dim=100, key_dim=50,value_dim=50,nheads=10,return_weights=True):
        super(multi_attention, self).__init__()
        
        self.key_extract=nn.Linear(input_dim,key_dim)
        self.value_extract=nn.Linear(input_dim,value_dim)
        
        self.query = Parameter(torch.Tensor(key_dim, nheads))
        self.soft=nn.Sigmoid()
        
        self.return_weights=return_weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))
    
    def forward(self, embeds):
        value=self.value_extract(embeds)
        keys=self.key_extract(embeds)
        
        similarity=torch.einsum('btj,jk->btk', keys, self.query)
        weights=self.soft(similarity)
        
        #Aggregate
        aggregations=torch.einsum('btj,btk->bjk', value, weights)

        if self.return_weights == True:
            return aggregations,weights
        else:
            return aggregations
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1,nheads=200,share_params=True):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model,nheads=200,share_params=share_params)
        self.ff = FeedForward(d_model,nheads=200,share_params=share_params)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x1,atn=self.attn(x,x,x)
        #here we add the original x 
        x2 = self.norm_1(self.dropout_1(x1)+x)        
        x3 = self.norm_2(self.dropout_2(self.ff(x2))+x2)
        
        return x3,atn

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm    
    
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    #if dropout is not None:
    #    scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output,scores

class FeedForward(nn.Module):
    def __init__(self, d_model , dropout = 0.1,nheads=200,share_params=True):
        super().__init__() 
        d_ff=int(d_model*2)
        # We set d_ff as a default to 2048
        if share_params==False:
            self.linear_1 = simple_projection_3d(d_model, d_ff,nheads)
            self.linear_2 = simple_projection_3d(d_ff, d_model,nheads)
        if share_params==True:
            self.linear_1 = nn.Linear(d_model, d_ff)
            self.linear_2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1,nheads=200,share_params=True):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        if share_params==False:

            self.q_linear = simple_projection_3d(d_model, d_model,nheads)
            self.v_linear = simple_projection_3d(d_model, d_model,nheads)
            self.k_linear = simple_projection_3d(d_model, d_model,nheads)
            
        if share_params==True:
            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
                        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

# calculate attention using function we will define next
        scores,w = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output,w    
