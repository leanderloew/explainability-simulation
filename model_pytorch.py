
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
        #weights=self.soft(similarity)
        weights=logistic(similarity)
        #weights=F.softmax(similarity, dim=1)
        
        #Aggregate
        aggregations=torch.einsum('btj,btk->bjk', value, weights)

        if self.return_weights == True:
            return aggregations,weights
        else:
            return aggregations
        

#class logistic(nn.Module):
#    
#    def __init__(c=1,a=20,b=np.e):
#        super().__init__()
#        self.c=c
#        self.a=a
#        self.b=b
#    def forward(x):
#        
#        return self.c/(1+self.a*self.b**(-x))
    
def logistic(x,c=1,a=20,b=np.e):
    return c/(1+a*b**(-x))       
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1,nheads=200,share_params=True):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model,nheads=200,share_params=share_params)
        self.ff = FeedForward(d_model,nheads=200,share_params=share_params,dropout=0)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x1,atn=self.attn(x,x,x)
        #here we add the original x 
        #x2 = self.norm_1(self.dropout_1(x1)+x)        
        x3 = self.ff(x1)
        
        #x2 = self.dropout_1(x1)        
        #x3 = self.dropout_2(self.ff(x2))
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
    #scores=scores-7
    #scores = F.softmax(scores, dim=-1)
    scores = logistic(scores)
    #score=torch.relu(scores)
    #scores = F.sigmoid(scores)
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
            self.linear_1 = nn.Linear(d_model, d_model)
            self.linear_2 = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        #x = self.linear_2(x)
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
            
            #self.q_linear.bias=nn.Parameter(torch.tensor(np.repeat(0,d_model)).float())
            #self.v_linear.bias=nn.Parameter(torch.tensor(np.repeat(0,d_model)).float())
            #self.k_linear.bias=nn.Parameter(torch.tensor(np.repeat(0,d_model)).float())
                        
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
#A hierachical model 

#Main idea: We dont want context aware features, but a context aware querry, 

class simple_fraud_model_exp(nn.Module):
    def __init__(self, d_model=32,heads=1,nlay=1,dropout=0,SelfA=True,return_w=False):
        
        super(simple_fraud_model_exp,self).__init__()
        self.return_w=return_w
        emb_d_1=int(d_model/2)
        emb_d_2=int(d_model/2)
        
        #two embeddings 
        self.embedding_1=nn.Embedding(num_embeddings=21,embedding_dim=emb_d_1)
        self.embedding_2=nn.Embedding(num_embeddings=21,embedding_dim=emb_d_2)

        
        if SelfA==True:
            self.encoder_layers=EncoderLayer(d_model=d_model,heads=heads,dropout=dropout,share_params=True)

        if SelfA==False:
            self.encoder_layers=FeedForward(d_model)

        self.mula=multi_attention(input_dim=d_model,key_dim=d_model,nheads=1,return_weights=True,value_dim=d_model)
        
        self.initial_linear=nn.Linear(d_model,d_model)
        
        self.fully_con=nn.Linear(d_model,d_model*4)
        self.relu=nn.ReLU()
        
        self.fully_con_1=nn.Linear(d_model*4,d_model*4)
        self.relu_1=nn.ReLU()
        
        self.final_fully_con=nn.Linear(d_model*4,1)
        self.sig=nn.Sigmoid()
        self.selfa=SelfA
        
        self.v2_ex=nn.Linear(d_model,d_model)
        self.k2_ex=nn.Linear(d_model,d_model)
        self.q2_ex=nn.Linear(d_model,d_model)
        self.soft=torch.nn.Sigmoid()       
        
    def forward(self, x1,x2,x3,return_w):
        e1=self.embedding_1(x1)
        e2=self.embedding_2(x2)

        cat=torch.cat([e1,e2],dim=2)
        if self.selfa==True:
            feat,w2=self.encoder_layers(cat)
        else:
            feat=self.encoder_layers(cat)
        
        
        ag,weights_=self.mula(feat)
        q2=self.q2_ex(ag.squeeze()).unsqueeze(dim=2)
        k2=self.k2_ex(feat)
        v2=self.v2_ex(feat)
        
        #Here instead of using the ag directly we use the ag as a seconde kind of query to aggregate with it. 
        #doesnt really change much.
        weights=self.soft(torch.einsum('btj,bjk->btk', k2, q2))
        ag=torch.einsum('btj,btk->bjk', v2, weights)        
        
        fc=self.relu(self.fully_con(ag.squeeze()))
        fc=self.relu_1(self.fully_con_1(fc))
        
        preds=self.sig(self.final_fully_con(fc))
        if return_w==False:
            return preds.squeeze()
        
        if return_w==True:
            if self.selfa==True:
                #w2=w2.squeeze()
                #w2=w2.permute((0,2,1))
                #weights=torch.bmm(w2,weights)
                return preds.squeeze(),torch.bmm(w2.squeeze(),weights)
            else:
                return preds.squeeze(),weights