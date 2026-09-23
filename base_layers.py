import torch
from torch import nn
import numpy as np


class FactorizeEmbeddingLayer(nn.Module):

    def __init__(self,vocab_size,d_model,maxlen):

        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.maxlen = maxlen

        self.emb = nn.Embedding(self.vocab_size,self.d_model // 2)
        self.fact_layer = nn.Linear(self.d_model // 2, self.d_model)
        self.pos_encoding = nn.Parameter(self._positional_encoding(),requires_grad=True)

    def _positional_encoding(self):
        pos = np.zeros((self.maxlen, self.d_model), dtype=np.float32)
        for p in range(self.maxlen):
            for i in range(self.d_model):
                exponent = 2 * (i // 2) / self.d_model
                angle = p / (10000 ** exponent)

                if i % 2 == 0:
                    pos[p, i] = np.sin(angle)
                else:
                    pos[p, i] = np.cos(angle)
                    
        return torch.tensor(pos,dtype=torch.float32)

    def forward(self,x):
        mask = x == 0
        x = self.fact_layer(self.emb(x)) + self.pos_encoding[:x.shape[1], :].unsqueeze(0)
        return x,mask


class MultiHeadAttention(nn.Module):

    def __init__(self,d_model,n_heads,look_ahead_mask=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_scores = None
        self.d_head = self.d_model // self.n_heads
        self.look_ahead_mask = look_ahead_mask

        self.q = nn.Linear(self.d_model,self.d_model,False)
        self.k = nn.Linear(self.d_model, self.d_model, False)
        self.v = nn.Linear(self.d_model, self.d_model, False)

        self.attn_drop = nn.Dropout(0.1)
        self.lin = nn.Linear(d_model,d_model)
        self.lin_drop = nn.Dropout(0.1)

    def _split_heads(self,x):
        x = torch.reshape(x,shape=(x.shape[0],x.shape[1],self.n_heads,self.d_head))
        return torch.permute(x,(0,2,1,3))

    def forward(self,x):

        x,mask = x
        q,k,v = x
        mask = mask[:,torch.newaxis,torch.newaxis,:]
        maxlen = q.shape[1]
        b = q.shape[0]

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        k_trans = torch.permute(k,(0,1,3,2))
        attn = torch.matmul(q,k_trans) / (self.d_head ** 0.5)
        attn = torch.where(mask,-1e8,attn)

        if self.look_ahead_mask:
            look_ahead_mask = torch.triu(torch.ones(maxlen, maxlen, device=attn.device),
                                         diagonal=1).bool()
            attn = attn.masked_fill(look_ahead_mask, -1e8)

        attn = torch.softmax(attn,-1)
        self.attn_scores = attn
        attn = self.attn_drop(attn)

        x = torch.matmul(attn,v)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x,(b,maxlen,self.d_model))
        x = self.lin(x)

        return self.lin_drop(x)


class FeedForward(nn.Module):

    def __init__(self,d_model,ff_units):
        super().__init__()
        self.d_model = d_model
        self.ff_units = ff_units

        self.ff1 = nn.Linear(d_model,ff_units)
        self.ff2 = nn.Linear(ff_units,d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self,x):
        x = self.ff1(x)
        x = torch.nn.functional.gelu(x)
        x = self.ff2(x)
        x = self.drop(x)
        return x




















