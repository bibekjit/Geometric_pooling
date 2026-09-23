from base_layers import FeedForward, MultiHeadAttention
from torch import nn


class EncoderLayer(nn.Module):

    def __init__(self,d_model,n_heads,ff_units):

        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_units = ff_units

        self.attn_layer = MultiHeadAttention(d_model,n_heads)
        self.ff_layer = FeedForward(d_model,ff_units)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,inputs):
        x,mask = inputs
        attn = self.attn_layer([[x,x,x],mask])
        x = self.norm1(attn + x)
        ff = self.ff_layer(x)
        x = self.norm2(ff + x)
        return x


class DecoderLayer(nn.Module):

    def __init__(self,d_model,n_heads,ff_units):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_units = ff_units

        self.self_attn_layer = MultiHeadAttention(d_model, n_heads, look_ahead_mask=True)
        self.cross_attn_layer = MultiHeadAttention(d_model, n_heads)
        self.ff_layer = FeedForward(d_model, ff_units)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,inputs):

        x,mask,en_out,en_mask = inputs
        self_attn = self.self_attn_layer(((x,x,x),mask))
        x = self.norm1(self_attn + x)
        cross_attn = self.cross_attn_layer(((x,en_out,en_out),en_mask))
        x = self.norm2(x + cross_attn)
        ff = self.ff_layer(x)
        x = self.norm3(x + ff)
        return x

