import torch
from torch import nn
from torch import Tensor
from typing import Callable, Optional
from layers.transformers.TSTEncoder import TSTEncoder

from utils.utils import positional_encoding


class TSTiEncoder(nn.Module): #i means channel-independent
    def __init__(self, 
                 c_in, 
                 patch_num, 
                 patch_len, 
                 max_seq_len=1024,
                 n_layers=3, 
                 d_model=128, 
                 n_heads=16, 
                 d_k=None, 
                 d_v=None,
                 d_ff=256, 
                 norm='BatchNorm', 
                 attn_dropout=0., 
                 dropout=0., 
                 act="gelu", 
                 store_attn=False,
                 key_padding_mask='auto', 
                 padding_var=None, 
                 attn_mask=None, 
                 res_attention=True, 
                 pre_norm=False,
                 pe='zeros', 
                 learn_pe=True, 
                 verbose=False, 
                 **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, 
                                  d_model, 
                                  n_heads, 
                                  d_k=d_k, 
                                  d_v=d_v, 
                                  d_ff=d_ff, 
                                  norm=norm, 
                                  attn_dropout=attn_dropout, 
                                  dropout=dropout,
                                  pre_norm=pre_norm, 
                                  activation=act, 
                                  res_attention=res_attention, 
                                  n_layers=n_layers, 
                                  store_attn=store_attn)
    
    def forward(self, x) -> Tensor: # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2) # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x) # x: [bs x nvars x patch_num x d_model]
        
        # u: [bs * nvars x patch_num x d_model]
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])) 
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u) # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1])) # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2) # z: [bs x nvars x d_model x patch_num]
        
        return z