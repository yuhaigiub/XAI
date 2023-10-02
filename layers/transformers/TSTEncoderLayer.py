from torch import nn
from layers.transformers.MultiheadAttention import MultiheadAttention
from layers.transformers.Transpose import Transpose
from torch import Tensor
from typing import Callable, Optional

from utils.utils import get_activation_fn

class TSTEncoderLayer(nn.Module):
    def __init__(self, 
                 q_len, 
                 d_model, 
                 n_heads, 
                 d_k=None, 
                 d_v=None, 
                 d_ff=256, 
                 store_attn=False,
                 norm='BatchNorm', 
                 attn_dropout=0, 
                 dropout=0., 
                 bias=True, 
                 activation=nn.GELU, 
                 res_attention=False, 
                 pre_norm=False):
        super(TSTEncoderLayer, self).__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, 
                                             n_heads, 
                                             d_k, 
                                             d_v, 
                                             attn_dropout=attn_dropout, 
                                             proj_dropout=dropout, 
                                             res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias), 
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self,
                src:Tensor, 
                prev:Optional[Tensor]=None, 
                key_padding_mask:Optional[Tensor]=None, 
                attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, 
                                                src, 
                                                src, 
                                                prev, 
                                                key_padding_mask=key_padding_mask, 
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, 
                                        src, 
                                        src, 
                                        key_padding_mask=key_padding_mask, 
                                        attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src