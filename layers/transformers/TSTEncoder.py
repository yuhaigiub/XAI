from torch import Tensor
from typing import Callable, Optional
from torch import nn

from layers.transformers.TSTEncoderLayer import TSTEncoderLayer

# Cell
class TSTEncoder(nn.Module):
    def __init__(self,
                 q_len,
                 d_model,
                 n_heads,
                 d_k=None,
                 d_v=None,
                 d_ff=None,
                 norm='BatchNorm',
                 attn_dropout=0.,
                 dropout=0.,
                 activation='gelu',
                 res_attention=False,
                 n_layers=1,
                 pre_norm=False,
                 store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len,
                                                     d_model,
                                                     n_heads=n_heads,
                                                     d_k=d_k,
                                                     d_v=d_v,
                                                     d_ff=d_ff,
                                                     norm=norm,
                                                     attn_dropout=attn_dropout,
                                                     dropout=dropout,
                                                     activation=activation,
                                                     res_attention=res_attention,
                                                     pre_norm=pre_norm,
                                                     store_attn=store_attn) for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self,
                src: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(output, 
                                     prev=scores, 
                                     key_padding_mask=key_padding_mask, 
                                     attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers:
                output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
