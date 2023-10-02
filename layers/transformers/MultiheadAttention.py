from torch import nn
from layers.transformers.SCPAttention import SCPAttention
from torch import Tensor
from typing import Callable, Optional

class MultiheadAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 d_k=None, 
                 d_v=None, 
                 res_attention=False, 
                 attn_dropout=0., 
                 proj_dropout=0., 
                 qkv_bias=True, 
                 lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super(MultiheadAttention, self).__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = SCPAttention(d_model, 
                                     n_heads, 
                                     attn_dropout=attn_dropout, 
                                     res_attention=self.res_attention, 
                                     lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self,
                Q: Tensor,
                K: Optional[Tensor] = None,
                V: Optional[Tensor] = None,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        # q_s: [bs x n_heads x max_q_len x d_k]
        # k_s: [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        # v_s: [bs x n_heads x q_len x d_v]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s,
                                                              k_s,
                                                              v_s,
                                                              prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s,
                                                 k_s,
                                                 v_s,
                                                 key_padding_mask=key_padding_mask,
                                                 attn_mask=attn_mask)

        # output: [bs x n_heads x q_len x d_v],
        # attn: [bs x n_heads x q_len x q_len],
        # scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        # output: [bs x q_len x n_heads * d_v]
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights
