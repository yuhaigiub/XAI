from torch import nn
from torch import Tensor
from typing import Callable, Optional
from layers.transformers.FlattenHead import FlattenHead

from layers.transformers.TSTiEncoder import TSTiEncoder

class PatchTST_Backbone(nn.Module):
    def __init__(self,
                 c_in:int, 
                 context_window:int, 
                 target_window:int, 
                 patch_len:int, 
                 stride:int, 
                 max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, 
                 d_model=128, 
                 n_heads=16, 
                 d_k:Optional[int]=None, 
                 d_v:Optional[int]=None,
                 d_ff:int=256, 
                 norm:str='BatchNorm', 
                 attn_dropout:float=0., 
                 dropout:float=0., 
                 act:str="gelu", 
                 key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, 
                 attn_mask:Optional[Tensor]=None, 
                 res_attention:bool=True, 
                 pre_norm:bool=False, 
                 store_attn:bool=False,
                 pe:str='zeros', 
                 learn_pe:bool=True, 
                 fc_dropout:float=0., 
                 head_dropout = 0, 
                 padding_patch = None,
                 pretrain_head:bool=False, 
                 head_type = 'flatten', 
                 individual = False, 
                 revin = True, 
                 affine = True, 
                 subtract_last = False,
                 verbose:bool=False, 
                 **kwargs):
        
        super(PatchTST_Backbone, self).__init__()
        
        # # RevIn
        # self.revin = revin
        # if self.revin: 
        #     self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
                
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, 
                                    patch_num=patch_num, 
                                    patch_len=patch_len, 
                                    max_seq_len=max_seq_len,
                                    n_layers=n_layers, 
                                    d_model=d_model, 
                                    n_heads=n_heads, 
                                    d_k=d_k, 
                                    d_v=d_v, 
                                    d_ff=d_ff,
                                    attn_dropout=attn_dropout, 
                                    dropout=dropout, 
                                    act=act, 
                                    key_padding_mask=key_padding_mask, 
                                    padding_var=padding_var,
                                    attn_mask=attn_mask, 
                                    res_attention=res_attention, 
                                    pre_norm=pre_norm, 
                                    store_attn=store_attn,
                                    pe=pe, 
                                    learn_pe=learn_pe, 
                                    verbose=verbose, 
                                    **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            # custom head passed as a partial func with all its kwargs
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten': 
            self.head = FlattenHead(self.individual, 
                                     self.n_vars, 
                                     self.head_nf, 
                                     target_window, 
                                     head_dropout=head_dropout)
        
    
    def forward(self, z): # z: [bs x nvars x seq_len]
        # # norm
        # if self.revin: 
        #     z = z.permute(0,2,1)
        #     z = self.revin_layer(z, 'norm')
        #     z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        # z: [bs x nvars x patch_num x patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # z: [bs x nvars x patch_len x patch_num] 
        z = z.permute(0,1,3,2)
        
        # model
        z = self.backbone(z) # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)     # z: [bs x nvars x target_window]
        
        # # denorm
        # if self.revin: 
        #     z = z.permute(0,2,1)
        #     z = self.revin_layer(z, 'denorm')
        #     z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))
