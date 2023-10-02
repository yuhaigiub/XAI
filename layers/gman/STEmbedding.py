import torch
from torch import nn
import torch.nn.functional as F

from layers.gman.FC import FC

class STEmbedding(nn.Module):
    def __init__(self, device, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.device = device
        
        self.FC_se = FC(device,
                        input_dims=[64, D], 
                        units=[D, D], 
                        activations=[F.relu, None],
                        bn_decay=bn_decay)
        
        # input_dims = time step per day + days per week = 288 + 7 = 295
        self.FC_te = FC(device,
                        input_dims=[295, D], 
                        units=[D, D], 
                        activations=[F.relu, None],
                        bn_decay=bn_decay) 

    def forward(self, SE, TE, T=288):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
            
        TE = torch.cat((dayofweek, timeofday), dim=-1).to(self.device)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        
        del dayofweek, timeofday
        return SE + TE