import torch.nn.functional as F
from torch import nn

from layers.FC import FC
from layers.SpatialAttention import SpatialAttention

class Model(nn.Module):
    def __init__(self, device, SE, bn_decay):
        super(Model, self).__init__()
        
        self.device = device
        
        L = 1
        K = 4
        d = 4
        D = K * d
        self.SE = SE
        
        self.FC_1 = FC(self.device, [1, D], [D, D], [F.relu, None], bn_decay)
        
        self.attention1 = [SpatialAttention(device, K, d, bn_decay) for _ in range(L)]
        self.attention2 = [SpatialAttention(device, K, d, bn_decay) for _ in range(L)]
        
        self.FC_2 = FC(self.device, [D, D], [D, 1], [F.relu, None], bn_decay)
    
    def forward(self, X, SE):
        X = self.FC_1(X)
        SE = self.SE.unsqueeze(0).unsqueeze(0).repeat(X.shape[0], X.shape[1], 1, 1)
        
        for layer in self.attention1:
            X = layer(X, SE)
        
        for layer in self.attention2:
            X = layer(X, SE)
        
        X = self.FC_2(X)
        
        del SE
        return X