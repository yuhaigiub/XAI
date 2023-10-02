import torch.nn.functional as F
from torch import nn

from layers.gman.FC import FC
from layers.gman.STAttention import STAttention
from layers.gman.STEmbedding import STEmbedding
from layers.gman.TransformAttention import TransformAttention

class Model(nn.Module):
    def __init__(self, device, SE, bn_decay, num_his=12, num_pred=12):
        super(Model, self).__init__()
        
        self.device = device
        self.num_his = num_his
        self.num_pred = num_pred
        
        L = 1
        K = 8
        d = 8
        D = K * d
        self.SE = SE
        self.embedding = STEmbedding(device, D, bn_decay)
        self.FC_1 = FC(self.device, [1, D], [D, D], [F.relu, None], bn_decay)
        
        self.attention1 = nn.ModuleList([STAttention(device, K, d, bn_decay) for _ in range(L)])
        
        self.transformAttention = TransformAttention(device, K, d, bn_decay)
        
        self.attention2 = nn.ModuleList([STAttention(device, K, d, bn_decay) for _ in range(L)])
        
        self.FC_2 = FC(self.device, [D, D], [D, 1], [F.relu, None], bn_decay)
    
    def forward(self, X, TE):
        X = self.FC_1(X)
        
        STE = self.embedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        
        for layer in self.attention1:
            X = layer(X, STE_his)
            
        X = self.transformAttention(X, STE_his, STE_pred)
        
        for layer in self.attention2:
            X = layer(X, STE_pred)
        
        X = self.FC_2(X)
        
        del STE, STE_his, STE_pred
        return X