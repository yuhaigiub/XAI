import torch
import torch.nn.functional as F
from torch import nn
from layers.gman.FC import FC

class SpatialAttention(nn.Module):
    def __init__(self, device, K, d, bn_decay):
        super(SpatialAttention, self).__init__()
        
        self.device = device
        D = K * d
        self.d = d
        self.K = K
        
        self.FC_q = FC(self.device, input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(self.device, input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(self.device, input_dims=2 * D, units=D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(self.device, input_dims=D, units=D, activations=F.relu, bn_decay=bn_decay)
    
    def forward(self, X, SE):
        batch_size = X.shape[0]
        X = torch.cat((X, SE.to(self.device)), dim=-1)
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        
        query_split = torch.split(query, self.d, dim=-1)
        key_split = torch.split(key, self.d, dim=-1)
        value_split = torch.split(value, self.d, dim=-1)
        
        query = torch.cat(query_split, dim=0)
        key = torch.cat(key_split, dim=0)
        value = torch.cat(value_split, dim=0)
        
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        
        X = self.FC(X)
        
        del batch_size, query, key, value
        del query_split, key_split, value_split, attention
        return X