import torch
from torch import nn
import torch.nn.functional as F
from layers.gman.FC import FC

class TransformAttention(nn.Module):
    def __init__(self, device, K, d, bn_decay):
        super(TransformAttention, self).__init__()
        self.device = device
        self.K = K
        self.d = d
        D = K * d
        
        self.FC_q = FC(device, D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(device, D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(device, D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(device, D, D, activations=F.relu, bn_decay=bn_decay)
        
    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        
        X = self.FC(X)
        
        del query, key, value, attention
        return X