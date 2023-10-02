import torch
import torch.nn.functional as F
from torch import nn

from layers.gman.FC import FC

class TemporalAttention(nn.Module):
    def __init__(self, device, K, d, bn_decay, mask=True):
        super(TemporalAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.mask = mask
        self.FC_q = FC(device, 2 * D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_k = FC(device, 2 * D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC_v = FC(device, 2 * D, D, activations=F.relu, bn_decay=bn_decay)
        self.FC = FC(device, D, D, activations=F.relu, bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        X = torch.cat((X, STE), dim=-1)

        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        
        # mask attention score
        if self.mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
            
        # softmax
        attention = F.softmax(attention, dim=-1)
        
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        
        X = self.FC(X)
        
        del query, key, value, attention
        return X