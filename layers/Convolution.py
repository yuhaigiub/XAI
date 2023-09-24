import torch
from torch import nn
import torch.nn.functional as F
import math

class Convolution(nn.Module):
    def __init__(self,
                 device,
                 input_dim, 
                 output_dim, 
                 kernel_size, 
                 stride=(1, 1), 
                 padding='SAME', 
                 use_bias=True, 
                 activation=F.relu, 
                 bn_decay=None):
        super(Convolution, self).__init__()
        self.device = device
        self.activation = activation
        
        if padding == 'SAME':
            padding_value = math.ceil(kernel_size / 2)
            self.padding_size = [padding_value, padding_value]
        else:
            self.padding_size = [0, 0]
        
        self.conv = nn.Conv2d(input_dim, output_dim , kernel_size, stride=stride, padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dim, momentum=bn_decay)
        
        self.conv.to(self.device)
        self.batch_norm.to(self.device)
        
        torch.nn.init.xavier_normal_(self.conv.weight)
        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)    

    def forward(self, X):
        X = X.permute(0, 3, 2, 1)
        X = F.pad(X, (self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]))
        
        X = self.conv(X)
        X = self.batch_norm(X)
        if self.activation != None:
            X = self.activation(X)
        
        X = X.permute(0, 3, 2, 1)
        
        return X
