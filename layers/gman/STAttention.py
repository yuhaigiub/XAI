from torch import nn
import torch

from layers.gman.GatedFusion import GatedFusion
from layers.gman.SpatialAttention import SpatialAttention
from layers.gman.TemporalAttention import TemporalAttention

class STAttention(nn.Module):
    def __init__(self, device, K, d, bn_decay, mask=False):
        super(STAttention, self).__init__()
        self.spatialAttention = SpatialAttention(device, K, d, bn_decay)
        self.temporalAttention = TemporalAttention(device, K, d, bn_decay, mask=mask)
        self.gatedFusion = GatedFusion(device, K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        
        del HS, HT
        return torch.add(X, H)