import torch
import torch.nn.functional as F
from torch import nn

from layers.gman.FC import FC

class GatedFusion(nn.Module):
    def __init__(self, device, D, bn_decay):
        super(GatedFusion, self).__init__()
        self.FC_xs = FC(device, D, D, activations=None, bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(device, D, D, activations=None, bn_decay=bn_decay, use_bias=False)
        self.FC_h = FC(device, [D, D], [D, D], activations=[F.relu, None], bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        
        del XS, XT, z
        return H
