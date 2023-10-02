from torch import nn

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)