from torch import nn
from layers.gman.Convolution import Convolution

class FC(nn.Module):
    def __init__(self, device, input_dims, units, activations, bn_decay, use_bias=True, dropout=0.):
        super(FC, self).__init__()
        
        self.device = device
        
        # one layer FC
        if isinstance(units, int):
            input_dims = [input_dims]
            units = [units]
            activations = [activations]
            
        # multiple layers FC
        elif isinstance(units, tuple):
            input_dims = list(input_dims)
            units = list(units)
            activations = list(activations)
        
        assert type(units) == list
        
        convs = []
        for input_dim, num_unit, activation in zip(input_dims, units, activations):
            layer = Convolution(device,
                                input_dim, 
                                num_unit, 
                                kernel_size=[1, 1], 
                                stride=[1, 1], 
                                padding="VALID", 
                                use_bias=use_bias, 
                                activation=activation, 
                                bn_decay=bn_decay,
                                dropout=dropout)
            convs.append(layer)
        
        self.convs = nn.ModuleList(convs)

    def forward(self, X):
        for conv in self.convs:
            X = conv(X)
        
        return X