import torch
from torch import nn
import os

from utils.loaders.loaders import load_adj, load_blackbox


device= torch.device('cuda')


class BlackBox(nn.Module):
    def __init__(self, model_dir, num_nodes):
        super(BlackBox, self).__init__()
        self.num_nodes = num_nodes
        _, _, self.adj_mx = load_adj(os.path.join(os.getcwd(), model_dir, 'adj_mx.pkl'))
        self.edge_index = [[], []]
        self.edge_weight = []

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(self.adj_mx.item((i, j)))

        self.edge_index = torch.tensor(self.edge_index)
        self.edge_weight = torch.tensor(self.edge_weight)

        self.model = load_blackbox(os.path.join(os.getcwd(), model_dir, 'model_params.pth')).to(device)
        
        # freeze training
        # for parameter in self.model.parameters():
        #     parameter.requires_grad = False
        
    def forward(self, X):
        return self.model(X, self.edge_index, self.edge_weight)