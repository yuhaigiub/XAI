import os
import torch

def load_SE(dataset_dir):
    with open(os.path.join(dataset_dir, "SE.txt"), mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        num_vertex, dims = int(temp[0]), int(temp[1])
        print("Num_vertex: ", num_vertex)
        print("Dimentions: ", dims)
        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            index = int(temp[0])
            SE[index] = torch.tensor([float(ch) for ch in temp[1:]])
    
    return SE