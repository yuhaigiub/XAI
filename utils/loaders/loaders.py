import os
import pickle
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


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_blackbox(filepath):
    return torch.load(filepath)
