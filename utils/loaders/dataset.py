import torch
import pandas as pd
import os

from utils.loaders.StandardScaler import StandardScaler
from utils.loaders.DataLoader import DataLoader

def makeSequence(data, seq_size=12):
    time_step, dims = data.shape
    num_sample = time_step - seq_size + 1
    x = torch.zeros(num_sample, seq_size, dims)
    for i in range(num_sample):
        x[i] = data[i: i + seq_size]
    return x

def makeSequenceXY(data, seq_size=12, pred_size=None):
    if pred_size == None:
        num_his, num_pred = seq_size, seq_size
    else:
        num_his, num_pred = seq_size, pred_size
        
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

def load_dataset(dataset_dir, batch_size):
    data = {}
    df = pd.read_hdf(os.path.join(dataset_dir, 'data.h5'))
    
    # infer dataset sizes
    data_sizes = df.shape[0]
    train_sizes = round(0.7 * data_sizes)
    test_sizes = round(0.2 * data_sizes)
    val_sizes = data_sizes - train_sizes - test_sizes
    
    # -----------------------------------------------------------
    train_data = torch.tensor(df[:train_sizes].to_numpy())
    val_data = torch.tensor(df[train_sizes:train_sizes + val_sizes].to_numpy())
    test_data = torch.tensor(df[-test_sizes:].to_numpy())
    
    #
    data['x_train'], data['y_train'] = makeSequenceXY(train_data)
    data['x_val'], data['y_val'] = makeSequenceXY(val_data)
    data['x_test'], data['y_test'] = makeSequenceXY(test_data)
    
    # -----------------------------------------------------------
    time = pd.DatetimeIndex(df.index)
    dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // (5 * 60)
    timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
    time = torch.cat((dayofweek, timeofday), -1)
    
    train_time = time[:train_sizes]
    val_time = time[train_sizes:train_sizes + val_sizes]
    test_time = time[-test_sizes:]
    #
    data['t_train'] = makeSequence(train_time)
    data['t_val'] = makeSequence(val_time)
    data['t_test'] = makeSequence(test_time)
    
    mean = data['x_train'][..., 0].mean()
    std = data['x_train'][..., 0].std()
    
    scaler = StandardScaler(mean, std)
    data['scaler'] = scaler
    
    for category in ['train', 'test', 'val']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data[category + '_loader'] = DataLoader(xs=data['x_' + category],
                                                ys=data['y_' + category],
                                                ts=data['t_' + category],
                                                batch_size=batch_size)
    
    return data