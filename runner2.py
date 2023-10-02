'''
Description:
Model: models/gman/model1.py

Task: Traffic Forecasting
'''

import os
import numpy as np

import torch
import torch.optim as optim

# import model.gman.model1 as gman
import model.patchTST.model1 as patchTST
from utils import metrics
from utils.loaders.dataset import load_dataset
from utils.loaders.loaders import load_SE

from model.blackBox.graphWavenet.wrapper import BlackBox
from utils.pertubate import Gaussian

fix_seed = 42
# random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

class Args():
    def __init__(self):
        self.mode = "Train"
        self.device = 'cuda:0'
        # loaders
        self.data_dir = 'data/METR-LA'
        self.blackbox_data_dir = 'data/pretrained/GraphWavenet'
        self.adj_file = 'data/adj_mx.pkl'
        self.log_dir = 'log'
        self.save_dir = 'saved_models'
        # trainer
        self.batch_size = 13
        self.epochs = 1
        # optimizer
        self.weight_decay = 0.0001
        self.learning_rate = 0.001
        self.bn_decay = 0.1
        self.num_nodes = 207
        self.threshold = 0.5
        
        # patchTST
        self.seq_len = 12
        self.pred_len = 12
        self.patch_len = 1
        self.stride = 1

args = Args()
device = torch.device(args.device)

data_loader = load_dataset(args.data_dir, args.batch_size)
SE = load_SE(args.data_dir)

scaler = data_loader['scaler']

# model = gman.Model(device, SE, args.bn_decay)
model = patchTST.Model(device=device,
                       context_window=args.seq_len, 
                       target_window=args.pred_len, 
                       patch_len=args.patch_len, 
                       stride=args.stride)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_fn = metrics.masked_rmse

log_file = open(args.log_dir + '/loss_train_log.txt', 'w')
os.makedirs(args.save_dir, exist_ok=True)

train_loss_per_epoch= []
val_loss_per_epoch= []
print ("Start training.....")

number_of_params = sum(p.numel() for p in model.parameters())
number_of_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of parameters:")
print('Total:', number_of_params)
print('Trainable:', number_of_trainable_params)

for i in range(1, args.epochs + 1):
    train_loss = []
    train_mape = []
    train_mae = []
    
    data_loader['train_loader'].shuffle()
    
    for j, (x, y, t) in enumerate(data_loader['train_loader'].get_iterator()):
        model.train()  # tell the model that you are training
        optimizer.zero_grad()  # set the gradient to zero
        
        # gman
        # trainX = torch.FloatTensor(np.expand_dims(x, axis=-1)).to(device)
        # trainY = torch.FloatTensor(np.expand_dims(y, axis=-1)).to(device)
        # trainTE = torch.FloatTensor(t).to(device)
        # output = model(trainX, trainTE)
        
        # PatchTST
        trainX = torch.FloatTensor(x).to(device)
        trainY = torch.FloatTensor(y).to(device)
        output = model(trainX)
        
        predY = scaler.inverse_transform(output)
        
        loss = loss_fn(predY, trainY, 0.0)
        
        mape = metrics.masked_mape(predY, trainY, 0.0)
        mae = metrics.masked_mae(predY, trainY, 0.0)
        
        train_loss.append(loss)
        train_mape.append(mape)
        train_mae.append(mae)
        
        if j % 100 == 0: 
            log = 'Iter: {:03d}, Train Loss: {:.4f}' 
            print(log.format(j, train_loss[-1]), flush=True)
        
        loss.backward()
        optimizer.step()
        
    epoch_train_loss = torch.mean(torch.tensor(train_loss))
    epoch_train_mape = torch.mean(torch.tensor(train_mape))
    epoch_train_mae = torch.mean(torch.tensor(train_mae))
    
    log_file.write(f'Epoch {i}, {args.mode} Loss: {epoch_train_loss:.4f}, {args.mode} MAPE: {epoch_train_mape:.4f}, {args.mode} MAE: {epoch_train_mae:.4f} \n')
    log_file.flush()
    
    print(f'Epoch {i}, Training Loss: {epoch_train_loss:.4f}')
    
    model.train(mode=False) # tell the model that you are not training
    if i % 2 == 0:
        print(f'epoch {i} trained')
        print(f'Model saved at epochs {i}')
        model_filename = os.path.join(args.save_dir, f'G_T_model_{i}.pth')
        torch.save(model, model_filename)

log_file.close()
print ("Done......")