'''
Description:
Model: models/gman/model1.py

Task: Traffic Forecasting
'''

import os
import numpy as np

import torch
import torch.optim as optim

import model.gman.model1 as gman
from utils import metrics
from utils.loaders.dataset import load_dataset
from utils.loaders.loaders import load_SE

fix_seed = 42
# random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

class Args():
    def __init__(self):
        self.mode = "Val"
        self.device = 'cuda:0'
        # loaders
        self.data_dir = 'data/METR-LA'
        self.blackbox_data_dir = 'data/pretrained/GraphWaveNet'
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
loss_fn = metrics.masked_rmse

log_file = open(args.log_dir + '/loss_val_log.txt', 'w')
os.makedirs(args.save_dir, exist_ok=True)
    
print ("Start evaluating.....")
max_pth =10
for itea in range(max_pth): 
    model = torch.load(args.save_dir + f'/G_T_model_{itea+1}.pth')
    for i in range(1, args.epochs + 1):
        val_loss = []
        val_mape = []
        val_mae = []
        
        data_loader['val_loader'].shuffle()
        with torch.no_grad():
            for j, (x, y, t) in enumerate(data_loader['val_loader'].get_iterator()):
                valX = torch.FloatTensor(x).to(device)
                valY = torch.FloatTensor(y).to(device)
                output = model(valX)
                
                predY = scaler.inverse_transform(output)
                
                loss = loss_fn(valY, predY, 0.0)
                
                mape = metrics.masked_mape(valY, predY, 0.0)
                mae = metrics.masked_mae(valY, predY, 0.0)
                
                val_loss.append(loss)
                val_mape.append(mape)
                val_mae.append(mae)
            
            epoch_val_loss = torch.mean(torch.tensor(val_loss))
            epoch_val_mape = torch.mean(torch.tensor(val_mape))
            epoch_val_mae = torch.mean(torch.tensor(val_mae))
            
            log_file.write(f'Epoch {itea + 1}, {args.mode} Loss: {epoch_val_loss:.4f}, {args.mode} MAPE: {epoch_val_mape:.4f}, {args.mode} RMSE: {epoch_val_mae:.4f} \n')
            log_file.flush()
            print(f'Epoch [{itea+1}], {args.mode} Loss: {epoch_val_loss:.4f}, {args.mode} MAPE: {epoch_val_mape:.4f}, {args.mode} MAE: {epoch_val_mae:.4f}')

log_file.close()
print ("Done......")