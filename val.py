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

np.random.seed(42)

class Args():
    def __init__(self):
        self.mode = "Val"
        self.device = 'cuda:0'
        # loaders
        self.data_dir = 'data/METR-LA'
        self.blackbox_file = 'data/models/graphwavenet.pth'
        self.adj_file = 'data/adj_mx.pkl'
        self.log_dir = 'log'
        self.save_dir = 'saved_models'
        # trainer
        self.batch_size = 16
        self.epochs = 10
        # optimizer
        self.weight_decay = 0.0001
        self.learning_rate = 0.001
        self.bn_decay = 0.1
        self.num_nodes = 207
        self.threshold = 0.5

args = Args()
device = torch.device(args.device)

data_loader = load_dataset(args.data_dir, args.batch_size)
SE = load_SE(args.data_dir)

scaler = data_loader['scaler']

model = torch.load(args.save_dir + "/G_T_model_10.pth")

# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
loss_fn = metrics.masked_rmse

log_file = open(args.log_dir + '/loss_val_log.txt', 'w')
os.makedirs(args.save_dir, exist_ok=True)

print ("Start training.....")

for i in range(1, args.epochs + 1):
    train_loss = []
    train_mape = []
    train_mae = []
    
    data_loader['val_loader'].shuffle()
    with torch.no_grad():
        for j, (x, y, t) in enumerate(data_loader['val_loader'].get_iterator()):
            trainX = torch.FloatTensor(np.expand_dims(x, axis=-1)).to(device)
            trainY = torch.FloatTensor(np.expand_dims(y, axis=-1)).to(device)
            trainTE = torch.FloatTensor(t).to(device)
            
            #model.eval()  # tell the model that you are training
            #optimizer.zero_grad()  
            
            output = model(trainX, SE)
            predY = scaler.inverse_transform(output)
            
            loss = loss_fn(predY, trainY, 0.0)
            
            mape = metrics.masked_mape(predY, trainY, 0.0)
            mae = metrics.masked_mae(predY, trainY, 0.0)
            
            train_loss.append(loss)
            train_mape.append(mape)
            train_mae.append(mae)
            
            # if j % 100 == 0: 
            #     log = 'Iter: {:03d}, Val Loss: {:.4f}' 
            #     print(log.format(j, train_loss[-1]), flush=True)
            
            # loss.backward()
            # optimizer.step()
            
        epoch_train_loss = torch.mean(torch.tensor(train_loss))
        epoch_train_mape = torch.mean(torch.tensor(train_mape))
        epoch_train_mae = torch.mean(torch.tensor(train_mae))
        
        log_file.write(f'Epoch {i}, {args.mode} Loss: {epoch_train_loss:.4f}, {args.mode} MAPE: {epoch_train_mape:.4f}, {args.mode} RMSE: {epoch_train_mae:.4f} \n')
        log_file.flush()
        
        print(f'Epoch [{i}], Val Loss: {epoch_train_loss:.4f}, Val MAPE: {epoch_train_mape:.4f}, Val RMSE: {epoch_train_mae:.4f}')
        
        #model.eval(mode=False) # tell the model that you are not training
    

log_file.close()
print ("Done......")