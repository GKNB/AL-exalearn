#!/usr/bin/env python

import io, os, sys
import h5py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error

import argparse

#----------------------Parser settings---------------------------

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--batch_size',     type=int,   default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs',         type=int,   default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr',             type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed',           type=int,   default=42,
                    help='random seed (default: 42)')
parser.add_argument('--device',         default='gpu', choices=['cpu', 'gpu'],
                    help='Whether this is running on cpu or gpu')
parser.add_argument('--cont_train',     default='no', choices=['no', 'yes'],
                    help='In phase 1, continue training or not')
parser.add_argument('--num_workers',    type=int,   default=1, 
                    help='set the number of op workers. only work for gpu')
parser.add_argument('--phase', type=int,            default=0,
                    help='the current phase idx, if non-zero, depending on cont_train, load ckpt')
parser.add_argument('--ckpt_dir',                  default='./',
                    help='the directory where save and load ckpt of model and optimizer')
parser.add_argument('--output_filename', default='test_output.hdf5',
                    help='the filename of input data, including train/test sample')
parser.add_argument('--global_test_filename', default='global_test.hdf5',
                        help='filename of global test set used for all training')

args = parser.parse_args()
args.cuda = ( args.device.find("gpu")!=-1 and torch.cuda.is_available() )
if args.cuda:
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)


#------------------------Model----------------------------
class FullModel(torch.nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 8)
        self.fc5 = torch.nn.Linear(8, 3)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x


#-----------------------------Loading data--------------------------------

fname = "training_input_phase_{}.hdf5".format(args.phase)
with h5py.File(fname, 'r') as f:
    X_test  = np.float32(f["X_test"][:])
    y_test  = np.float32(f["y_test"][:])
    X_train = np.float32(f["X_train"][:])
    y_train = np.float32(f["y_train"][:])

fname = args.global_test_filename
with h5py.File(fname, 'r') as f:
    X_global_test = np.float32(f["X_global_test"][:])
    y_global_test = np.float32(f["y_global_test"][:])

X_train_torch = torch.from_numpy(X_train)
y_train_torch = torch.from_numpy(y_train).reshape(-1,y_train.shape[1])
X_train_torch = FloatTensor(X_train_torch)
y_train_torch = FloatTensor(y_train_torch)

X_test_torch = torch.from_numpy(X_test)
y_test_torch = torch.from_numpy(y_test).reshape(-1,y_test.shape[1])
X_test_torch = FloatTensor(X_test_torch)
y_test_torch = FloatTensor(y_test_torch)

X_global_test_torch = torch.from_numpy(X_global_test)
y_global_test_torch = torch.from_numpy(y_global_test).reshape(-1,y_global_test.shape[1])
X_global_test_torch = FloatTensor(X_global_test_torch)
y_global_test_torch = FloatTensor(y_global_test_torch)

print("X_train: {}, y_train: {}, X_test: {}, y_test: {}, X_global_test: {}, y_global_test: {}".format(X_train_torch.shape, y_train_torch.shape, X_test_torch.shape, y_test_torch.shape, X_global_test_torch.shape, y_global_test_torch.shape))


kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_dataset = torch.utils.data.TensorDataset(X_test_torch,y_test_torch)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

global_test_dataset = torch.utils.data.TensorDataset(X_global_test_torch,y_global_test_torch)
global_test_loader = torch.utils.data.DataLoader(
        global_test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

#----------------------------setup model---------------------------------

model = FullModel()
if args.phase > 0 and args.cont_train == 'yes':
    ckpt = torch.load(args.ckpt_dir + "/ckpt_phase{}.pt".format(args.phase-1))
    model.load_state_dict(ckpt['model'])

if args.cuda:
    model.cuda()

#---------------------------setup optimizer------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
if args.phase > 0 and args.cont_train == 'yes':
    optimizer.load_state_dict(ckpt['optimizer'])

final_loss = torch.nn.MSELoss(reduction='none')
criterion = torch.nn.MSELoss()

#------------------------------start training----------------------------------

def train(epoch):
    model.train()
    
    running_loss  = torch.tensor(0.0)
    if args.device == "gpu":
        running_loss = running_loss.cuda()
    print("len of train_loader = {}".format(len(train_loader)))

    for batch_idx, current_batch in enumerate(train_loader):     
        if args.cuda:
            inp, current_batch_y = current_batch[0].cuda(), current_batch[1].cuda()
        else:
            inp, current_batch_y = current_batch[0],        current_batch[1]

        optimizer.zero_grad()
        output = model(inp)
        gndtruth = current_batch_y

        loss = criterion(output, gndtruth)
        loss.backward()
        optimizer.step()
        running_loss  += loss.item()

    running_loss  = running_loss  / len(train_loader)
    print("Average loss: {:15.8f}".format(running_loss))

def test():
    model.eval()
    
    test_loss  = torch.tensor(0.0)
    for inp, current_batch_y in global_test_loader:
        if args.cuda:
            inp, current_batch_y = inp.cuda(), current_batch_y.cuda()

        output = model(inp)
        test_loss += criterion(output, current_batch_y).item()

    test_loss /= len(global_test_loader)
    print("Average test loss: {:15.8f}".format(test_loss))

def final_test():
    model.eval()

    test_output = np.empty((0, 1))
    for inp, current_batch_y in test_loader:
        if args.cuda:
            inp, current_batch_y = inp.cuda(), current_batch_y.cuda()

        output = model(inp)
        test_loss = final_loss(output, current_batch_y).cpu().detach().numpy()
        test_loss = np.mean(test_loss, axis=1, keepdims=True)
        test_output = np.concatenate((test_output, test_loss), axis=0)

    print("Average = {}".format(np.mean(test_output)))
    test_output = np.concatenate((y_test, test_output), axis=1)
    print(test_output)
    print(test_output.shape)

    fname = args.output_filename
    with h5py.File(fname, 'w') as f:
        f.create_dataset("final_test_output", data = test_output)


for epoch in range(1, args.epochs + 1):
    e_start = time.time()
    train(epoch)
    test()
final_test()

state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
torch.save(state, args.ckpt_dir + "/ckpt_phase{}.pt".format(args.phase))
