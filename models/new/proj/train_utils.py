import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=1,
                enable_checkpoint_ensemble=True):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    for run in range(3):
        print("split: ",run+1)
        train_loader = torch.utils.data.DataLoader(train[run], batch_size=batch_size, shuffle=True)
        for epoch in range(n_epochs):
            scheduler.step()

            model.train()
            avg_loss_trn = 0.
            #print("\ttrain\t\n")
            for data in tqdm(train_loader, disable=False):
                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = model(*x_batch)    

                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                avg_loss_trn += loss.item() / len(train_loader)
            model.eval()
            run_val = (run+1) if ( (run+1) <3) else 0
            val_loader = torch.utils.data.DataLoader(train[run_val], batch_size=batch_size, shuffle=True)
            #print("\tval\t\n")
            avg_loss_val = 0.0
            for data in tqdm(val_loader,disable=False):
                x_batch = data[:-1]
                y_batch = data[-1]

                y_pred = model(*x_batch)
                loss = loss_fn(y_pred,y_batch)

                avg_loss_val += loss.item()/len(val_loader)
            print("\ttrn loss:\t",avg_loss_trn)
            print("\tval loss: ",avg_loss_val)
            
def train_model_unified(model, train, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=1,
                enable_checkpoint_ensemble=True):

    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epochs):
        scheduler.step()

        model.train()
        avg_loss_trn = 0.
        print("\tepoch:\t",epoch)
        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]

            y_pred = model(*x_batch)    

            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss_trn += loss.item() / len(train_loader)
        model.eval()
        #val_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        #print("\tval\t\n")
        #avg_loss_val = 0.0
        #for data in tqdm(val_loader,disable=False):
        #    x_batch = data[:-1]
        #    y_batch = data[-1]

        #    y_pred = model(*x_batch)
        #    loss = loss_fn(y_pred,y_batch)

        #    avg_loss_val += loss.item()/len(val_loader)
        print("\ttrn loss:\t",avg_loss_trn)
        #print("\tval loss: ",avg_loss_val)

def get_preds(model, ds, batch_size=512):
    
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_preds = np.zeros((len(ds),7))
    for i, x_batch in tqdm(enumerate(ds_loader)):
        y_pred = sigmoid(model(*x_batch[:-1]).detach().cpu().numpy())

        test_preds[i * batch_size:(i+1) * batch_size] = y_pred
        
    return test_preds

def get_preds_test(model, ds, batch_size=512):
    
    ds_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_preds = np.zeros((len(ds),7))
    for i, x_batch in tqdm(enumerate(ds_loader)):
        y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

        test_preds[i * batch_size:(i+1) * batch_size] = y_pred
        
    return test_preds