#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                      Title: Utils to train and evaluate models                      ##
##                                   Date: 2024/10/16                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate(model, ds_val, batch_size=16, num_workers=4):
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in tqdm(dl_val):
            out_val, loss_val = model(X_val, y_val)
            val_loss += loss_val.item()
    return val_loss/len(dl_val)

def train(model, ds_train, ds_val, batch_size=16, epochs=10, eval_steps=100, lr=1e-3, num_workers=4):
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    cnt_updates = 0
    loss_hist = []
    for epoch in range(epochs):
        for X, y in dl_train:
            out, loss = model(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt_updates -=- 1
            loss_hist.append(loss.item())
            if cnt_updates % eval_steps == 0:
                loss_val = evaluate(model, ds_val, batch_size)
                print(f'Epoch: {epoch}, Update: {cnt_updates}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f}')
                model.train() 
    return loss_hist

def encode_text(text, ds):
    return torch.tensor([ds.char2Idx(c) for c in text])

def decode_text(tensor, ds):
    return ''.join([ds.idx2Char(idx) for idx in tensor])