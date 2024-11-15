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
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, ds_val, batch_size=16, num_workers=4):
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in tqdm(dl_val):
            out_val, loss_val = model(X_val, y_val)
            val_loss += loss_val.item()
    return val_loss/len(dl_val)

def train(model, ds_train, ds_val, batch_size=16, epochs=10, eval_steps=100, lr=1e-3, num_workers=4, tensorboard=False, model_name='model'):
    os.makedirs('models', exist_ok=True)
    log_dir = 'runs'
    if tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
        init_text = 'Harry and Ron '
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    cnt_updates = 0
    best_loss = float('inf')
    for epoch in range(epochs):
        for X, y in dl_train:
            out, loss = model(X, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt_updates -=- 1
            tensorboard and writer.add_scalar('Loss/train', loss.item(), cnt_updates)
            if cnt_updates % eval_steps == 0:
                loss_val = evaluate(model, ds_val, batch_size)
                tensorboard and (text:=generate_text(model, init_text, ds_val, n=500, sample=True))
                tensorboard and writer.add_text('Text/val', text, cnt_updates)
                tensorboard and writer.add_scalar('Loss/val', loss_val, cnt_updates)
                if loss_val < best_loss:
                    best_loss = loss_val
                    model.save(os.path.join('models', model_name + '.pth'))
                print(f'Epoch: {epoch}, Update: {cnt_updates}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f}')
                model.train() 
    return model

def encode_text(text, ds):
    return torch.tensor([ds.char2Idx(c) for c in text])

def decode_text(tensor, ds):
    return ''.join([ds.idx2Char(idx) for idx in tensor])

def generate_text(model, init_text, ds, n=100, sample=True):
    X = encode_text(init_text, ds).unsqueeze(0)
    X_gen = model.generate(X, n=n, sample=True)
    return decode_text(X_gen, ds)
