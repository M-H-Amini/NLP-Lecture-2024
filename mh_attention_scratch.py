#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                           Title: Attention with averaging                           ##
##                                   Date: 2024/10/16                                  ##
##                                                                                     ##
#########################################################################################

##  Description: 

from mh_data import MHDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class MHAttention(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100):
        super(MHAttention, self).__init__()
        self.input_dim = input_dim
        
        
    def forward(self, x):
        B, T, C = x.shape
        w = torch.tril(torch.ones(T, T, device=x.device))  ##  (T, T)
        w = w.masked_fill(w == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        x = w @ x ##  (T, T) @ (B, T, C) = (B, T, C)
        return x

class MHLLM(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100):
        super(MHLLM, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.attention = MHAttention(input_dim, vocab_size)
        self.fc = nn.Linear(input_dim, vocab_size)
        
    def forward(self, x, y=None):
        B, T = x.shape
        x = self.embedding(x)  ##  (B, T) -> (B, T, C)
        x = self.attention(x)  ##  (B, T, C)
        x = self.fc(x)  ##  (B, T, C) -> (B, T, V)
        loss = None
        if y is not None:
            loss = F.cross_entropy(x.view(B*T, -1), y.view(-1))
        return x, loss

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
        
    
if __name__ == "__main__":
    ds = MHDataset('HP1.txt', train=True, window_size=5, step_size=1)
    X = ds[0][0].unsqueeze(0)
    y = ds[0][1].unsqueeze(0)
    model = MHLLM(input_dim=3, vocab_size=ds.vocab_size)
    out, loss = model(X, y)
    print(out.shape)
    print(loss)
    input('Press Enter to continue...')
    ds_train = MHDataset('HP1.txt', train=True, window_size=15, step_size=1)
    ds_val = MHDataset('HP1.txt', train=False, window_size=15, step_size=1)
    train(model, ds_train, ds_val, batch_size=512, epochs=10, eval_steps=100, lr=1e-3, num_workers=4)