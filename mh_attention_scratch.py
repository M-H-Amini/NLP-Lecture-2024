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
        x = w @ x  ##  (T, T) @ (B, T, C) = (B, T, C)
        return x

class MHLLM(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100, context_len=5):
        super(MHLLM, self).__init__()
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.context_len = context_len
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
    
    def generate(self, x_init, n=100, sample=True):
        _, T = x_init.shape
        x = x_init  ##  (1, T)
        resp = x.squeeze(0).tolist()
        if T > self.context_len:
            x = x[:, -self.context_len:]  ##  (1, T) -> (1, context_len)
        for _ in range(n):  
            xx, __ = self.forward(x)
            xx = xx[0, -1, :]  ##  (1, T) -> (V,)  
            if sample:
                xx = torch.multinomial(F.softmax(xx, dim=-1), 1)  ##  (V,) -> (1,)
            else:
                xx = xx.argmax().unsqueeze(0)  ##  (V,) -> (1,)
            x = torch.cat([x, xx.unsqueeze(0)], dim=1)[:, -self.context_len:]
            resp.append(xx.item())
        return torch.tensor(resp)

    
if __name__ == "__main__":
    ds = MHDataset('HP1.txt', train=True, window_size=5, step_size=1)
    X = ds[0][0].unsqueeze(0)
    y = ds[0][1].unsqueeze(0)
    model = MHLLM(input_dim=3, vocab_size=ds.vocab_size)
    out, loss = model(X, y)

    init_text = 'Harry Potter'
    X = encode_text(init_text, ds).unsqueeze(0)
    Xgen = model.generate(X, n=100, sample=True)
    print(decode_text(Xgen, ds))

    ds_train = MHDataset('HP1.txt', train=True, window_size=15, step_size=1)
    ds_val = MHDataset('HP1.txt', train=False, window_size=15, step_size=1)
    train(model, ds_train, ds_val, batch_size=1024, epochs=10, eval_steps=100, lr=1e-3, num_workers=4)
    init_text = 'Harry Potter'
    X = encode_text(init_text, ds).unsqueeze(0)
    Xgen = model.generate(X, n=100, sample=True)
    print(decode_text(Xgen, ds))
