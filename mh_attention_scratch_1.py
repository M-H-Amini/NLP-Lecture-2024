#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                Title: Self-Attention                                ##
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
from mh_utils import train, encode_text, decode_text, generate_text


class MHAttention(nn.Module):
    def __init__(self, input_dim=512, head_size=512, vocab_size=100):
        super(MHAttention, self).__init__()
        self.input_dim = input_dim
        self.head_size = head_size
        self.q_layer = nn.Linear(input_dim, head_size, bias=False)
        self.k_layer = nn.Linear(input_dim, head_size, bias=False)
        self.v_layer = nn.Linear(input_dim, head_size, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_layer(x)  ##  (B, T, H)
        k = self.k_layer(x)  ##  (B, T, H)
        v = self.v_layer(x)  ##  (B, T, H)
        w = q @ k.transpose(-1, -2)  ##  (B, T, H) @ (B, H, T) = (B, T, T)
        w = w * self.head_size**-.5
        w = torch.tril(w)  ##  (T, T)
        w = w.masked_fill(w == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        x = w @ v  ##  (T, T) @ (B, T, H) = (B, T, H)
        return x

class MHLLM(nn.Module):
    def __init__(self, input_dim=512, head_size=512, vocab_size=100, context_len=5):
        super(MHLLM, self).__init__()
        self.input_dim = input_dim
        self.head_size = head_size
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.t_embedding = nn.Embedding(vocab_size, input_dim)
        self.p_embedding = nn.Embedding(context_len, input_dim)
        self.attention = MHAttention(input_dim, head_size, vocab_size)
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, x, y=None):
        B, T = x.shape
        t_embed = self.t_embedding(x)  ##  (B, T) -> (B, T, C)
        p_embed = self.p_embedding(torch.arange(T, device=x.device))  ##  (T,) -> (T, C)
        x = t_embed + p_embed
        x = self.attention(x)  ##  (B, T, H)
        x = self.fc(x)  ##  (B, T, H) -> (B, T, V)
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

    def save(self, filepath):
        checkpoint = {
            'input_dim': self.input_dim,
            'head_size': self.head_size,
            'vocab_size': self.vocab_size,
            'context_len': self.context_len,
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        if device is None:
            device = torch.device('cpu')
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            head_size=checkpoint['head_size'],
            vocab_size=checkpoint['vocab_size'],
            context_len=checkpoint['context_len']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)
    
if __name__ == "__main__":
    context_len = 30
    input_dim = 512
    head_size = 512
    epochs = 10
    batch_size = 512
    eval_steps = 100
    lr = 1e-4
    n_generated = 100
    model_name = 'attn_self'
    init_text = 'Harry Potter'
    books = ['HP1.txt', 'HP2.txt', 'HP3.txt']

    ds_train = MHDataset(books, train=True, window_size=context_len, step_size=5)
    ds_val = MHDataset(books, train=False, window_size=context_len, step_size=context_len)
    
    model = MHLLM(input_dim=input_dim, head_size=head_size, vocab_size=ds_train.vocab_size, context_len=context_len)
    print('Before training:')
    print(generate_text(model, init_text, ds_val, n=n_generated, sample=True))

    train(model, ds_train, ds_val, batch_size=batch_size, epochs=epochs, eval_steps=eval_steps, lr=lr, num_workers=4, tensorboard=True, model_name=model_name)

    print('After training:')
    print(generate_text(model, init_text, ds_val, n=n_generated, sample=True))