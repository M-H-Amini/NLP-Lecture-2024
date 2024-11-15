#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                                   Title: No Context                                 ##
##                                   Date: 2024/10/18                                  ##
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

class MHLLM(nn.Module):
    def __init__(self, vocab_size=100):
        super(MHLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 1)
        
    def forward(self, x, y=None):
        B, T = x.shape  ##  x is redundant here, but we keep it for compatibility with other models
        x = self.embedding(torch.tensor([[c for c in range(self.vocab_size)] for _ in range(B)]).to(x.device))[:, :, 0]  ##  (B, V)
        loss = None
        if y is not None:
            loss = F.cross_entropy(x, y[:, -1])
        return x, loss
    
    def generate(self, x_init, n=100, sample=True):
        _, T = x_init.shape
        x = x_init  ##  (1, T)
        resp = x.squeeze(0).tolist()
        for _ in range(n):  
            xx, __ = self.forward(x)  ##  (1, T) -> (1, V)
            xx = xx[0]  ##  (1, V) -> (V,)
            if sample:
                xx = torch.multinomial(F.softmax(xx, dim=-1), 1)  ##  (V,) -> (1,)
            else:
                xx = xx.argmax().unsqueeze(0)  ##  (V,) -> (1,)
            x = torch.cat([x, xx.unsqueeze(0)], dim=1)
            resp.append(xx.item())
        return torch.tensor(resp)
    
    def save(self, filepath):
        checkpoint = {
            'vocab_size': self.vocab_size,
            'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        if device is None:
            device = torch.device('cpu')
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)


    
if __name__ == "__main__":
    context_len = 30
    input_dim = 512
    n_layers = 6
    epochs = 10
    batch_size = 512
    eval_steps = 100
    lr = 1e-2
    n_generated = 100
    model_name = 'attn_no_context'
    init_text = 'Harry Potter'
    books = ['HP1.txt', 'HP2.txt', 'HP3.txt']

    ds_train = MHDataset(books, train=True, window_size=context_len, step_size=5)
    ds_val = MHDataset(books, train=False, window_size=context_len, step_size=context_len)
    
    model = MHLLM(vocab_size=ds_train.vocab_size)
    print('Before training:')
    print(generate_text(model, init_text, ds_val, n=n_generated, sample=True))

    train(model, ds_train, ds_val, batch_size=batch_size, epochs=epochs, eval_steps=eval_steps, lr=lr, num_workers=4, tensorboard=True, model_name=model_name)

    print('After training:')
    print(generate_text(model, init_text, ds_val, n=n_generated, sample=True))