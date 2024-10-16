import torch
import torch.nn as nn

class MHAttention(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100, window_len=20, n_heads=10):
        super(MHAttention, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.window_len = window_len
        self.n_heads = n_heads
        self.attention_heads_1 = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.attention_heads_2 = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, x, is_causal=True):
        mask = None
        if is_causal:
            mask = torch.tril(torch.ones(self.window_len, self.window_len, device=x.device)).bool()
            mask = ~mask
        x = self.embedding(x)
        x, w = self.attention_heads_1(x, x, x, attn_mask=mask, is_causal=True)
        x, w = self.attention_heads_2(x, x, x, attn_mask=mask, is_causal=True)
        x = self.fc(x)
        return x

def generate(model, vocab, text, n=100, sample=True, device='cpu'):
    response = [vocab.index(c) for c in text]
    if len(text) < model.window_len:
        text = ' ' * (model.window_len - len(text)) + text
    x_text = torch.tensor([vocab.index(c) for c in text[-model.window_len:]]).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            y = model(x_text)
            y = torch.softmax(y, dim=-1)[:, -1, :]
            if sample:
                y = torch.multinomial(y, num_samples=1)
            else:
                y = y.argmax(dim=-1).unsqueeze(0)
            response.append(y.item())
            x_text = torch.cat([x_text, y], dim=1)[:, -model.window_len:]
    model.train()
    return ''.join([vocab[idx] for idx in response])



if __name__ == "__main__":
    from mh_data import MHDataset
    ds = MHDataset('HP1.txt', train=True, window_size=20, step_size=1)
    X = ds[0][0].unsqueeze(0)
    print(X.shape)
    model = MHAttention(input_dim=500, vocab_size=ds.vocab_size)
    out = model(X)
    print(out.shape)
    text = generate(model, ds.vocab, 'Harry Potter', sample=True)
    print(text)