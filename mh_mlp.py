import torch
import torch.nn as nn

class MHMLP(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100, n_hidden_1=50, n_hidden_2=15, dropout=0.4):
        super(MHMLP, self).__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.fc1 = nn.Linear(input_dim, n_hidden_1)
        self.droput_1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.droput_2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(n_hidden_2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        x = torch.relu(self.fc1(x))
        x = self.droput_1(x)
        x = torch.relu(self.fc2(x))
        x = self.droput_2(x)
        x = self.fc3(x)
        return x
    
def generate(model, vocab, text, n=100, sample=True, device='cpu'):
    x_text = torch.tensor([vocab.index(c) for c in text]).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            y = model(x_text)
            y = torch.softmax(y, dim=-1)
            if sample:
                y = torch.multinomial(y, num_samples=1)
            else:
                y = y.argmax(dim=-1).unsqueeze(0)
            x_text = torch.cat([x_text, y], dim=1)
    model.train()
    return ''.join([vocab[idx] for idx in x_text.squeeze(0).tolist()])


if __name__ == "__main__":
    from mh_data import MHDataset
    ds = MHDataset('HP1.txt', train=True, window_size=10, step_size=1)
    mlp = MHMLP(input_dim=500, vocab_size=ds.vocab_size)
    print(mlp)
    x = torch.tensor([[1, 2, 3, 4, 5]])
    y = mlp(x)
    print(y.shape)
    text = generate(mlp, ds.vocab, 'Harry Potter', sample=True)
    print(text)
    