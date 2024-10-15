import torch
import torch.nn as nn

class MHLSTM(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100, hidden_size=50, num_layers=1):
        super(MHLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
        
def generate(model, vocab, text, n=100, sample=True, device='cpu'):
    x_text = torch.tensor([vocab.index(c) for c in text]).unsqueeze(0).to(device)
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
            x_text = torch.cat([x_text, y], dim=1)
    model.train()
    return ''.join([vocab[idx] for idx in x_text.squeeze(0).tolist()])



if __name__ == "__main__":
    from mh_data import MHDataset
    ds = MHDataset('HP1.txt', train=True, window_size=10, step_size=1)
    X = ds[0][0].unsqueeze(0)
    print(X.shape)
    model = MHLSTM(input_dim=500, vocab_size=ds.vocab_size)
    print(model(X).shape)
    text = generate(model, ds.vocab, 'Harry Potter', sample=False)
    print(text)
    
    