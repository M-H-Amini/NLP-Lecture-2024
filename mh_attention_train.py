import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mh_data import MHDataset
from mh_attention import MHAttention, generate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
batch_size = 512
input_dim = 500
window_size = 100
step_size = 5
n_heads = 25

ds = MHDataset('HP1.txt', train=True, window_size=window_size, step_size=step_size)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

model = MHAttention(input_dim=input_dim, vocab_size=ds.vocab_size, window_len=window_size, n_heads=n_heads)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []

text = 'Harry and Ron were'.lower()
initial_text = generate(model, ds.vocab, text, sample=True, device=device)
print('Initial text:', initial_text)

for epoch in range(epochs):
    for i, (X, y) in enumerate(dl):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred.view(-1, ds.vocab_size), y.view(-1))
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

final_text = generate(model, ds.vocab, text, n=200, sample=True)
print('Final text:', final_text)
        