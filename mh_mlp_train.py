import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mh_data import MHDataset
from mh_mlp import MHMLP, generate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 5
batch_size = 512
input_dim = 500


ds = MHDataset('HP1.txt', train=True, window_size=4, step_size=1)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

mlp = MHMLP(input_dim=input_dim, vocab_size=ds.vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

loss_history = []

text = 'Harry Potter'
initial_text = generate(mlp, ds.vocab, 'Harry Potter', sample=True, device=device)
print('Initial text:', initial_text)


for epoch in range(epochs):
    for i, (X, y) in enumerate(dl):
        X, y = X.to(device), y.to(device)
        y = y[:, -1]
        optimizer.zero_grad()
        y_pred = mlp(X)
        loss = criterion(y_pred, y)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

final_text = generate(mlp, ds.vocab, 'Harry Potter', sample=True)
print('Final text:', final_text)
        


