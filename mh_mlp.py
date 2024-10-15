import torch
import torch.nn as nn
import torch.optim as optim
from mh_data import MHDataset


class MLP(nn.Module):
    def __init__(self, input_dim=500, vocab_size=100, n_hidden_1=50, n_hidden_2=15):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, vocab_size)

    def forward(self, x):
        pass 


if __name__ == "__main__":
    mlp = MLP()
    print(mlp)