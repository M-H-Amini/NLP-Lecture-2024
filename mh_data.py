import torch 
from torch.utils.data import Dataset

class MHDataset(Dataset):
    def __init__(self, filename, train=True, train_ratio=0.8, window_size=5, step_size=1, lowercase=False):
        self.txt_file = open(filename, 'r').read()
        if lowercase:
            self.txt_file = self.txt_file.lower()
        file_len = len(self.txt_file)
        print(f'File length: {file_len}')
        if train:
            self.txt_file = self.txt_file[:int(file_len*train_ratio)]
        else:
            self.txt_file = self.txt_file[int(file_len*train_ratio):]
        self.window_size = window_size
        self.step_size = step_size
        self.vocab = list(sorted(set(self.txt_file)))
        self.vocab_size = len(self.vocab)
        # self.indices = list(range(0, file_len, self.step_size))  ##  Indices of beginning of each window
        self.indices = [i for i in range(0, len(self.txt_file) - self.window_size - 1, self.step_size)]

        
        

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        return torch.tensor([self.char2Idx(c) for c in self.txt_file[start:end]]), torch.tensor([self.char2Idx(c) for c in self.txt_file[start+1:end+1]])
    
    def char2Idx(self, char):
        return self.vocab.index(char)
    
    def idx2Char(self, idx):
        return self.vocab[idx]

if __name__ == "__main__":
    dataset = MHDataset('HP1.txt', train=True, train_ratio=.8, window_size=5, step_size=1)
    print(len(dataset))
    print(dataset[0])