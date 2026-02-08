import torch
import torch.nn as nn

class MinimalModel(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
