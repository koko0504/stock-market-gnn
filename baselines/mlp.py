import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
    
    def forward(self, X):
        h = torch.relu(self.fc1(X))
        return self.fc2(h)
