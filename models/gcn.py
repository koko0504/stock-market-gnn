import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, X, A_norm):
        out = self.W(A_norm @ X)
        return torch.relu(out)

class TwoGCNLayer(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden)
        self.layer2 = nn.Linear(hidden, out_features, bias=False)
    
    def forward(self, X, A_norm):
        H1 = self.layer1(X, A_norm)
        H2 = self.layer2(A_norm @ H1)
        return H2
