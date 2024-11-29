import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(FeedForwardNet, self).__init__()
        self.l1 = nn.Linear(in_features=d_model, out_features=2048)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(in_features=2048, out_features=d_model)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))