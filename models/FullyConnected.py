import torch
import torch.nn as nn
from torch.nn import functional as F


# Fully Connected 2-layer 512 node Neural Network
class FullyConnected(nn.Module):
  def __init__(self, n_classes, d_input, act_fn):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(d_input, 512),
                             act_fn(),
                             nn.Linear(512, 512),
                             act_fn(),
                             nn.Linear(512, n_classes))

  def forward(self, x):
    return self.net(x)
