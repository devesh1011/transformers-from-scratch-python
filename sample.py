import torch.nn as nn
import torch

pe = torch.zeros(5000, 512)
position = torch.arange(0, 512, dtype=torch.float).unsqueeze(1)
