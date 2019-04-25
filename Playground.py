import numpy as np
from sklearn import metrics
import torch
from torch import nn

m = nn.Linear(2, 2)
input = torch.randn(2, 2, 2)
output = m(input)
print(output.size())
print(input)
print(output)
