import numpy as np
from sklearn import metrics
import torch

test = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
test = torch.FloatTensor(test)

aux = []
for i in range(len(test)):
    aux.append(torch.sum(test[i]))

aux = torch.FloatTensor(aux)

print((aux > 0) * 3)