import numpy as np
from sklearn import metrics
import torch

groups = [0, 0, 1, 1, 1]
test = [[0, 0, 1, 1 , 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
test = torch.FloatTensor(test)

print()

aux = []

for t in range(len(test)):
    for group in list(set(groups)):
        sum_value = 0
        for i in ([i for i,x in enumerate(groups) if x == group]):
            sum_value += test[t][i]
            pass

        for i in ([i for i,x in enumerate(groups) if x == group]):
            #print(test[t][i])
            #print((sum_value > 0).float() * 3)
            test[t][i] = test[t][i] * (sum_value > 0).float() * 0.1
            pass
        pass


print(test)