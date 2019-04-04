import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, pos_weight=None, groups=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight
        self.groups = groups

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class BCE_for_non_zero(nn.Module):
    def __init__(self, logits=False, reduce=True, pos_weight=None, alpha=1):
        super(BCE_for_non_zero, self).__init__()
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False, pos_weight=self.pos_weight)


        aux = []
        for i in range(len(targets)):
            aux.append(torch.sum(targets[i]))

        aux = [aux, aux]
        aux = torch.FloatTensor(aux)
        aux = (aux > 0).float()
        if torch.cuda.is_available():
            aux = aux.cuda()

        aux = aux.t()

        # print(BCE_loss)
        # print(targets)
        # print(aux)
        # print(aux.size())
        BCE_loss = BCE_loss * aux
        BCE_loss = BCE_loss * self.alpha

        if self.reduce:
            return torch.mean(BCE_loss)
        else:
            return BCE_loss