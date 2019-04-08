import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight


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

class FocalLoss_for_non_zero(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, pos_weight=None, class_factor=1, groups=None):
        super(FocalLoss_for_non_zero, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight
        self.groups = groups
        self.class_factor = class_factor


    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.groups == None:
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
            F_loss = F_loss * aux
            F_loss = F_loss * self.class_factor
        else:
            #print(targets)
            #print(F_loss)
            for t in range(len(targets)):
                for group in list(set(self.groups)):
                    sum_value = 0
                    for i in ([i for i, x in enumerate(self.groups) if x == group]):
                        sum_value += targets[t][i]
                        pass

                    for i in ([i for i, x in enumerate(self.groups) if x == group]):
                        #print(F_loss[t][i])
                        #print((sum_value > 0).float() * self.alpha)
                        #torch.mul(F_loss[t][i], torch.FloatTensor((sum_value > 0).float() * self.alpha).cuda())
                        if group is 0:
                            F_loss[t][i] = F_loss[t][i].cpu() * self.class_factor
                        else:
                            F_loss[t][i] = F_loss[t][i].cpu() * ((sum_value > 0).float() * self.class_factor)

                            F_loss[t][i] = F_loss[t][i].cuda()
                        pass
                    pass
        #print(F_loss)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class BCE_for_non_zero(nn.Module):
    def __init__(self, logits=False, reduce=True, pos_weight=None, alpha=1, groups=None):
        super(BCE_for_non_zero, self).__init__()
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight
        self.groups = groups

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False, pos_weight=self.pos_weight)

        if self.groups == None:
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
        else:
            #print(targets)
            #print(BCE_loss)
            for t in range(len(targets)):
                for group in list(set(self.groups)):
                    sum_value = 0
                    for i in ([i for i, x in enumerate(self.groups) if x == group]):
                        sum_value += targets[t][i]
                        pass

                    for i in ([i for i, x in enumerate(self.groups) if x == group]):
                        #print(BCE_loss[t][i])
                        #print((sum_value > 0).float() * self.alpha)
                        #torch.mul(BCE_loss[t][i], torch.FloatTensor((sum_value > 0).float() * self.alpha).cuda())
                        if group is 0:
                            BCE_loss[t][i] = BCE_loss[t][i].cpu() * self.alpha
                        else:
                            BCE_loss[t][i] = BCE_loss[t][i].cpu() * ((sum_value > 0).float() * self.alpha)

                        BCE_loss[t][i] = BCE_loss[t][i].cuda()
                        pass
                    pass
        #print(BCE_loss)

        if self.reduce:
            return torch.mean(BCE_loss)
        else:
            return BCE_loss