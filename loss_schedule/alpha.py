import numpy as np
import torch
import torch.nn as nn


class Alpha_Indentity(nn.Module):
    def __init__(self):
        super(Alpha_Indentity, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output1, output2, target):
        if output1 is not None:
            l1 = self.cross_entropy_loss(output1, target)
            l2 = self.cross_entropy_loss(output2, target)
            multiplied_loss = l1 * l2
            
        else:
            multiplied_loss = self.cross_entropy_loss(output2, target)
            
        return multiplied_loss.mean()

class Alpha_Sig(nn.Module):
    def __init__(self, warm_up_epoch=None):
        super(Alpha_Sig, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.warm_up_epoch = warm_up_epoch

    def forward(self, output1, output2, target, epoch=None):
        if output1 is not None:
            if self.warm_up_epoch is None or epoch > self.warm_up_epoch:
                l1 = self.cross_entropy_loss(output1, target)
                l1 = torch.sigmoid(l1)
                l2 = self.cross_entropy_loss(output2, target)
                multiplied_loss = l1 * l2
            else:
                multiplied_loss = self.cross_entropy_loss(output2, target)
        else:
            multiplied_loss = self.cross_entropy_loss(output2, target)
            
        return multiplied_loss.mean()

class Alpha_Tanh(nn.Module):
    def __init__(self):
        super(Alpha_Tanh, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output1, output2, target):
        if output1 is not None:
            l1 = self.cross_entropy_loss(output1, target)
            l1 = torch.tanh(l1)
            l2 = self.cross_entropy_loss(output2, target)
            multiplied_loss = l1 * l2
            
        else:
            multiplied_loss = self.cross_entropy_loss(output2, target)
            
        return multiplied_loss.mean()
