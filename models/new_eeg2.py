
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple

# Flatten function
class Flatten(nn.Module):
    def forward(self, x=0):
        N, C, H, W = x.size()
        return x.view(N, -1)

#---------------------------------------------
#main
class EEGNet(nn.Module):
    """docstring for EEG"""
    def __init__(self, nb_classes=4, Chans = 22, Samples = 1000, 
             dropoutRate = 0.25, kernLength = 125, F1 = 4, 
             D = 2, F2 = 8, norm_rate = 0.25, dropoutType = 'Dropout'):
        super(EEGNet, self).__init__()
        BN_momentum = 0.1
        ELU_alpha = 0.9
        # drop_out = 0.4
        # self.arg = arg
        self.conv1=nn.Conv2d(1,F1,kernel_size=(1, kernLength), padding=(0,62),stride=1)
        self.bn1=nn.BatchNorm2d(num_features=F1, eps=1e-05, momentum=BN_momentum, affine=True)
        self.depthwise=nn.Conv2d(F1,F2,kernel_size=(Chans,1))
            # nn.DepthwiseConv2D(kernel_size=[Chans, 1], use_bias = False, 
                                   # depth_multiplier = D,
                                   # depthwise_constraint = max_norm(1.))(block1)

        self.bn2=nn.BatchNorm2d(num_features=F2, eps=1e-05, momentum=BN_momentum, affine=True)
        self.elu=nn.ELU(alpha=ELU_alpha, inplace=True)
        self.meanpool1=nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout=nn.Dropout(dropoutRate)

        self.pointwise=nn.Conv2d(F2,F2,(1,17),padding=(0,8))
            # nn.SeparableConv2D(1,F2,(1,16))
        self.bn3=nn.BatchNorm2d(num_features=F2, eps=1e-05, momentum=BN_momentum, affine=True)
        self.meanpool2=nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.flatten = Flatten()

            # nn.Dense(nb_classes)
        self.Classifer=nn.Linear(F2*np.int(Samples/32), nb_classes)
        self.softmax=nn.Softmax(dim=-1)
            
    def forward(self, x):
        x=self.bn1(self.conv1(x))
        x=self.elu(self.bn2(self.depthwise(x)))
        x=self.dropout(self.meanpool1(x))

        x=self.bn3(self.pointwise(x))
        x=self.dropout(self.meanpool2(x))

        x = self.flatten(x)
        # print(type(x))
        x=self.Classifer(x)
        # print(type(x))

        # print(x)
        out = self.softmax(x)

        return out
