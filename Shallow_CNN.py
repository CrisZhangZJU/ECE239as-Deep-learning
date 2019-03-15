import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import torchvision.transforms
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

criteria = F.cross_entropy

# Flatten function
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

# Permute function
class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)

class CNN_Shallow(nn.Module):
    def __init__(self):
        super(CNN_Shallow, self).__init__()
        BN_momentum = 0.1
        ELU_alpha = 0.9
        drop_out = 0.4
        self.feature = nn.Sequential(
                        
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=0),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            # 40, 22, 976
            
                                     #nn.Dropout2d(p=0.2),
            nn.Conv2d(40, 40, kernel_size=(8,1), stride=1, padding=0),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            nn.ELU(alpha=ELU_alpha, inplace=True),
           
                                     #nn.ELU(alpha=ELU_alpha, inplace=True),
            # 40, 15, 976
            nn.Conv2d(40, 40, kernel_size=(8,1), stride=1, padding=0),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            # 40, 8, 976
                         
            nn.Conv2d(40, 40, kernel_size=(8,1), stride=1, padding=0),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            
            # 40, 1, 976
            Permute(),
            # 1, 40, 976

            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            # 1, 40, 61

            Flatten(),

            nn.Linear(2440, 4),

        )


    def forward(self, x):

        out = self.feature(x)

        return out
