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

class CNN_Deep(nn.Module):
    def __init__(self):
        super(CNN_Deep, self).__init__()
        BN_momentum = 0.2
        ELU_alpha = 0.9
        drop_out = 0.4
        self.feature = nn.Sequential(

            nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, padding=0),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=BN_momentum, affine=True),
            # 25, 22, 991

            nn.Conv2d(25, 25, kernel_size=(22, 1), stride=1, padding=0),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=BN_momentum, affine=True),
            # 25, 1, 991

            Permute(),
            # 1, 25, 991

            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            # 1, 25, 330

            nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=BN_momentum, affine=True),
            # 50, 1, 321

            Permute(),
            # 1, 50, 323

            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            # 1, 50, 107

            nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=BN_momentum, affine=True),
            # 100, 1, 98

            Permute(),
            # 1, 100, 98

            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            # 1, 100, 32

            nn.Conv2d(1, 200, kernel_size=(100, 10), stride=1, padding=0),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=BN_momentum, affine=True),
            # 200, 1, 23

            Permute(),
            # 1, 200, 23

            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            # 1, 200, 7

            Flatten(),

            nn.Linear(1400, 4),

        )


    def forward(self, x):

        out = self.feature(x)

        return out