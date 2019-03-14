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

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
        super(RNN_LSTM, self).__init__()

        BN_momentum = 0.2
        ELU_alpha = 0.9

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

            # 1, 22, 1000
            # nn.AvgPool2d(kernel_size=(1, 30), stride=(1, 20)),
            # 1, 22, 49

        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, 4)
        )

    def forward(self, x):

        x = self.feature(x)
        B, C, H, W = x.size()
        x = x.view(B, H, W).permute(0, 2, 1)

        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.fc(r_out[:, -1, :])

        return out