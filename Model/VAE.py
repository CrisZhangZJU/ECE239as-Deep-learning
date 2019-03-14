from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Flatten function
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

# Permute function
class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(784, 400)
        # self.fc21 = nn.Linear(400, 20)
        # self.fc22 = nn.Linear(400, 20)
        # self.fc3 = nn.Linear(20, 400)
        # self.fc4 = nn.Linear(400, 784)
        BN_momentum = 0.2
        ELU_alpha = 0.9
        self.Conv1 = nn.Sequential(

            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=0),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            # 40, 22, 976

            nn.Conv2d(40, 40, kernel_size=(22, 1), stride=1, padding=0),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            # 40, 1, 976

            Permute(),
            # 1, 40, 976

            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            # 1, 40, 61

            Flatten(),
            # 40 * 61
        )

        self.fc11 = nn.Linear(40*61, 400)
        self.fc12 = nn.Linear(40*61, 400)

        self.fc2 = nn.Sequential(
            nn.Linear(400, 40*976),
            nn.ReLU(),
        )
        # self.fc3 = nn.Linear(40*61, 400)

        self.Conv2 = nn.Sequential(

            # 40, 1, 976
            nn.ConvTranspose2d(40, 40, (22, 1), 1, 0),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            # 40, 22, 976

            nn.ConvTranspose2d(40, 1, (1, 25), 1, 0),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            # 1, 22, 1000
        )


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
