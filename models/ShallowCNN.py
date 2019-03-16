import torch.nn as nn
import torch.nn.functional as F


# Flatten function
class Flatten(nn.Module):
    def forward(self, x=0):
        N, C, H, W = x.size()
        return x.view(N, -1)

# Permute function
class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)


class shallow_CNN(nn.Module):
    def __init__(self):
        super(shallow_CNN, self).__init__()
        BN_momentum = 0.2
        ELU_alpha = 0.9
        drop_out = 0.4
        # Why the kernel size is (1,25) but not (25, 1) here? 
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(1,25), stride=1, padding=0)
        # self.conv2 = nn.Conv2d()
        # self.conv2 = nn.Conv2d ()
        self.bn1=nn.BatchNorm2d(num_features =40, eps= 1e-05, momentum=BN_momentum, affine= True)
        self.elu=nn.ELU(alpha=ELU_alpha, inplace=True)
        self.permute= Permute()
        self.conv2 =  nn.Conv2d(40,40,kernel_size =(22,1), stride=1, padding=0)
        self.meanpool = nn.AvgPool2d(kernel_size=(1,75), stride= (1,15))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(in_features=2440,out_classes=4)

    def forward(self, x):
        # print(x.shape)
        # Input x: 20*1*22*1000 (N*C*H*W)
        x = self.conv1(x)
        x = self.bn1(x)
        # N， 40, 22, 976

        x = self.conv2(x)
        x = self.elu(x)
        x = self.bn1(x)
        # N， 40，1， 976
        x = self.permute(x)
        # N, 1, 40, 976

        x = self.meanpool(x)
        # N, 1, 40,61
        # print(len(x))
        x = self.flatten(x)
        # N*2440
        # print(x.shape)
        x = self.fc1(x)
        return x



# net = shallow_CNN()