import torch.nn as nn


# Flatten function
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)


# Permute function
class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1, 3)

# In this round, I add drop out to all conv2D and rnn-2_layer
class RNN0_CNN(nn.Module):
    def __init__(self, rnn_num_layers=1, rnn_dropout=0.5):
        super(RNN0_CNN, self).__init__()
        BN_momentum = 0.2
        ELU_alpha = 0.9
        self.hidden_size = 22
        n_layers = rnn_num_layers
        rnn_dropout = rnn_dropout
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else rnn_dropout), bidirectional=True)
        self.feature = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1, padding=0),
            # nn.dropout(0.5),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            # 40, 22, 976
            nn.Conv2d(40, 40, kernel_size=(22, 1), stride=1, padding=0),
            # nn.dropout(0.5),
            nn.ELU(alpha=ELU_alpha, inplace=True),
            nn.BatchNorm2d(num_features=40, eps=1e-05, momentum=BN_momentum, affine=True),
            # 40, 1, 976
            Permute(),
            # 1, 40, 976
            nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15)),
            # 1, 40, 61
            Flatten(),
            nn.Linear(2440, 4),
        )

    def forward(self, x):
        # x shape bsz,1,22,1000
        x = x.squeeze()
        bsz,  _, _, = x.size()
        x = x.permute(2, 0, 1)  # 1000,bsz,22
        outputs, _ = self.gru(x)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # 1000, bsz, 22
        outputs = outputs.transpose(0, 1)  # bsz,1000, 22
        outputs = outputs.transpose(1, 2)  # bsz,22, 1000
        cnn_inputs = outputs.unsqueeze(1)
        out = self.feature(cnn_inputs)
        return out


class RNN1_CNN(nn.Module):
    def __init__(self):
        super(RNN1_CNN, self).__init__()
        BN_momentum = 0.2
        ELU_alpha = 0.9
        self.hidden_size = 1000
        n_layers = 1
        rnn_dropout = 0
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else rnn_dropout), bidirectional=True)
        self.feature = nn.Sequential(
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
            nn.Linear(2440, 4),
        )

    def forward(self, x):
        # x shape bsz,1,22,1000
        x = x.squeeze()
        bsz, _, _ = x.size()
        x = x.tranpose(1, 2)
        x = x.permute(2, 0, 1)  # 22,bsz,1000
        outputs, _ = self.gru(x)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # 22,bsz,1000
        outputs = outputs.transpose(0, 1)  # bsz,22, 1000
        cnn_inputs = outputs.unsqueeze(1)
        out = self.feature(cnn_inputs)
        return out


