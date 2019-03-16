import torch.nn as nn
import torch
from .transformer_package.transformer import TransformerEncoder

class Transformer0(nn.Module):
    """
    这里我采取的是把1000个时刻seq_len看成序列长度, 22个位置num_positions看成dim,但是我先把 seq_len变短，num_positions 变厚
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """

    def __init__(self, cut=20, num_layers=6, heads=4, drop_out=0.1, max_relative_positions=0,):
        super(Transformer0, self).__init__()
        self.cut = cut
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=cut*22, heads=heads, d_ff=cut*22,
                                                      dropout=drop_out, max_relative_positions=max_relative_positions,)
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(22*1000, 4)

    def forward(self, x, ):
        """
        :param x: size [bsz, 1, num_positions, seq_len,] [ bsz,22,1000]
        :return: y_: size  [bsz, 4]
        """
        x = x.squeeze()
        x = x.transpose(1, 2)  # bsz, 1000, 22
        bsz = x.size()[0]
        x = torch.reshape(x, (bsz, int(1000/self.cut), self.cut*22))  # e.g. 20, 50, 440
        outputs = self.transformer_encoder(x)
        encoder_outputs = torch.reshape(outputs, (bsz, 22 * 1000))
        y_ = self.softmax(self.classifier(encoder_outputs))
        return y_


class Transformer1(nn.Module):
    """
    这里我采取的是把1000个时刻seq_len看成dim, 22个位置num_positions看成序列长度
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """

    def __init__(self,num_layers=6, heads=8, drop_out=0.1, max_relative_positions=0):
        super(Transformer1, self).__init__()
        self.transformer_encoder = TransformerEncoder(num_layers=num_layers, d_model=1000, heads=heads, d_ff=1000,
                                                 dropout=drop_out, max_relative_positions=max_relative_positions,)
        self.softmax = nn.Softmax(dim=-1)
        self.classifier = nn.Linear(22*1000, self.num_labels)

    def forward(self, x, ):
        """
        :param x: size [bsz, 1, num_positions, seq_len,] [ bsz,22,1000]
        :return: y_: size  [bsz, 4]
        """
        x = x.squeeze()
        bsz, num_positions, seq_len = x.size()
        outputs = self.transformer_encoder(x)  # bsz, 22, 1000
        encoder_outputs = torch.reshape(outputs, (bsz, 22 * 1000))
        y_ = self.softmax(self.classifier(encoder_outputs))
        return y_


