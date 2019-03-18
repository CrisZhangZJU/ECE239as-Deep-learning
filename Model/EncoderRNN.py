import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    这是凌晨跑出59 的那个，其实因为这个给gru的shape是 num_positions(22),bsz(20),seq_len(1000)， hidden_size和model dim应该为1000才比较合理，到现在我也不知道
    1. 22个检测位置并未符合直觉的时序关系，2. dim按照我以往的思维应该设为1000，不知为何设为22也能有效
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """

    def __init__(self, linear_dropout=0.3, rnn_dropout=0, hidden_size=22, n_layers=2, num_labels=4, model_dim=22,  bidirectional=True
    ):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.model_dim = model_dim
        self.drop_out = nn.Dropout(
            linear_dropout)  # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        self.softmax = nn.Softmax(dim=-1)
        #   because our input size is a word embedding with number of features == hidden_size
        self.classifier = nn.Linear(self.model_dim, self.num_labels)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else rnn_dropout), bidirectional=bidirectional)
        self.linear = nn.Linear(1000 * 22, self.model_dim)
        self.relu = nn.ReLU()

    def forward(self, x, ):
        """
        :param x: size [bsz, 1, num_positions, seq_len,]
        :return: y_: size  [bsz, 4]
        """
        x = x.squeeze()
        bsz, num_positions, seq_len = x.size()
        x = x.permute(2, 0, 1)
        # x=x.reshape(1000,20,22)
        # [num_positions,bsz, seq_len]
        # print("EncoderRNN before gru x size", x.size())
        outputs, _ = self.gru(x)
        if self.bidirectional:
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # 这里就很诡异了，如果Hidden size没有很好对应的话，为何还会有效
        outputs = outputs.transpose(0, 1)
        encoder_outputs = torch.reshape(outputs, (bsz, 22 * 1000))
        y_ = self.softmax(self.classifier(self.drop_out(self.relu(self.linear(encoder_outputs)))))
        return y_


class EncoderRNN0(nn.Module):
    """
    第0版修改，和上面一样，给gru的shape是 num_positions(22),bsz(20),seq_len(1000), 把dim 换成1000， linear减少为一个
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """

    def __init__(self, linear_dropout=0.3, rnn_dropout=0, hidden_size=1000, n_layers=2, num_labels=4, model_dim=1000, bidirectional=True
    ):
        super(EncoderRNN0, self).__init__()
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.model_dim = model_dim
        self.drop_out = nn.Dropout(linear_dropout)  # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        self.softmax = nn.Softmax(dim=-1)
        #   because our input size is a word embedding with number of features == hidden_size
        self.classifier = nn.Linear(22*1000, self.num_labels)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else rnn_dropout), bidirectional=bidirectional)
        self.relu = nn.ReLU()

    def forward(self, x, ):
        """
        :param x: size [bsz, 1, num_positions, seq_len,]
        :return: y_: size  [bsz, 4]
        """
        x = x.squeeze()
        bsz, num_positions, seq_len = x.size()
        x = x.permute(2, 0, 1)
        # [num_positions,bsz, seq_len]
        print("EncoderRNN0 before gru x size",x.size())
        outputs, _ = self.gru(x)
        if self.bidirectional:
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)
        encoder_outputs = torch.reshape(outputs, (bsz, 22 * 1000))
        y_ = self.softmax((self.drop_out(self.classifier(encoder_outputs))))
        return y_


class EncoderRNN1(nn.Module):
    """
    第1版修改，和原始版本不同之处在于形状换成我原来的思维，按照我之前的思维，给gru的shape是 seq_len(1000) ,bsz(20),num_positions(22) , dim 仍为22，
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """

    def __init__(self, linear_dropout=0.3, rnn_dropout=0, hidden_size=22, n_layers=2, num_labels=4, model_dim=22, bidirectional=True
    ):
        super(EncoderRNN1, self).__init__()
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.model_dim = model_dim
        self.drop_out = nn.Dropout(
            linear_dropout)  # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        self.softmax = nn.Softmax(dim=-1)
        #   because our input size is a word embedding with number of features == hidden_size
        self.classifier = nn.Linear(self.model_dim, self.num_labels)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else rnn_dropout), bidirectional=bidirectional)
        self.linear = nn.Linear(1000 * 22, self.model_dim)
        self.relu = nn.ReLU()

    def forward(self, x, ):
        """
        :param x: size [bsz, 1, num_positions, seq_len,]
        :return: y_: size  [bsz, 4]
        """
        x = x.squeeze()
        x = x.tranpose(1, 2)
        bsz, seq_len, num_positions = x.size()
        x = x.permute(2, 0, 1)
        # [seq_len, bsz, num_positions]
        outputs, _ = self.gru(x)
        if self.bidirectional:
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)
        encoder_outputs = torch.reshape(outputs, (bsz, 22 * 1000))
        y_ = self.softmax(self.classifier(self.drop_out(self.relu(self.linear(encoder_outputs)))))
        return y_