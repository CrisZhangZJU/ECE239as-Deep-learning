import torch
import torch.nn as nn
#
#
# def relative_matmul(x, z, transpose):
#     """Helper function for relative positions attention."""
#     batch_size = x.shape[0]
#     heads = x.shape[1]
#     length = x.shape[2]
#     x_t = x.permute(2, 0, 1, 3)
#     x_t_r = x_t.reshape(length, heads * batch_size, -1)
#     if transpose:
#         z_t = z.transpose(1, 2)
#         x_tz_matmul = torch.matmul(x_t_r, z_t)
#     else:
#         x_tz_matmul = torch.matmul(x_t_r, z)
#     x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
#     x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
#     return x_tz_matmul_r_t
#
#
# def generate_relative_positions_matrix(length, max_relative_positions,
#                                        cache=False):
#     """Generate the clipped relative positions matrix
#        for a given length and maximum relative positions"""
#     if cache:
#         distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
#     else:
#         range_vec = torch.arange(length)
#         range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
#         distance_mat = range_mat - range_mat.transpose(0, 1)
#     distance_mat_clipped = torch.clamp(distance_mat,
#                                        min=-max_relative_positions,
#                                        max=max_relative_positions)
#     # Shift values to be >= 0
#     final_mat = distance_mat_clipped + max_relative_positions
#     return final_mat
# class MultiHeadedAttention(nn.Module):
#     """Multi-Head Attention module from "Attention is All You Need"
#     :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
#
#     Similar to standard `dot` attention but uses
#     multiple attention distributions simulataneously
#     to select relevant items.
#
#     .. mermaid::
#
#        graph BT
#           A[key]
#           B[value]
#           C[query]
#           O[output]
#           subgraph Attn
#             D[Attn 1]
#             E[Attn 2]
#             F[Attn N]
#           end
#           A --> D
#           C --> D
#           A --> E
#           C --> E
#           A --> F
#           C --> F
#           D --> O
#           E --> O
#           F --> O
#           B --> O
#
#     Also includes several additional tricks.
#
#     Args:
#        head_count (int): number of parallel heads
#        model_dim (int): the dimension of keys/values/queries,
#            must be divisible by head_count
#        dropout (float): dropout parameter
#     """
#
#     def __init__(self, head_count, model_dim, dropout=0.1,
#                  max_relative_positions=0,
#                  lb=None, before_softmax=True,
#                  before_padding=True, d=None, softmax_again=True,
#                  attention_type='encoder_self_attn', selected_self_attn='all'
#
#                  ):
#         assert model_dim % head_count == 0
#         self.dim_per_head = model_dim // head_count
#         self.model_dim = model_dim
#
#         self.d = d
#         # if selected_self_attn != 'all':
#         #     if selected_self_attn == 'two':
#         #         if attention_type == 'decoder_self_attn':
#         #             self.d = 0
#         #     elif attention_type != selected_self_attn:
#         #         self.d = 0
#         if attention_type == 'decoder_self_attn':
#             self.d = 0
#         self.softmax_again = softmax_again
#         self.lb = lb
#         self.before_softmax = before_softmax
#         self.before_padding = before_padding
#         # self.encoder_self_attn = encoder_self_attn
#         # self.decoder_self_attn = decoder_self_attn
#         # self.decoder_context_attn = decoder_context_attn
#
#         super(MultiHeadedAttention, self).__init__()
#         self.head_count = head_count
#
#         self.linear_keys = nn.Linear(model_dim,
#                                      head_count * self.dim_per_head)
#         self.linear_values = nn.Linear(model_dim,
#                                        head_count * self.dim_per_head)
#         self.linear_query = nn.Linear(model_dim,
#                                       head_count * self.dim_per_head)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)
#         self.final_linear = nn.Linear(model_dim, model_dim)
#
#         self.max_relative_positions = max_relative_positions
#
#         if max_relative_positions > 0:
#             vocab_size = max_relative_positions * 2 + 1
#             self.relative_positions_embeddings = nn.Embedding(
#                 vocab_size, self.dim_per_head)
#
#     def forward(self, key, value, query, mask=None,
#                 layer_cache=None, type=None):
#         """
#         Compute the context vector and the attention vectors.
#
#         Args:
#            key (FloatTensor): set of `key_len`
#                key vectors ``(batch, key_len, dim)``
#            value (FloatTensor): set of `key_len`
#                value vectors ``(batch, key_len, dim)``
#            query (FloatTensor): set of `query_len`
#                query vectors  ``(batch, query_len, dim)``
#            mask: binary mask indicating which keys have
#                non-zero attention ``(batch, query_len, key_len)``
#         Returns:
#            (FloatTensor, FloatTensor):
#
#            * output context vectors ``(batch, query_len, dim)``
#            * one of the attention vectors ``(batch, query_len, key_len)``
#         """
#
#         # CHECKS
#         # batch, k_len, d = key.size()
#         # batch_, k_len_, d_ = value.size()
#         # aeq(batch, batch_)
#         # aeq(k_len, k_len_)
#         # aeq(d, d_)
#         # batch_, q_len, d_ = query.size()
#         # aeq(batch, batch_)
#         # aeq(d, d_)
#         # aeq(self.model_dim % 8, 0)
#         # if mask is not None:
#         #    batch_, q_len_, k_len_ = mask.size()
#         #    aeq(batch_, batch)
#         #    aeq(k_len_, k_len)
#         #    aeq(q_len_ == q_len)
#         # END CHECKS
#
#         batch_size = key.size(0)
#         dim_per_head = self.dim_per_head
#         head_count = self.head_count
#         key_len = key.size(1)
#         query_len = query.size(1)
#         device = key.device
#
#         def shape(x):
#             """Projection."""
#             return x.view(batch_size, -1, head_count, dim_per_head) \
#                 .transpose(1, 2)
#
#         def unshape(x):
#             """Compute context."""
#             return x.transpose(1, 2).contiguous() \
#                     .view(batch_size, -1, head_count * dim_per_head)
#
#         # 1) Project key, value, and query.
#         if layer_cache is not None:
#             if type == "self":
#                 query, key, value = self.linear_query(query),\
#                                     self.linear_keys(query),\
#                                     self.linear_values(query)
#                 key = shape(key)
#                 value = shape(value)
#                 if layer_cache["self_keys"] is not None:
#                     key = torch.cat(
#                         (layer_cache["self_keys"].to(device), key),
#                         dim=2)
#                 if layer_cache["self_values"] is not None:
#                     value = torch.cat(
#                         (layer_cache["self_values"].to(device), value),
#                         dim=2)
#                 layer_cache["self_keys"] = key
#                 layer_cache["self_values"] = value
#             elif type == "context":
#                 query = self.linear_query(query)
#                 if layer_cache["memory_keys"] is None:
#                     key, value = self.linear_keys(key),\
#                                  self.linear_values(value)
#                     key = shape(key)
#                     value = shape(value)
#                 else:
#                     key, value = layer_cache["memory_keys"],\
#                                layer_cache["memory_values"]
#                 layer_cache["memory_keys"] = key
#                 layer_cache["memory_values"] = value
#         else:
#             key = self.linear_keys(key)
#             value = self.linear_values(value)
#             query = self.linear_query(query)
#             key = shape(key)
#             value = shape(value)
#
#         if self.max_relative_positions > 0 and type == "self":
#             key_len = key.size(2)
#             # 1 or key_len x key_len
#             relative_positions_matrix = generate_relative_positions_matrix(
#                 key_len, self.max_relative_positions,
#                 cache=True if layer_cache is not None else False)
#             #  1 or key_len x key_len x dim_per_head
#             relations_keys = self.relative_positions_embeddings(
#                 relative_positions_matrix.to(device))
#             #  1 or key_len x key_len x dim_per_head
#             relations_values = self.relative_positions_embeddings(
#                 relative_positions_matrix.to(device))
#
#         query = shape(query)
#
#         key_len = key.size(2)
#         query_len = query.size(2)
#
#         # 2) Calculate and scale scores.
#         query = query / math.sqrt(dim_per_head)
#         # batch x num_heads x query_len x key_len
#         query_key = torch.matmul(query, key.transpose(2, 3))
#
#         if self.max_relative_positions > 0 and type == "self":
#             scores = query_key + relative_matmul(query, relations_keys, True)
#         else:
#             scores = query_key
#         scores = scores.float()
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
#             scores = scores.masked_fill(mask, -1e18)
#         # 3) Apply attention dropout and compute context vectors.
#         attn = self.softmax(scores).to(query.dtype)
#
#         drop_attn = self.dropout(attn)
#
#         context_original = torch.matmul(drop_attn, value)
#
#         if self.max_relative_positions > 0 and type == "self":
#             context = unshape(context_original
#                               + relative_matmul(drop_attn,
#                                                 relations_values,
#                                                 False))
#         else:
#             context = unshape(context_original)
#
#         output = self.final_linear(context)
#         # CHECK
#         # batch_, q_len_, d_ = output.size()
#         # aeq(q_len, q_len_)
#         # aeq(batch, batch_)
#         # aeq(d, d_)
#
#         # Return one attn
#         top_attn = attn \
#             .view(batch_size, head_count,
#                   query_len, key_len)[:, 0, :, :] \
#             .contiguous()
#
#         return output, top_attn


class EncoderRNN(nn.Module):
    """
    :Args:
        hidden_size: number of the positions
        n_layers: number of the rnn layers
    """
    def __init__(
            # self, hidden_size, embedding, n_layers=1, dropout=0
            self, hidden_size=22, n_layers=1, num_labels=4, model_dim=22, dropout=0, bidirectional=True
    ):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        # self.embedding = embedding
        self.bidirectional = bidirectional
        self.model_dim = model_dim
        self.drop_out = nn.Dropout(dropout)        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'

        # self.rnn_encoder = EncoderRNN(hidden_size=self.model_dim, n_layers=self.n_layers, dropout=self.drop_out)
        self.softmax = nn.Softmax(dim=-1)
        #   because our input size is a word embedding with number of features == hidden_size
        self.classifier = nn.Linear(self.model_dim,self.num_labels)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=bidirectional)

    def forward(
            # self, input_seq, input_lengths, hidden=None
            self, x,
    ):
        """
        :param x: size [bsz, seq_len, num_positions]
        :return: outputs: size  [bsz,sequence_len,num_positions] combine each rnn hidden states together
        """
        # x = x.transpos(0, 1)
        x=x.squeeze() 
        x= x.permute(2,0,1)
        # x=x.reshape(1000,20,22)
        # [sequence_length,bsz, num_positions]
        outputs, _ = self.gru(x)
        if self.bidirectional:
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs.transpose(0, 1)
        # encoder_outputs=self.rnn_encoder(x)
        # last_hidden= encoder_outputs[:,-1,:]
        # outputs =self.softmax(self.classifier(last_hidden))
        # return outputs


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class SAN(nn.Module):
    def __init__(self, model_dim, dropout=0.1
                 ):
        self.model_dim = model_dim

        super(SAN, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """
        # 2) Calculate and scale scores.
        # batch  x query_len x key_len
        scores = torch.matmul(query, key.transpose(1, 2)).float()
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value)
        output = self.final_linear(context)
        return output


class EncoderSAN(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout,):
        super(EncoderSAN, self).__init__()

        self.self_attn = SAN(d_model, dropout=dropout,)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (FloatTensor): ``(batch_size, src_len, model_dim)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(x)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,)
        out = self.dropout(context) + x
        return self.feed_forward(out)


class Model(nn.Module):
    def __init__(self, encoder_type='BiGRU_linear',num_rnn_layer=2):
        super(Model,self).__init__()
        self.encoder_type = encoder_type
        self.model_dim = 22
        self.drop_out = 0
        self.num_layers= num_rnn_layer
        self.num_labels = 4
        self.dropout = 0.3
        self.san_encoder = EncoderSAN(d_model=self.model_dim, d_ff=self.model_dim, dropout=self.drop_out)
        self.rnn_encoder = EncoderRNN(hidden_size=self.model_dim, n_layers=self.num_layers, dropout=self.drop_out)
        self.classifier = nn.Linear(self.model_dim,self.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_1 = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(self.dropout)
        self.w_1 = nn.Linear(self.model_dim*100, 40)
        self.w_2 = nn.Linear(40, self.model_dim)
        self.layer_norm = nn.LayerNorm(self.model_dim*100, eps=1e-6)
        self.linear = nn.Linear(1000*22, self.model_dim)

    def forward(self,x):
        """
        :param x: (FloatTensor): ``(batch_size, src_len, model_dim)``
        :return: outputs : bsz,num_labels
        """
        bsz = x.size()[0]
        print('input size',x.size())
        if self.encoder_type == "BiGRU_last":
            encoder_outputs = self.rnn_encoder(x)
            last_hidden = encoder_outputs[:, -1, :]  # bsz*dim
            y_ = self.softmax(self.classifier(last_hidden))
        if self.encoder_type == "SAN_GRU_last":
            encoder_outputs = self.rnn_encoder(self.san_encoder(x))
            last_hidden = encoder_outputs[:, -1, :]  # bsz*dim
            y_ = self.softmax(self.classifier(last_hidden))
        if self.encoder_type == 'BiGRU_every_10':
            encoder_outputs = self.rnn_encoder(x)
            
            every_10 = torch.stack([encoder_outputs[:, (i+1)*10-1, :] for i in range(100)])
            condense_encoder_ouputs = torch.reshape(every_10,(bsz,100*self.model_dim))
            inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(condense_encoder_ouputs))))
            output = self.dropout_2(self.w_2(inter))
            y_ = self.softmax(self.classifier(output))
        if self.encoder_type == 'BiGRU_linear':
            encoder_outputs = torch.reshape(self.rnn_encoder(x),(bsz,22*1000))
            y_ = self.softmax(self.classifier(self.dropout_1(self.relu(self.linear(encoder_outputs)))))
        return y_



