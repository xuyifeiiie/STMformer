import torch
import torch.nn as nn
from layer.attention import FullAttention, ProbAttention


def split_X_into_x_by_VM(Xt, instances_index_vm):
    """
    :param Xt: [B, T, N, d_model] 一批数据某一时刻的X
    :param instances_index_vm: (List)[n1,n2,...] n1表示节点是在VM1上的个数，n2表示在VM2上的个数，。。。 torch.Size([1, 12, 6, 1])
    :return: [[B, T, N1, d_model],...], tuple类型; N1+N2+...=N
    """
    index = instances_index_vm[0][0]
    index_list = index.squeeze().tolist()
    index_list = list(map(int, index_list))
    Xt = torch.split(Xt, index_list, dim=2)
    return Xt


class VMAttentionModule(nn.Module):
    def __init__(self, num_instances=56, attention=None, d_model=512, n_heads=8, factor=5, dropout=0.1,
                 history_steps=16,
                 output_attention=False, activation="relu", mix=False, mask_flag=False):
        super(VMAttentionModule, self).__init__()
        self.num_instances = num_instances
        self.attention = attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        self.history_steps = history_steps
        self.output_attention = output_attention
        self.activation = activation
        self.mask_flag = mask_flag
        self.mix = mix

        if attention == 'prob':
            self.self_attention = ProbAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mask_flag=self.mask_flag,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention,
            )
        else:
            self.self_attention = FullAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mask_flag=self.mask_flag,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention,
            )

        self.d_query = d_model // n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.query_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(self.d_model, self.d_values * self.n_heads)
        self.out_projection = nn.Linear(self.d_values * self.n_heads, self.d_model)

        self.norm1 = nn.BatchNorm1d(self.history_steps * self.num_instances)
        self.norm2 = nn.BatchNorm1d(self.history_steps * self.num_instances)
        self.linear = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, instances_index_vm, attention_mask=None):
        """
        :param x: B T N d_model
        :param instances_index_vm: instances on VMs index like [6,7,...] 6 pods deployed on vm1,...
        :param attention_mask:
        :return: B T N d_model, [[B T H N1 N1],[B T H N2 N2],...]
        """
        B, T, N, d_model = x.shape
        H = self.n_heads

        x = x.reshape(B, -1, self.d_model)  # B T N d_model -> B TN d_model
        x = self.norm1(x)  # B TN d_model ->  B TN d_model
        x = x.reshape(B, T, N, d_model)  # B TN d_model -> B T N d_model

        # x_tuple: [[B, T, N1, d_model],[B, T, N2, d_model],...], tuple类型; N1+N2+...=N
        x_tuple = split_X_into_x_by_VM(x, instances_index_vm=instances_index_vm)
        x_attns = []
        attns = []

        # x_instance: B, T, N1, d_model
        for x_instance in x_tuple:
            # B T N1 d_model -> B T N1 d_keys * n_heads -> B T N1 n_heads d_keys
            queries = self.query_projection(x_instance).view(B, T, -1, H, self.d_query)
            # B T S1 d_model -> B T S1 d_keys * n_heads -> B T S1 n_heads d_keys
            keys = self.key_projection(x_instance).view(B, T, -1, H, self.d_keys)
            # B T S1 d_model -> B T S1 d_values * n_heads -> B T S1 n_heads d_values
            values = self.value_projection(x_instance).view(B, T, -1, H, self.d_values)

            # queries: B T N1 n_heads d_query; keys: B T S1 n_heads d_keys; values: B T S1 n_heads d_values
            # x_attn: B T N1 H d_values, attn: B T H N1 S1
            x_attn, attn = self.self_attention(queries, keys, values, attention_mask=attention_mask)
            # x_attn: B T N1 H d_value -> B T N1 H*d_values
            x_attn = x_attn.reshape(B, T, -1, H * self.d_values)  # B T N1 H d_values -> B T N1 H*d_values
            x_attns.append(x_attn)
            attns.append(attn)

        x_cat = torch.cat(x_attns, dim=2)  # B T N H*d_values
        x_cat = self.out_projection(x_cat)  # B T N H*d_values -> B T N d_model
        x_out = (x_cat + x).reshape(B, -1, self.d_model)  # B T N d_model -> B TN d_model
        output = self.norm2(x_out)  # B TN d_model
        output = output.reshape(B, T, N, d_model)  # B TN d_model -> B T N d_model
        return self.dropout(output), attns
