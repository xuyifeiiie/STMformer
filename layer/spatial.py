import torch
import torch.nn as nn
from utils.masking import GATMask
import torch.nn.functional as F

"""
A_update = A_mask * A   A.shape = TN TN
A_update matmal X  X.shape = B TN d_model

"""


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, dropout=0.1, history_steps=16, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = input_dim  # 节点表示向量的输入特征维度
        self.out_features = output_dim  # 节点表示向量的输出特征维度
        # self.num_of_instances = num_of_instances
        self.history_steps = history_steps
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(2 * output_dim, 1)
        # self.adj_mask = nn.Linear(num_of_instances, num_of_instances)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # self.W = nn.Parameter(torch.zeros(size=(d_model, d_model)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        # self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # if self.out_features != self.in_features:
        #     self.linear3 = nn.Linear(self.out_features, self.in_features)

    def forward(self, x, adj_mx):
        """
        :param x: B T N d_model
        :param adj_mx: B T N N -> TN TN
        :return: B T N d_model
        """
        B, T, N, d_model = x.shape

        h = self.linear1(x)  # B T N input_dim -> B T N output_dim

        # h: B T N output_dim -> B T N N*output_dim -> B T N*N output_dim
        # h: B T N output_dim -> B T N*N output_dim
        # cat: B T N*N 2output_dim ->  B T N N 2output_dim
        a_input = torch.cat([h.repeat(1, 1, 1, N).view(B, T, N * N, -1), h.repeat(1, 1, N, 1)],
                            dim=3).view(B, T, N, -1, 2 * self.out_features)

        # B T N N 2output_dim -> B T N N 1 -> B T N N
        e = self.leakyrelu(self.linear2(a_input).squeeze(4))
        # B T N N -> B N T N -> B N TN -> B TN TN
        e = e.permute(0, 2, 1, 3).reshape(B, N, -1).repeat(1, T, 1)
        # B T N N 1 => B T N N 图注意力的相关系数（未归一化）

        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        adj_mx = adj_mx.reshape(B, -1, N).repeat(1, 1, T)
        gat_mask = GATMask(B, self.history_steps, N, adj_mx, device=x.device)  # B TN TN
        mask_value = -1e12 if e.dtype == torch.float32 else -1e4
        mask_attention_adj = e.masked_fill_(gat_mask.mask, mask_value)  # B TN TN
        # 原attention: B T N N， attention: B TN TN
        attention = F.softmax(mask_attention_adj.float(), dim=-1)  # softmax形状保持不变 N, N，得到归一化的注意力权重！
        attention = self.dropout(attention)
        # 原h_output: B T N N * B T N output_dim -> B T N output_dim
        # h_output = torch.einsum('btnn, btnd->btnd', attention, h)
        # B T N output_dim -> B TN output_dim
        h = h.reshape(B, -1, self.out_features)
        # B TN TN * B TN output_dim -> B TN output_dim
        h_output = torch.einsum('btt, btd->btd', attention, h)
        # B TN output_dim -> B T N output_dim
        h_output = h_output.reshape(B, T, N, -1)

        if self.concat:
            return F.elu(h_output)
        else:
            return h_output


class SpatialRelationModule(nn.Module):
    def __init__(self, d_model, hidden_dim, alpha=0.2, n_heads=8, dropout=0.1, history_steps=24):
        """
        :param d_model: 输入输出维度
        :param hidden_dim: 隐藏维度
        :param alpha: 激活函数LeakyRelu的参数
        :param n_heads: 多头
        :param dropout: 丢失率
        :param history_steps: 历史时间步
        """
        super(SpatialRelationModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_keys = d_model // n_heads
        self.d_model = d_model

        self.graph_attentions_layers = nn.ModuleList([GraphAttentionLayer(input_dim=self.d_model,
                                                                          output_dim=self.d_keys,
                                                                          alpha=alpha,
                                                                          dropout=dropout,
                                                                          history_steps=history_steps,
                                                                          concat=True)
                                                      for _ in range(n_heads)])
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(input_dim=(self.d_keys * n_heads),
                                           output_dim=self.d_model,
                                           alpha=alpha,
                                           dropout=dropout,
                                           history_steps=history_steps,
                                           concat=False)
    def forward(self, x, adj_in):
        """
        :param x: B T N d_model
        :param adj_in: N N
        :return: B T N d_model
        """
        x = self.dropout(x)
        graph_attention_list = []
        for graph_attentions_layer in self.graph_attentions_layers:
            graph_attention_list.append(graph_attentions_layer(x, adj_in))
        # [[B T N output_dim],[B T N output_dim],...]
        x = torch.cat(graph_attention_list, dim=3)  # 将每个head得到的表示进行拼接
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj_in))  # 输出并激活
        return x
