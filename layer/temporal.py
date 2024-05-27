import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
import pywt

from layer.attention import TemporalProbAttention, TemporalFullAttention
from layer.common import FlattenPatch
from layer.crosscorrelation import CrossCorrealtion
from layer.embed import PatchEmbedding
from utils.tools import adj_to_edge_index


def FFT_for_Period_Node(x, k=10):
    """
    :param x: B 2T N*d_model   B 2T N d_model
    :param k:
    :return: period: B k N, top_matrix: B k N*d_model  B k N
    """
    xf = torch.fft.rfft(x, dim=1)  # B S N d_model
    # find period by amplitudes
    frequency_amplitude_matrix = abs(xf).mean(-1)  # B S N, S 频率个数  振幅相关的权重 多维特征进行了平均值统一处理。应该BN分开处理
    frequency_amplitude_matrix[:, 0, :] = 0
    _, top_matrix = torch.topk(frequency_amplitude_matrix, k, dim=1)  # B k N return top values and index
    top_matrix = top_matrix.detach().cpu().numpy()
    period = x.shape[1] // top_matrix  # B k N(N*d_model)
    return period, top_matrix  # B k N(N*d_model), B k N(N*d_model)


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)  # [B, T, C]
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)  # T
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)  # k
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # T//k
    return period, abs(xf).mean(-1)[:, top_list]  # T//k, B k


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            if i % 2 == 0:
                continue
            else:
                kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
            # kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(int(self.num_kernels / 2)):
            # for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlockV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            # 1,3 1,5 1,7
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionDownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.parallel_kernels_num = (self.num_kernels // 2) + 2  # 5
        kernels = []
        for i in range(self.num_kernels // 2):  # 0,1,2
            kernels.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Sequential(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=[1, 3], padding=[0, 1]),
                    nn.Conv2d(out_channels, out_channels, kernel_size=[3, 1], padding=[1, 0])
                ) * (i + 1)),
            ))

        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        kernels.append(nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        ))

        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.parallel_kernels_num):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class InceptionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.init_weight = init_weight

        if self.out_channels >= self.in_channels:
            self.parallel_kernels_num = int(math.log(self.out_channels // self.in_channels, 2))  # 512/64=8 -> 3
        else:
            pass

        self.conv_pool = {}
        for i in range(self.parallel_kernels_num):
            cin = int(math.pow(2, i) * self.in_channels)  # 64, 128, 256
            self.conv_pool[i] = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(cin, cin, kernel_size=1),
                    nn.Conv2d(cin, cin, kernel_size=[1, 3], padding=[0, 1]),
                    nn.Conv2d(cin, cin, kernel_size=[3, 1], padding=[1, 0]),
                ),
                nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                )
            ])

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.parallel_kernels_num):
            res_layer = []
            for seq in self.conv_pool[i]:
                self.conv_pool[i].to(x.device)
                res = seq(x)
                res_layer.append(res)
            x = torch.concat(res_layer, dim=1)
        return x


class TemporalRelationModule(nn.Module):
    def __init__(self, d_model, num_features, hidden_dim, kernels=6, dropout=0.1, top_k=5, history_steps=16,
                 predict_steps=16):
        super(TemporalRelationModule, self).__init__()

        self.d_model = d_model
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_kernels = kernels
        self.dropout = dropout
        self.k = top_k
        self.history_steps = history_steps
        self.predict_steps = predict_steps

        # self.input_channel_linear = nn.Linear(self.d_model, 1)
        self.input_step_linear = nn.Linear(self.history_steps, self.history_steps + self.predict_steps)

        # self.conv = nn.Sequential(
        #     InceptionDownBlock(in_channels=self.d_model,
        #                        out_channels=self.hidden_dim,
        #                        num_kernels=self.num_kernels),
        #     nn.GELU(),
        #     InceptionUpBlock(in_channels=self.hidden_dim,
        #                      out_channels=self.d_model,
        #                      num_kernels=self.num_kernels)
        # )
        self.conv = nn.Sequential(
            InceptionBlockV2(in_channels=self.d_model,
                             out_channels=self.hidden_dim,
                             num_kernels=self.num_kernels),
            nn.GELU(),
            InceptionBlockV2(in_channels=self.hidden_dim,
                             out_channels=self.d_model,
                             num_kernels=self.num_kernels)
        )

        self.output_step_linear = nn.Linear(self.history_steps + self.predict_steps, self.history_steps)
        # self.output_channel_linear = nn.Linear(1, self.d_model)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        """
        N * d_model 是原来考虑的维度，由于计算复杂度高。故而先对x的d_model改成features，再求平均，消除N的维度，
        :param x: B T N d_model
        :param adj: B T N N
        :return: B T N d_model
        """
        B, T, N, d_model = x.shape
        x = x.permute(0, 2, 3, 1)  # B T N d_model -> B N d_model T
        x = self.input_step_linear(x)  # B N d_model T -> B N d_model 2T
        x = x.permute(0, 3, 1, 2)  # B N d_model 2T -> B 2T N d_model

        adj_tmp = adj.sum(dim=0).sum(dim=0)  # B T N N -> N N
        edge_index_tmp = adj_to_edge_index(adj_tmp)
        row, col = edge_index_tmp
        d_in = degree(col, adj_tmp.shape[1]).float()  # N
        d_out = degree(row, adj_tmp.shape[1]).float()  # N
        d_weight = F.softmax(d_in + d_out, dim=0)

        # B 2T N d_model -> B 2T d_model N -> B 2T d_model
        x = x.permute(0, 1, 3, 2)
        x = torch.einsum("btdn,n->btd", x, d_weight)

        period_list, period_weight = FFT_for_Period(x, self.k)  # 2T//k, B k

        res_list = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.history_steps + self.predict_steps) % period != 0:
                length = (((self.history_steps + self.predict_steps) // period) + 1) * period
                delta = length - (self.history_steps + self.predict_steps)
                padding = torch.zeros([x.shape[0], delta, x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)  # B length d_model
            else:
                length = (self.history_steps + self.predict_steps)
                out = x  # B length/2T d_model
            # reshape B num_period period d_model -> B d_model num_period period
            out = out.reshape(x.shape[0], length // period, period, x.shape[2])
            out = out.permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            # B d_model num_period period -> B num_period period d_model -> B 2T d_model
            out = out.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[2])
            res_list.append(out[:, :(self.history_steps + self.predict_steps), :])
        res = torch.stack(res_list, dim=-1)  # B 2T d_model k
        # adaptive aggregation
        # B k
        period_weight = F.softmax(period_weight, dim=1)
        # B k -> B 1 1 k -> B 2T d_model k
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, self.history_steps + self.predict_steps,
                                                                       self.d_model, 1)
        # B 2T d_model k * B 2T d_model k -> B 2T d_model k -> B 2T d_model
        res = torch.sum(res * period_weight, -1)
        # B 2T d_model -> B 1 2T d_model -> B 1 d_model 2T
        res = res.reshape(B, -1, self.history_steps + self.predict_steps, self.d_model).permute(0, 1, 3, 2)
        # B 1 d_model 2T -> B 1 d_model T -> B T 1 d_model
        res = self.output_step_linear(res).permute(0, 3, 1, 2).repeat(1, 1, N, 1)
        res = self.dropout_layer(res)

        return res


class TemporalCrossRelationModule(nn.Module):
    def __init__(self, num_features, num_instances, d_model, hidden_dim, n_heads, history_steps, dropout,
                 nb_random_features=64, nb_gumbel_sample=64, tau=0.25, rb_order=1, rb_trans='sigmoid',
                 use_edge_loss=False):
        super(TemporalCrossRelationModule, self).__init__()

        self.num_features = num_features
        self.num_instances = num_instances
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.history_steps = history_steps
        self.dropout = dropout
        self.patch_len = 4
        self.stride = 2
        self.padding = 2

        self.nb_random_features = nb_random_features
        self.nb_gumbel_sample = nb_gumbel_sample
        self.tau = tau
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

        self.patch_embedding = PatchEmbedding(patch_len=self.patch_len,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dropout=self.dropout)
        self.patch_features_embedding = nn.Linear(4 * self.d_model, self.d_model)

        self.crossrelation = CrossCorrealtion(d_model=self.d_model,
                                              hidden_dim=self.hidden_dim,
                                              num_instances=self.num_instances,
                                              n_heads=self.n_heads,
                                              history_steps=self.history_steps,
                                              patch_len=self.patch_len,
                                              stride=self.stride,
                                              padding=self.padding,
                                              nb_random_features=self.nb_random_features,
                                              nb_gumbel_sample=self.nb_gumbel_sample,
                                              rb_order=self.rb_order,
                                              rb_trans=self.rb_trans,
                                              use_edge_loss=self.use_edge_loss)

        num_patch = int((self.history_steps + self.padding - self.patch_len) / self.stride + 1)

        self.flatten = FlattenPatch(num_patch, history_steps, dropout=0.1)

    def forward(self, x, adj):
        """

        :param x: B T N d_model
        :param adj:
        :return: B T N d_model
        """
        B, T, N, D = x.shape
        # B num_patch N d_model*patch_len -> # B num_patch N d_model patch_len
        x_patch = self.patch_embedding(x, if_keepdim=False).reshape(B, -1, N, self.d_model, self.patch_len)
        _, T_hat, _, d_model, patch_len = x_patch.shape
        x = x_patch.reshape(B, T_hat, N, -1)  # B T' N d_model*pat_len
        x = self.patch_features_embedding(x)  # B T' N d_model*pat_len -> B T' N d_model  2, 32, 56, 512
        if self.use_edge_loss:
            # B T' N d_model
            x, edge_loss = self.crossrelation(x, adj)
            x = self.flatten(x)
            return x, edge_loss
        else:
            x = self.crossrelation(x, adj)
            # print("x shape", x.shape)  # 2, 16, 56, 512
            x = self.flatten(x)
            return x


class TemporalMaskedSelfAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, dropout, mask_flag, factor, d_keys=None, d_values=None,
                 mix=False,
                 output_attention=False, num_instances=56, history_steps=16, predict_steps=16):
        super(TemporalMaskedSelfAttentionLayer, self).__init__()

        self.attention = attention
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.mask_flag = mask_flag
        self.factor = factor
        self.mix = mix
        self.output_attention = output_attention
        self.num_instances = num_instances
        self.history_steps = history_steps
        self.predict_steps = predict_steps

        self.d_keys = d_keys or (d_model // n_heads)
        self.d_values = d_values or (d_model // n_heads)

        if attention == "prob":
            self.self_attention = TemporalProbAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mask_flag=self.mask_flag,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention)
        else:
            self.self_attention = TemporalFullAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mask_flag=self.mask_flag,
                factor=self.factor,
                attention_dropout=self.dropout,
                output_attention=self.output_attention)

        # self.query_projection = nn.Linear(self.num_instances * self.d_model, self.d_keys * self.n_heads)
        # self.key_projection = nn.Linear(self.num_instances * self.d_model, self.d_keys * self.n_heads)
        # self.value_projection = nn.Linear(self.num_instances * self.d_model, self.d_values * self.n_heads)
        # self.out_projection = nn.Linear(self.d_values * self.n_heads, self.num_instances * self.d_model)

        self.query_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.key_projection = nn.Linear(self.d_model, self.d_keys * self.n_heads)
        self.value_projection = nn.Linear(self.d_model, self.d_values * self.n_heads)
        self.out_projection = nn.Linear(self.d_values * self.n_heads, self.d_model)

    def forward(self, queries, keys, values, attn_mask):
        """
        :param queries: B T N d_model
        :param keys: B T N d_model
        :param values: B T N d_model
        :param attn_mask: BN H T T Default None
        :return: B T N d_model, BN H T T
        """
        B, T, N, D = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)  # B T N d_model -> B T N self.d_keys * self.n_heads
        keys = self.key_projection(keys)  # B T N d_model -> B T N self.d_keys * self.n_heads
        values = self.value_projection(values)  # B T N d_model -> B T N self.d_values * self.n_heads

        queries = queries.permute(0, 2, 1, 3)  # B T N self.d_keys * self.n_heads -> B N T self.d_keys * self.n_heads
        keys = keys.permute(0, 2, 1, 3)  # B T N self.d_keys * self.n_heads -> B N T self.d_keys * self.n_heads
        values = values.permute(0, 2, 1, 3)  # B T N self.d_values * self.n_heads -> B N T self.d_values * self.n_heads

        queries = queries.reshape(-1, T, H, self.d_keys)  # B N T self.d_keys * self.n_heads -> BN T H d_keys
        keys = keys.reshape(-1, T, H, self.d_keys)  # B N T self.d_keys * self.n_heads -> BN T H d_keys
        values = values.reshape(-1, T, H, self.d_values)  # B N T self.d_values * self.n_heads -> BN T H d_values

        # queries = queries.reshape(B, T, -1)  # B T N d_model -> B T N*d_model
        # keys = keys.reshape(B, T, -1)  # B T N d_model -> B T N*d_model
        # values = values.reshape(B, T, -1)  # B T N d_model -> B T N*d_model
        #
        # # B T N*d_model -> B T d_keys * H -> B T H d_keys
        # queries = self.query_projection(queries).reshape(B, T, H, -1)
        # # B T S*d_model -> B T d_keys * H -> B T H d_keys
        # keys = self.key_projection(keys).reshape(B, T, H, -1)
        # # B T S*d_model -> B T d_values * H -> B T H d_values
        # values = self.value_projection(values).reshape(B, T, H, -1)

        # out BN T H d_value
        # attn BN H T T
        out, attn = self.self_attention(queries, keys, values, attn_mask)
        out = out.reshape(-1, T, H * self.d_values)  # BN T H d_value -> BN T H*d_value
        out = self.out_projection(out)  # BN T H*d_value -> BN T d_model
        # out = out.reshape(B, T, -1, self.d_model)  # B T num_instances * d_model -> B T N d_model
        out = out.reshape(B, -1, T, self.d_model).permute(0, 2, 1, 3)
        return out, attn  # B T N d_model, BN H T T


class WaveletInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(WaveletInceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):  # 0, 1, 2
            # for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TemporalWaveletInceptionModule(nn.Module):
    def __init__(self, d_model, history_steps, num_instances, kernels, dwt_level=3):
        super(TemporalWaveletInceptionModule, self).__init__()
        self.d_model = d_model
        self.history_steps = history_steps
        self.num_instances = num_instances
        self.kernels = kernels
        self.dwt_level = dwt_level
        self.cwt_dim = 512
        self.totalscal = 100

        self.dwt2d = DWT_2D(wavename='db8')

        self.idwt2d = IDWT_2D(wavename='db8')

        self.dim_reduction = nn.Linear(self.d_model, self.cwt_dim)
        self.dim_resconstruction = nn.Linear(self.cwt_dim, self.d_model)

        self.subwave_inception = WaveletInceptionBlock(in_channels=1,
                                                       out_channels=1,
                                                       num_kernels=self.kernels)

        # self.alpha = nn.Parameter(torch.rand(self.d_model, self.d_model), requires_grad=True)
        self.output_linear = nn.Linear(self.d_model, self.d_model)
        # self.LH_W = nn.ModuleList()
        # for i in range(self.dwt_level):
        #     self.LH_W.append(
        #         nn.Linear(self.d_model // math.pow(2, i+1), self.d_model // math.pow(2, i+1))
        #     )
        # self.HL_W = nn.ModuleList()
        # for i in range(self.dwt_level):
        #     self.LH_W.append(
        #         nn.Linear(self.d_model // math.pow(2, i + 1), self.d_model // math.pow(2, i + 1))
        #     )
        # self.HH_W = nn.ModuleList()
        # for i in range(self.dwt_level):
        #     self.LH_W.append(
        #         nn.Linear(self.d_model // math.pow(2, i + 1), self.d_model // math.pow(2, i + 1))
        #     )

    def forward(self, x, adj):
        B, T, N, D = x.shape
        # x_cwt = self.dim_reduction(x)  # B T N d_model -> B T N cwt_dim
        # B T N D -> B N T D
        x_dwt2d = x.permute(0, 2, 1, 3)
        x_dwt2d = x_dwt2d.reshape(-1, T, D).unsqueeze(1)  # B N T D -> BN 1 T D

        # wavefamlies = ["cgau6", "gaus6"]
        # totalscal = self.totalscal
        # module_start_time = time.time()
        # for b in range(B):
        #     for n in range(N):
        #         for d in range(self.cwt_dim):
        #             data = x_cwt[b, :, n, d].cpu().detach().numpy()
        #             # print(data)
        #             cw_list = []
        #             for wavename in wavefamlies:
        #                 # w = pywt.Wavelet(wavename)
        #                 fc = pywt.central_frequency(wavename)
        #                 cparam = 2 * fc * totalscal
        #                 scales = cparam / np.arange(totalscal, 1, -1)
        #                 coef, freqs = pywt.cwt(data=data, scales=scales, wavelet=wavename)  # 100,T 100
        #                 coef = torch.from_numpy(coef).float().to(x.device)
        #                 freqs = torch.from_numpy(freqs).unsqueeze(0).float().to(x.device)
        #                 coef_weights = torch.matmul(torch.abs(freqs), torch.softmax(coef, dim=0)).squeeze(0).to(x.device)  # 1 T
        #                 cw_list.append(coef_weights)
        #             cw = torch.stack(cw_list, dim=0).mean(dim=0)  # T
        #             x_cwt[b, :, n, d] = x_cwt[b, :, n, d] * cw
        # module_end_time = time.time()
        # print("wavelet of compute time is {}s".format(module_end_time - module_start_time))
        #
        # x_cwt = self.dim_resconstruction(x_cwt)  # B T N cwt_dim -> B T N d_model

        LL_dict = {}
        LH_dict = {}
        HL_dict = {}
        HH_dict = {}
        LL = x_dwt2d
        for i in range(self.dwt_level):
            LL, LH, HL, HH = self.dwt2d(LL)  # BN 1 T D -> BN 1 T/pow(2,i) D/pow(2,i)
            factor = torch.median(torch.abs(HH), dim=-1, keepdim=True)[0] / 0.6745
            threshold = factor * math.sqrt(2 * math.log(T))
            LH_soft = soft_thresholding(LH, threshold)
            HL_soft = soft_thresholding(HL, threshold)
            HH_soft = soft_thresholding(HH, threshold)
            LL = self.subwave_inception(LL)
            LH_soft = self.subwave_inception(LH_soft)
            HL_soft = self.subwave_inception(HL_soft)
            HH_soft = self.subwave_inception(HH_soft)

            # H_freq = LH_soft * self.LH_W[i] + HL_soft * self.HL_W[i] + HH_soft * self.HH_W[i]

            LL_dict[i] = LL
            LH_dict[i] = LH_soft
            HL_dict[i] = HL_soft
            HH_dict[i] = HH_soft

        if self.dwt_level >= 2:
            for i in range(self.dwt_level):
                LL = LL_dict[abs(self.dwt_level - 1 - i)]
                LH = LH_dict[abs(self.dwt_level - 1 - i)]
                HL = HL_dict[abs(self.dwt_level - 1 - i)]
                HH = HH_dict[abs(self.dwt_level - 1 - i)]
                LL = self.idwt2d(LL, LH, HL, HH)
                if self.dwt_level - 2 - i >= 0:
                    LL_dict[abs(self.dwt_level - 2 - i)] = LL
                else:
                    break
        else:
            LL = LL_dict[0]
            LH = LH_dict[0]
            HL = HL_dict[0]
            HH = HH_dict[0]
            LL = self.idwt2d(LL, LH, HL, HH)

        x_dwt2d = LL
        # B N T D -> BN T 1 D
        x_dwt2d = x_dwt2d.permute(0, 2, 1, 3)

        # x_res = torch.matmul(x_cwt, self.alpha) + torch.matmul(x_dwt2d, (1 - self.alpha))
        # x_res = torch.matmul(x_dwt2d, self.alpha)
        x_res = self.output_linear(x_dwt2d)
        x_res = x_res.squeeze(2).reshape(B, -1, T, D).permute(0, 2, 1, 3)
        return x_res
