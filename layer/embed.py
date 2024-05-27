import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """

        :param x: B T N C
        :return:
        """
        B, T, N, C = x.shape
        # padding on the both ends of time series
        x = x.reshape(B, T, -1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        x = self.avg(x.permute(0, 2, 1))  # B T NC -> B NC T
        x = x.reshape(B, N, -1, T).permute(0, 3, 1, 2)  # B N C T -> B T N C
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """

        :param x: B T N C
        :return:
        """
        moving_mean = self.moving_avg(x)  # B T N C
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size_list):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size_list
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size_list]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class TokenEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=num_features, out_channels=embed_dim,
                                   kernel_size=3, padding=padding, padding_mode='zeros')
        # self.tokenLinear = nn.Linear(in_features = num_features, out_features = embed_dim, bias = True)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.0003)

    def forward(self, x):
        """
        :param x: B T N num_features
        :return: B T N embed_dim
        """
        B, T, N, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # B T N num_features -> B T N embed_dim
        x = x.reshape(B, T, N, -1)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=15000):
        """
        :param embed_dim:
        :param max_len:
        """
        super(PositionalEmbedding, self).__init__()

        self.embed_dim = embed_dim
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, embed_dim).float()
        self.pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # max_len 1
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()  # embed_dim/2

        self.pe[:, 0::2] = torch.sin(position * div_term)  # max_len embed_dim/2
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0)  # 1 max_len embed_dim
        # self.register_buffer('pe', self.pe)

    def forward(self, x):
        """
        :param x: B T N num_features
        :return: B T N embed_dim
        """
        B, T, N, C = x.shape
        reshape_x = x.reshape(B, -1, C)  # B T N num_features -> B T*N num_features
        out = self.pe[:, :reshape_x.size(1)].repeat(B, 1, 1)  # B T*N embed_dim
        out = out.reshape(B, T, N, self.embed_dim)  # B T*N embed_dim -> B T N embed_dim
        return out


class SpatialEmbedding(nn.Module):
    def __init__(self, eigenmaps_k, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.linear = nn.Linear(eigenmaps_k, embed_dim)

    def forward(self, eigenmaps):
        spatial_embedding = self.linear(eigenmaps)

        return spatial_embedding


class FixedEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(num_features, embed_dim).float()  # num_features embed_dim
        w.require_grad = False

        position = torch.arange(0, num_features).float().unsqueeze(1)  # num_features 1
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()  # embed_dim/2

        w[:, 0::2] = torch.sin(position * div_term)  # num_features embed_dim/2
        w[:, 1::2] = torch.cos(position * div_term)  # num_features embed_dim/2

        self.emb = nn.Embedding(num_features, embed_dim)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, embed_dim)

    def forward(self, x):
        return self.embed(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, embed_dim)
        self.hour_embed = Embed(hour_size, embed_dim)
        self.weekday_embed = Embed(weekday_size, embed_dim)
        self.day_embed = Embed(day_size, embed_dim)
        self.month_embed = Embed(month_size, embed_dim)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(num_features=num_features, embed_dim=embed_dim)
        self.position_embedding = PositionalEmbedding(embed_dim=embed_dim)
        self.temporal_embedding = TemporalEmbedding(embed_dim=embed_dim, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(embed_dim=embed_dim, embed_type=embed_type,
                                                               freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):  # 原文timesnet x_mark是关于时间特征的序列
        if x_mark is not None:
            x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        else:
            x = self.value_embedding(x) + self.position_embedding(x).to(x.device)

        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len=16, stride=8, padding=8, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # self.padding_patch_layer = nn.ReplicationPad1d((int(padding / 2), int(padding / 2)))
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))


        # self.patch_features_embedding = nn.Linear(self.num_features * self.patch_len, self.d_model)
        # # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, if_keepdim):
        """

        :param x: B T N C
        :return:
        """
        # do patching
        B, T, N, C = x.shape
        x = x.reshape(B, T, -1).permute(0, 2, 1)  # B T N C -> B T NC -> B NC T
        x = self.padding_patch_layer(x)  # B NC T -> B NC L(T+padding)

        if if_keepdim:
            x = x.unfold(dimension=-1, size=self.patch_len, step=1)  # B NC L -> B NC T patch_len
        else:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # B NC L -> B NC num_patch patch_len

        x = x.permute(0, 2, 1, 3)  # B NC num_patch patch_len -> B num_patch NC patch_len
        # B num_patch NC patch_len -> B num_patch N C*patch_len
        x = x.reshape(B, -1, N, C * self.patch_len)  # B num_patch N C*patch_len

        return self.dropout(x)
