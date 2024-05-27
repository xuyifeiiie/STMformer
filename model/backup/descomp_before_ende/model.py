import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.STMFormer.encoder import Encoder
from model.STMFormer.decoder import Decoder
from layer.embed import DataEmbedding, PatchEmbedding, series_decomp, series_decomp_multi


class STMFormer(nn.Module):
    def __init__(self, settings, device=torch.device('cuda:0')):
        super(STMFormer, self).__init__()

        # self.global_adj = global_adj
        self.num_features = settings.num_features
        self.embed_dim = settings.embed_dim
        self.en_in_dim = settings.en_in_dim
        self.de_in_dim = settings.de_in_dim
        self.out_dims = settings.out_dims
        self.hidden_dim = settings.hidden_dim
        self.d_model = settings.d_model
        self.after_d_model = settings.after_d_model
        self.n_heads = settings.n_heads
        self.num_instances = settings.num_instances
        self.num_attention_layers = settings.num_attention_layers
        self.num_en_layers = settings.num_en_layers
        self.num_de_layers = settings.num_de_layers
        self.factor = settings.factor
        self.kernels = settings.kernels
        self.dropout = settings.dropout
        self.attention = settings.attention
        self.embed = settings.embed
        self.freq = settings.freq
        self.activation = settings.activation
        self.use_mask = settings.use_mask
        self.if_output_attention = settings.if_output_attention
        self.co_train = settings.co_train
        self.if_patch = settings.if_patch
        self.distil = settings.distil
        self.mix = settings.mix
        self.strides = settings.strides
        self.history_steps = settings.history_steps
        self.predict_steps = settings.predict_steps
        self.labels_steps = settings.history_steps // 2
        self.decomp = settings.decomp
        self.decomp_kernels = settings.decomp_kernels
        self.multidecomp = settings.multidecomp
        self.multidecomp_kernels = settings.multidecomp_kernels
        self.patch_len = int(self.history_steps / 4)
        self.device = device
        self.if_wavelet = settings.if_wavelet

        if self.decomp:
            self.series_decomp = series_decomp(self.decomp_kernels)
        elif self.multidecomp:
            self.series_decomp = series_decomp_multi(self.multidecomp_kernels)

        self.trend_projection = nn.Linear(self.num_features, self.d_model)

        # Encoding
        self.data_embedding = DataEmbedding(self.num_features, self.embed_dim, self.embed, self.freq, self.dropout)

        self.patch_embedding = PatchEmbedding(3, 1, 2, self.dropout)
        self.patch_features_embedding = nn.Linear(self.embed_dim * 3, self.embed_dim)

        self.input_linear = nn.Linear(self.embed_dim, self.d_model)

        # Encoder
        self.encoder = Encoder(settings=settings, device=self.device)

        self.distil_linear = nn.Linear(self.out_dims[self.num_en_layers], self.d_model)

        if self.activation == 'GELU':
            self.activation_layer = nn.GELU()
        else:
            self.activation_layer = nn.LeakyReLU(0.1)

        # Decoder
        self.decoder = Decoder(settings=settings, device=device)

        # Output
        self.forecast_layer = nn.Linear(self.d_model, self.num_features)
        if self.co_train:
            self.label_layer = nn.Linear(self.d_model, 2)

    def forward(self, x, y, x_mark_enc, x_mark_dec, adj_in, adj_out, instances_index_vm,
                mask=None):
        """

        :param x: B T N F
        :param y: B T N F
        :param x_mark_enc:
        :param x_mark_dec:
        :param adj_in: B T N N
        :param adj_out: B T N N
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :param mask:
        :return: B T N F, optional[Tensor(B T N 1), None]
        """
        label_output = None
        x_mark = None
        y_mark = None

        if self.decomp or self.multidecomp:
            mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.predict_steps, 1, 1)
            seasonal_init, trend_init = self.series_decomp(x)  # x - moving_avg, moving_avg
            # decoder input
            trend_init = torch.cat([trend_init[:, -self.labels_steps:, :, :], mean[:, :self.labels_steps, :, :]],
                                   dim=1)  # B label_len+0.5pred_len Cin
            trend_init = self.trend_projection(trend_init)
            # (0, 0, 0, self.pred_len)前两位表示左右填充数 后两位表示上下填充数
            seasonal_init = seasonal_init.reshape(x.shape[0], x.shape[1], -1)
            seasonal_init = F.pad(seasonal_init[:, -self.labels_steps:, :], (0, 0, 0, self.labels_steps))
            seasonal_init = seasonal_init.reshape(x.shape[0], x.shape[1], -1, x.shape[3])

            x = self.data_embedding(x, x_mark)  # B T N F -> B T N embed_dim
            y = self.data_embedding(seasonal_init, y_mark)  # B T N F -> B T N embed_dim

        else:
            x = self.data_embedding(x, x_mark)  # B T N F -> B T N embed_dim
            y = self.data_embedding(x, y_mark)  # B T N F -> B T N embed_dim
            trend_init = None

        if self.if_patch:
            x = self.patch_embedding(x, if_keepdim=True)  # B num_patch N C*patch_len
            y = self.patch_embedding(y, if_keepdim=True)  # B num_patch N C*patch_len  8, 64, 56, 1536
            x = self.patch_features_embedding(x)  # B num_patch N C*patch_len -> B num_patch N d_model
            y = self.patch_features_embedding(y)  # B num_patch N C*patch_len -> B num_patch N d_model

        x = self.input_linear(x)  # B T N embed_dim -> B T N en_in_dim(d_model)

        module_start_time = time.time()
        # B T N en_in_dim(d_model) -> B T N d_model
        en_output, en_vm_attention_outer_list = self.encoder(x, adj_in, instances_index_vm)
        module_end_time = time.time()
        print("encoder compute time is {}".format(module_end_time - module_start_time))

        if self.distil:
            # B T N out_dims[-1] -> B T N d_model
            en_output = self.distil_linear(en_output)

        y = self.input_linear(y)  # B T N embed_dim -> B T N de_in_dim(d_model)


        module_start_time = time.time()
        # B T N d_model, B T N de_in_dim(d_model) -> B T N d_model
        # de_output, _ = self.decoder(en_output, y, en_vm_attention_outer_list, adj_out, instances_index_vm)
        de_output, _ = self.decoder(en_output, y, en_vm_attention_outer_list, adj_in, instances_index_vm, trend_init)
        module_end_time = time.time()
        print("decoder compute time is {}".format(module_end_time - module_start_time))

        forecast_output = self.forecast_layer(de_output)  # B T N d_model -> B T N num_features

        if self.co_train:
            label_output = self.label_layer(de_output)
            return forecast_output, label_output
        else:
            return forecast_output, None

