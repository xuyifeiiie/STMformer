import time

import torch
import torch.nn as nn
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
        x = self.data_embedding(x, x_mark)  # B T N F -> B T N embed_dim
        y = self.data_embedding(y, y_mark)  # B T N F -> B T N embed_dim

        if self.if_patch:
            x = self.patch_embedding(x, if_keepdim=True)  # B num_patch N C*patch_len
            y = self.patch_embedding(y, if_keepdim=True)  # B num_patch N C*patch_len  8, 64, 56, 1536
            x = self.patch_features_embedding(x)  # B num_patch N C*patch_len -> B num_patch N d_model
            y = self.patch_features_embedding(y)  # B num_patch N C*patch_len -> B num_patch N d_model
            # x_patch = self.patch_embedding(x, if_keepdim=False)
            # y_patch = self.patch_embedding(y, if_keepdim=False)

        if self.decomp or self.multidecomp:
            if self.decomp:
                x_res, x_trend = self.series_decomp(x)
            elif self.multidecomp:
                x_res, x_trend = self.series_decomp(x)

            x_res = self.input_linear(x_res)  # B T N embed_dim -> B T N en_in_dim(d_model)
            x_trend = self.input_linear(x_trend)

            module_start_time = time.time()
            # B T N en_in_dim(d_model) -> B T N d_model(out_dims[-1])
            en_output_res, en_vm_attention_outer_list1 = self.encoder(x_res, adj_in, instances_index_vm)
            module_end_time = time.time()
            print("encoder compute time is {}".format(module_end_time - module_start_time))

            en_output_res = self.activation_layer(en_output_res)

            if self.distil:
                # B T N out_dims[-1] -> B T N d_model
                en_output_res = self.distil_linear(en_output_res)

            en_output = en_output_res + x_trend

            if self.decomp:
                y_res, y_trend = self.series_decomp(y)
            elif self.multidecomp:
                y_res, y_trend = self.series_decomp(y)

            y_res = self.input_linear(y_res)  # B T N embed_dim -> B T N en_in_dim(d_model)
            y_trend = self.input_linear(y_trend)

            module_start_time = time.time()
            # B T N d_model, B T N de_in_dim(d_model) -> B T N d_model
            de_output_res, _ = self.decoder(en_output, y_res, en_vm_attention_outer_list1, adj_out, instances_index_vm)
            module_end_time = time.time()
            print("decoder compute time is {}".format(module_end_time - module_start_time))

            de_output_res = self.activation_layer(de_output_res)

            de_output = de_output_res + y_trend

            forecast_output = self.forecast_layer(de_output)  # B T N d_model -> B T N num_features

            if self.co_train:
                label_output = self.label_layer(de_output)
                label_output = torch.softmax(label_output, dim=-1)
                return forecast_output, label_output
            else:
                return forecast_output, None

        else:
            x = self.input_linear(x)  # B T N embed_dim -> B T N en_in_dim(d_model)

            module_start_time = time.time()
            # B T N en_in_dim(d_model) -> B T N d_model
            en_output, en_vm_attention_outer_list = self.encoder(x, adj_in, instances_index_vm)
            module_end_time = time.time()
            print("encoder compute time is {}".format(module_end_time - module_start_time))

            if self.distil:
                # B T N out_dims[-1] -> B T N d_model
                en_output = self.distil_linear(en_output)

            # y = self.data_embedding(y, y_mark)  # B T N F -> B T N embed_dim

            y = self.input_linear(y)  # B T N embed_dim -> B T N de_in_dim(d_model)

            module_start_time = time.time()
            # B T N d_model, B T N de_in_dim(d_model) -> B T N d_model
            de_output, _ = self.decoder(en_output, y, en_vm_attention_outer_list, adj_out, instances_index_vm)
            module_end_time = time.time()
            print("decoder compute time is {}".format(module_end_time - module_start_time))

            forecast_output = self.forecast_layer(de_output)  # B T N d_model -> B T N num_features

            if self.co_train:
                label_output = self.label_layer(de_output)
                return forecast_output, label_output
            else:
                return forecast_output, None

