import time

import torch
import torch.nn as nn
from layer.attention import AttentionLayer
from layer.common import FeedForwardLayer, NormLayer, DistilFeedForwardLayer
from layer.embed import series_decomp, series_decomp_multi
from layer.infrastructure import VMAttentionModule
from layer.spatial import SpatialRelationModule
from layer.temporal import TemporalRelationModule, TemporalCrossRelationModule, TemporalWaveletInceptionModule


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')

        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        """

        :param x: B T N c_in
        :return:
        """
        B, T, N, c_in = x.shape
        x = x.reshape(B, -1, c_in)
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        x = x.reshape(B, T, -1, c_in)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, settings, layer_num, device=torch.device('cuda:0')):
        super(EncoderLayer, self).__init__()

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
        self.nb_random_features = settings.nb_random_features
        self.nb_gumbel_sample = settings.nb_gumbel_sample
        self.tau = settings.tau
        self.rb_order = settings.rb_order
        self.rb_trans = settings.rb_trans
        self.use_edge_loss = settings.use_edge_loss
        self.dropout = settings.dropout
        self.attention = settings.attention
        self.embed = settings.embed
        self.freq = settings.freq
        self.activation = settings.activation
        self.use_mask = settings.use_mask
        self.if_output_attention = settings.if_output_attention
        self.co_train = settings.co_train
        self.distil = settings.distil
        self.mix = settings.mix
        self.strides = settings.strides
        self.history_steps = settings.history_steps
        self.predict_steps = settings.predict_steps
        self.decomp = settings.decomp
        self.decomp_kernels = settings.decomp_kernels
        self.multidecomp = settings.multidecomp
        self.multidecomp_kernels = settings.multidecomp_kernels
        self.decomp_stride = settings.decomp_stride
        self.layer_num = layer_num
        self.device = device
        self.if_wavelet = settings.if_wavelet

        if self.distil:
            self.d_model = self.out_dims[self.layer_num]
            self.after_d_model = self.out_dims[self.layer_num + 1]

        if self.decomp:
            self.series_decomp = series_decomp(self.decomp_kernels)
        elif self.multidecomp:
            self.series_decomp = series_decomp_multi(self.multidecomp_kernels)

        self.adj_norm_layer = nn.LayerNorm(self.num_instances)

        self.vm_attention_modules = nn.ModuleList([VMAttentionModule(num_instances=self.num_instances,
                                                                     attention=None,
                                                                     d_model=self.d_model,
                                                                     n_heads=self.n_heads,
                                                                     factor=self.factor,
                                                                     dropout=self.dropout,
                                                                     history_steps=self.history_steps,
                                                                     output_attention=False,
                                                                     activation="relu",
                                                                     mix=False,
                                                                     mask_flag=False)
                                                   for _ in range(self.num_attention_layers)])

        self.spatial_message_modules = nn.ModuleList(SpatialRelationModule(d_model=self.d_model,
                                                                           hidden_dim=self.hidden_dim,
                                                                           alpha=0.2,
                                                                           n_heads=self.n_heads,
                                                                           dropout=self.dropout,
                                                                           history_steps=self.history_steps)
                                                     for _ in range(self.num_attention_layers))

        if self.if_wavelet:
            self.temporal_self_message_modules = nn.ModuleList(TemporalWaveletInceptionModule(d_model=self.d_model,
                                                                                              history_steps=self.history_steps,
                                                                                              num_instances=self.num_instances,
                                                                                              kernels=self.kernels,
                                                                                              dwt_level=2)
                                                               for _ in range(self.num_attention_layers))
        else:
            self.temporal_self_message_modules = nn.ModuleList(TemporalRelationModule(d_model=self.d_model,
                                                                                      num_features=self.num_features,
                                                                                      hidden_dim=self.hidden_dim,
                                                                                      kernels=self.kernels,
                                                                                      dropout=self.dropout,
                                                                                      top_k=self.factor,
                                                                                      history_steps=self.history_steps,
                                                                                      predict_steps=self.predict_steps)
                                                               for _ in range(self.num_attention_layers))

        self.temporal_cross_message_modules = nn.ModuleList(
            TemporalCrossRelationModule(num_features=self.num_features,
                                        num_instances=self.num_instances,
                                        d_model=self.d_model,
                                        hidden_dim=self.hidden_dim,
                                        n_heads=self.n_heads,
                                        history_steps=self.history_steps,
                                        dropout=self.dropout,
                                        nb_random_features=self.nb_random_features,
                                        nb_gumbel_sample=self.nb_gumbel_sample,
                                        tau=self.tau,
                                        rb_order=self.rb_order,
                                        rb_trans=self.rb_trans,
                                        use_edge_loss=self.use_edge_loss
                                        )
            for _ in range(self.num_attention_layers))

        # self.alpha = nn.Linear(self.d_model, self.d_model, bias=False)
        self.alpha = nn.Parameter(torch.rand(self.d_model, self.d_model), requires_grad=True)

        # self.norm_layer = NormLayer(self.history_steps)
        self.norm_layer = NormLayer(self.history_steps, self.d_model)

        self.trend_layer0 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer1 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer3 = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.distil:
            self.feedforward = DistilFeedForwardLayer(self.d_model, self.after_d_model)
            self.trend_feedforward = DistilFeedForwardLayer(self.d_model, self.after_d_model)
        else:
            self.feedforward = FeedForwardLayer(self.d_model, self.after_d_model)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj, instances_index_vm, attn_mask=None):
        adj = self.adj_norm_layer(adj)
        if self.decomp:
            x_res, x_trend = self.series_decomp(x)
        elif self.multidecomp:
            x_res, x_trend = self.series_decomp(x)
        else:
            x_res = x
            x_trend = torch.zeros_like(x_res)

        x_trend = self.trend_layer0(x_trend)
        x_raw = x_res

        vm_attention_list = []
        for vm_attention_layer in self.vm_attention_modules:
            # x:B T N d_model attns:[B T H N1 N1,...],List
            x_res, vm_attention = vm_attention_layer(x_res, instances_index_vm, attn_mask)
            vm_attention_list.append(vm_attention)

        x_res = self.norm_layer(x_raw, x_res)

        if self.decomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        elif self.multidecomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        else:
            x_res = x_res
            x_trend1 = torch.zeros_like(x_res)

        x_trend1 = self.trend_layer1(x_trend1)
        x_trend = x_trend + x_trend1
        x_raw = x_res

        # module_start_time = time.time()
        for spatial_message_layer in self.spatial_message_modules:
            x_res = spatial_message_layer(x_res, adj)
        # module_end_time = time.time()
        # print("spatial_message_modules of encoder compute time is {}".format(module_end_time - module_start_time))

        x_res = self.norm_layer(x_raw, x_res)

        if self.decomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        elif self.multidecomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        else:
            x_res = x_res
            x_trend1 = torch.zeros_like(x_res)

        x_trend1 = self.trend_layer2(x_trend1)
        x_trend = x_trend + x_trend1
        x_raw = x_res

        # module_start_time = time.time()
        self_res = x_res
        for temporal_layer in self.temporal_self_message_modules:
            self_res = temporal_layer(self_res, adj)
        # x_res = self_res
        # module_end_time = time.time()
        # print("temporal_self_message_modules of encoder compute time is {}".format(module_end_time - module_start_time))

        cross_res = x_res
        for temporal_layer in self.temporal_cross_message_modules:
            cross_res = temporal_layer(cross_res, adj)
        # x_res = cross_res

        # x_res = self.alpha(self_res) + self.alpha(cross_res)
        x_res = torch.matmul(self_res, self.alpha) + torch.matmul(cross_res, (1 - self.alpha))

        x_res = self.norm_layer(x_raw, x_res)

        if self.decomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        elif self.multidecomp:
            x_res, x_trend1 = self.series_decomp(x_res)
        else:
            x_res = x_res
            x_trend1 = torch.zeros_like(x_res)

        x_trend1 = self.trend_layer3(x_trend1)
        x_trend = x_trend + x_trend1
        x_raw = x_res

        x_res = self.feedforward(x_res)

        if not self.distil:
            x_res = self.norm_layer(x_raw, x_res)
            x = x_res + x_trend
        else:
            x_trend = self.trend_feedforward(x_trend)
            x = x_res + x_trend

        x = self.dropout_layer(x)

        return x, vm_attention_list


class Encoder(nn.Module):
    def __init__(self, settings, device=torch.device('cuda:0')):
        super(Encoder, self).__init__()
        self.num_en_layers = settings.num_en_layers
        self.out_dims = settings.out_dims
        self.dropout = settings.dropout
        self.distil = settings.distil
        self.device = device

        if not self.distil:
            self.encoder_layer_stack = nn.ModuleList([EncoderLayer(settings=settings, layer_num=i, device=device)
                                                      for i in range(self.num_en_layers)])
        else:
            self.encoder_layer_stack = nn.ModuleList([EncoderLayer(settings=settings, layer_num=i, device=device)
                                                      for i in range(self.num_en_layers)])
            self.encoder_layer_conv_stack = nn.ModuleList([ConvLayer(c_in=self.out_dims[i + 1])
                                                           for i in range(self.num_en_layers)])

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj_in, instances_index_vm):
        """
        :param x: B T N embed_dim(d_model)
        :param adj_in:
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :return: B T N d_model
        """
        en_vm_attention_outer_list = []
        if not self.distil:
            for encoder_layer in self.encoder_layer_stack:
                x, vm_attention_list = encoder_layer(x, adj_in, instances_index_vm)
                x = self.dropout_layer(x)
                en_vm_attention_outer_list.append(vm_attention_list)
        else:
            for encoder_layer, encoder_layer_conv in zip(self.encoder_layer_stack, self.encoder_layer_conv_stack):
                x, vm_attention_list = encoder_layer(x, adj_in, instances_index_vm)
                x = self.dropout_layer(x)
                en_vm_attention_outer_list.append(vm_attention_list)
                x = encoder_layer_conv(x)
                x = self.dropout_layer(x)
        return x, en_vm_attention_outer_list
