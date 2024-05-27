import time

import torch
import torch.nn as nn

from layer.attention import AttentionLayer
from layer.common import FeedForwardLayer, NormLayer
from layer.embed import series_decomp, series_decomp_multi
from layer.infrastructure import VMAttentionModule
from layer.spatial import SpatialRelationModule
from layer.temporal import TemporalRelationModule, TemporalMaskedSelfAttentionLayer, TemporalCrossRelationModule, \
    TemporalWaveletInceptionModule
from utils.masking import TimeSeriesMask


class DecoderLayer(nn.Module):
    def __init__(self, settings, device=torch.device('cuda:0')):
        super(DecoderLayer, self).__init__()

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
        self.device = device
        self.if_wavelet = settings.if_wavelet

        if self.decomp:
            self.series_decomp = series_decomp(self.decomp_kernels)
        elif self.multidecomp:
            self.series_decomp = series_decomp_multi(self.multidecomp_kernels)

        self.adj_norm_layer = nn.LayerNorm(self.num_instances)

        # self.masked_temporal_self_attention = TemporalMaskedSelfAttentionLayer(attention=self.attention,
        #                                                                        d_model=self.d_model,
        #                                                                        n_heads=self.n_heads,
        #                                                                        dropout=self.dropout,
        #                                                                        mask_flag=True,
        #                                                                        factor=self.factor,
        #                                                                        output_attention=self.if_output_attention,
        #                                                                        num_instances=self.num_instances)
        #
        self.self_attention = AttentionLayer(attention=self.attention,
                                             d_model=self.d_model,
                                             n_heads=self.n_heads,
                                             dropout=self.dropout,
                                             mask_flag=True,
                                             factor=self.factor,
                                             output_attention=False)

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

        self.temporal_cross_message_modules = nn.ModuleList(TemporalCrossRelationModule(num_features=self.num_features,
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

        self.alpha = nn.Parameter(torch.rand(self.d_model, self.d_model), requires_grad=True)

        self.norm_layer = NormLayer(self.history_steps, self.d_model)

        self.trend_layer0 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer1 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer2 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer3 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer4 = nn.Linear(self.d_model, self.d_model, bias=False)
        self.trend_layer5 = nn.Linear(self.d_model, self.d_model, bias=False)

        self.feedforward = FeedForwardLayer(self.d_model, self.after_d_model)

        self.layernorm = nn.LayerNorm(self.after_d_model)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, y, en_vm_attention_outer_list, adj, instances_index_vm, attn_mask=None):
        """
        :param x: B T N d_model  output of encoder
        :param y: B T N d_model
        :param en_vm_attention_outer_list:
        :param adj:
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :param attn_mask: None
        :param trend_init: trend_init
        :return: B T N d_model
        """
        adj = self.adj_norm_layer(adj)

        y_res = y
        y_raw = y
        y_trend_fin = torch.zeros_like(y).to(y.device)

        # y_res, _ = self.masked_temporal_self_attention(y_res, y_res, y_res, attn_mask)
        # y_res = self.norm_layer(y_raw, y_res)
        #
        # if self.decomp:
        #     y_res, y_trend_part = self.series_decomp(y_res)
        # elif self.multidecomp:
        #     y_res, y_trend_part = self.series_decomp(y_res)
        # else:
        #     y_res = y_res
        #     y_trend_part = torch.zeros_like(y_res)
        #
        # y_trend_part = self.trend_layer1(y_trend_part)
        # y_trend = y_trend + y_trend_part
        # y_raw = y_res
        #
        y_res, _ = self.self_attention(y_res, x, x, attn_mask)
        y_res = self.norm_layer(y_raw, y_res)

        if self.decomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        elif self.multidecomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        else:
            y_res = y_res
            y_trend_part = torch.zeros_like(y_res)
        y_trend_part = self.trend_layer2(y_trend_part)
        y_trend_fin += y_trend_part
        y_raw = y_res

        vm_attention_list = []
        for vm_attention_layer in self.vm_attention_modules:
            # x:B T N d_model attns:[B T H N1 N1,...],List
            y_res, vm_attention = vm_attention_layer(y_res, instances_index_vm, attn_mask)
            vm_attention_list.append(vm_attention)

        y_res = self.norm_layer(y_raw, y_res)

        if self.decomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        elif self.multidecomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        else:
            y_res = y_res
            y_trend_part = torch.zeros_like(y_res)

        y_trend_part = self.trend_layer3(y_trend_part)
        y_trend_fin += y_trend_part
        y_raw = y_res

        for spatial_message_layer in self.spatial_message_modules:
            y_res = spatial_message_layer(y_res, adj)
        y_res = self.norm_layer(y_raw, y_res)

        if self.decomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        elif self.multidecomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        else:
            y_res = y_res
            y_trend_part = torch.zeros_like(y_res)

        y_trend_part = self.trend_layer4(y_trend_part)
        y_trend_fin += y_trend_part
        y_raw = y_res

        # module_start_time = time.time()
        self_res = y_res
        for temporal_layer in self.temporal_self_message_modules:
            self_res = temporal_layer(self_res, adj)
        # module_end_time = time.time()
        # print("temporal_message_modules of decoder compute time is {}".format(module_end_time - module_start_time))

        cross_res = y_res
        for temporal_layer in self.temporal_cross_message_modules:
            cross_res = temporal_layer(cross_res, adj)

        y_res = torch.matmul(self_res, self.alpha) + torch.matmul(cross_res, (1 - self.alpha))
        y_res = self.norm_layer(y_raw, y_res)

        if self.decomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        elif self.multidecomp:
            y_res, y_trend_part = self.series_decomp(y_res)
        else:
            y_res = y_res
            y_trend_part = torch.zeros_like(y_res)

        y_trend_part = self.trend_layer5(y_trend_part)
        y_trend_fin += y_trend_part

        y_res = y_res + y_trend_fin
        y_raw = y_res

        y_res = self.feedforward(y_res)
        y_res = self.norm_layer(y_raw, y_res)

        y = y_res
        # y_trend = y_trend + trend_init

        y = self.dropout_layer(y)
        # y_trend = self.dropout_layer(y_trend)

        # return y, y_trend, vm_attention_list
        return y, vm_attention_list


class Decoder(nn.Module):
    def __init__(self, settings, device=torch.device('cuda:0')):
        super(Decoder, self).__init__()
        self.num_de_layers = settings.num_de_layers
        self.dropout = settings.dropout
        self.device = device

        self.decoder_layer_stack = nn.ModuleList(
            [DecoderLayer(settings, device=device) for _ in range(self.num_de_layers)]
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, enc_ouput, y, en_vm_attention_outer_list, adj, instances_index_vm):
        """
        :param x: B T N d_model
        :param y: B T N de_in_dim
        :param en_vm_attention_outer_list:
        :param adj:
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :return: B T N d_model
        """

        de_vm_attention_outer_list = []
        x = torch.zeros_like(y).to(y.device)
        for decoder_layer in self.decoder_layer_stack:
            # x, x_trend, vm_attention_list = decoder_layer(enc_ouput, y, en_vm_attention_outer_list, adj, instances_index_vm, trend_init)
            x, vm_attention_list = decoder_layer(enc_ouput, y, en_vm_attention_outer_list, adj, instances_index_vm)
            x = self.dropout_layer(x)
            # x_trend = self.dropout_layer(x_trend)
            # trend_init = x_trend + trend_init
            de_vm_attention_outer_list.append(vm_attention_list)

        return x, de_vm_attention_outer_list
