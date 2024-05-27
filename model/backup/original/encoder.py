import torch
import torch.nn as nn
from layer.attention import AttentionLayer
from layer.common import FeedForwardLayer, NormLayer
from layer.infrastructure import VMAttentionModule
from layer.spatial import SpatialRelationModule
from layer.temporal import TemporalRelationModule


class EncoderLayer(nn.Module):
    def __init__(self, num_features, embed_dim, en_in_dim, de_in_dim, out_dim, hidden_dim, d_model=512,
                 after_d_model=128, n_heads=16, num_of_instances=56, num_attention_layers=2, num_en_layers=6,
                 num_de_layers=6, factor=10, kernels=12, dropout=0.1, attention='prob', embed='fixed', freq='h',
                 activation='gelu', use_mask=True, if_output_attention=True, co_train=False, distil=False, mix=True,
                 history_steps=16, predict_steps=16, strides=16, device=torch.device('cuda:0')):
        super(EncoderLayer, self).__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim
        self.en_in_dim = en_in_dim
        self.de_in_dim = de_in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.after_d_model = after_d_model
        self.n_heads = n_heads
        self.num_of_instances = num_of_instances
        self.num_attention_layers = num_attention_layers
        self.num_en_layers = num_en_layers
        self.num_de_layers = num_de_layers
        self.factor = factor
        self.kernels = kernels
        self.dropout = dropout
        self.attention = attention
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.use_mask = use_mask
        self.if_output_attention = if_output_attention
        self.co_train = co_train
        self.distil = distil
        self.mix = mix
        self.strides = strides
        self.history_steps = history_steps
        self.predict_steps = predict_steps
        self.device = device

        # self.self_attention = AttentionLayer(attention=self.attention,
        #                                      d_model=self.d_model,
        #                                      n_heads=self.n_heads,
        #                                      dropout=dropout,
        #                                      mask_flag=False,
        #                                      factor=self.factor,
        #                                      output_attention=False)

        self.vm_attention_modules = nn.ModuleList([VMAttentionModule(num_of_instances=self.num_of_instances,
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

        self.temporal_message_modules = nn.ModuleList(TemporalRelationModule(d_model=self.d_model,
                                                                             hidden_dim=self.hidden_dim,
                                                                             kernels=self.kernels,
                                                                             dropout=self.dropout,
                                                                             top_k=self.factor,
                                                                             history_steps=self.history_steps,
                                                                             predict_steps=self.predict_steps)
                                                      for _ in range(self.num_attention_layers))

        self.norm_layer = NormLayer(self.history_steps)

        self.feedforward = FeedForwardLayer(self.d_model, self.after_d_model)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj_in, instances_index_vm, attention_mask=None):
        x1 = x
        x1_raw = x1

        vm_attention_list = []
        for vm_attention_layer in self.vm_attention_modules:
            # x:B T N d_model attns:[B T H N1 N1,...],List
            x1, vm_attention = vm_attention_layer(x1, instances_index_vm, attention_mask)
            vm_attention_list.append(vm_attention)

        x1 = self.norm_layer(x1_raw, x1)
        x1_raw = x1

        for spatial_message_layer in self.spatial_message_modules:
            x1 = spatial_message_layer(x1, adj_in)

        x1 = self.norm_layer(x1_raw, x1)
        x1_raw = x1

        for temporal_message_layer in self.temporal_message_modules:
            x1 = temporal_message_layer(x1)

        x1 = self.norm_layer(x1_raw, x1)
        x1_raw = x1

        x1 = self.feedforward(x1)

        x1 = self.norm_layer(x1_raw, x1)

        x1 = self.dropout_layer(x1)

        return x1, vm_attention_list


class Encoder(nn.Module):
    def __init__(self, num_features, embed_dim, en_in_dim, de_in_dim, out_dim, hidden_dim, d_model=512,
                 after_d_model=128, n_heads=16, num_of_instances=56, num_attention_layers=2, num_en_layers=6,
                 num_de_layers=6, kernels=12, factor=10, dropout=0.1, attention='prob', embed='fixed', freq='h',
                 activation='gelu', use_mask=True, if_output_attention=True, co_train=False, distil=True, mix=True,
                 history_steps=16, predict_steps=16, strides=16, device=torch.device('cuda:0')):
        super(Encoder, self).__init__()
        # self.global_adj = global_adj
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.en_in_dim = en_in_dim
        self.de_in_dim = de_in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.after_d_model = after_d_model
        self.n_heads = n_heads
        self.num_of_instances = num_of_instances
        self.num_attention_layers = num_attention_layers
        self.num_en_layers = num_en_layers
        self.num_de_layers = num_de_layers
        self.factor = factor
        self.kernels = kernels
        self.dropout = dropout
        self.attention = attention
        self.embed = embed
        self.freq = freq
        self.activation = activation
        self.use_mask = use_mask
        self.if_output_attention = if_output_attention
        self.co_train = co_train
        self.distil = distil
        self.mix = mix
        self.strides = strides
        self.history_steps = history_steps
        self.predict_steps = predict_steps
        self.device = device

        self.encoder_layer_stack = nn.ModuleList([EncoderLayer(num_features=self.num_features,
                                                               embed_dim=self.embed_dim,
                                                               en_in_dim=self.en_in_dim,
                                                               de_in_dim=self.de_in_dim,
                                                               out_dim=self.out_dim,
                                                               hidden_dim=self.hidden_dim,
                                                               d_model=self.d_model,
                                                               after_d_model=self.after_d_model,
                                                               n_heads=self.n_heads,
                                                               num_of_instances=self.num_of_instances,
                                                               num_attention_layers=self.num_attention_layers,
                                                               num_en_layers=self.num_en_layers,
                                                               num_de_layers=self.num_de_layers,
                                                               factor=self.factor,
                                                               kernels=self.kernels,
                                                               dropout=self.dropout,
                                                               attention=self.attention,
                                                               embed=self.embed,
                                                               freq=self.freq,
                                                               activation=self.activation,
                                                               use_mask=self.use_mask,
                                                               if_output_attention=self.if_output_attention,
                                                               co_train=self.co_train,
                                                               distil=self.distil,
                                                               mix=self.mix,
                                                               history_steps=self.history_steps,
                                                               predict_steps=self.predict_steps,
                                                               strides=self.strides)
                                                  for _ in range(self.num_en_layers)])

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, adj_in, instances_index_vm):
        """
        :param x: B T N embed_dim(d_model)
        :param adj_in:
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :return: B T N d_model
        """
        en_vm_attention_outer_list = []
        for encoder_layer in self.encoder_layer_stack:
            x, vm_attention_list = encoder_layer(x, adj_in, instances_index_vm)
            x = self.dropout_layer(x)
            en_vm_attention_outer_list.append(vm_attention_list)
        return x, en_vm_attention_outer_list
