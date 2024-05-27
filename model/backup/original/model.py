import torch
import torch.nn as nn
from model.backup.original.encoder import Encoder
from model.backup.original.decoder import Decoder
from layer.embed import DataEmbedding


class STMFormer(nn.Module):
    def __init__(self, num_features, embed_dim, en_in_dim, de_in_dim, out_dim, hidden_dim, d_model=512,
                 after_d_model=128, n_heads=16, num_of_instances=42, num_attention_layers=2, num_en_layers=6,
                 num_de_layers=6, factor=10, kernels=12, dropout=0.1, attention='prob', embed='fixed', freq='h',
                 activation='gelu', use_mask=True, if_output_attention=True, co_train=False, distil=True, mix=True,
                 history_steps=16, predict_steps=16, strides=16, device=torch.device('cuda:0')):
        super(STMFormer, self).__init__()

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

        # Encoding
        self.en_embedding = DataEmbedding(self.num_features, self.embed_dim, self.embed, self.freq, self.dropout)
        self.de_embedding = DataEmbedding(self.num_features, self.embed_dim, self.embed, self.freq, self.dropout)

        # self.input_en_linear = nn.Linear(self.embed_dim, self.en_in_dim)
        # self.input_de_linear = nn.Linear(self.embed_dim, self.de_in_dim)
        self.input_en_linear = nn.Linear(self.embed_dim, self.d_model)
        self.input_de_linear = nn.Linear(self.embed_dim, self.d_model)

        # Encoder
        self.encoder = Encoder(num_features=self.num_features,
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
                               strides=self.strides
                               )

        # Decoder
        self.decoder = Decoder(num_features=self.num_features,
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

        # Output
        self.forecast_layer = nn.Linear(self.d_model, self.num_features)
        if self.co_train:
            self.label_layer = nn.Linear(self.d_model, 2)

    def forward(self, x, y, y_label, adj_in, adj_out, instances_index_vm):
        """
        :param x: B T N F
        :param y: B T N F
        :param y_label: B T N 1
        :param adj_in: N N
        :param instances_index_vm: (List)[n1,n2,...] n1表示几个节点是在VM1上，n2表示在VM2上，。。。
        :return: B T N F, optional[Tensor(B T N 1), None]
        """
        label_output = None
        x_mark = None
        x = self.en_embedding(x, x_mark)  # B T N F -> B T N embed_dim
        # if self.embed_dim != self.en_in_dim:
        x = self.input_en_linear(x)  # B T N embed_dim -> B T N en_in_dim(d_model)
        # B T N en_in_dim(d_model) -> B T N d_model
        en_output, en_vm_attention_outer_list = self.encoder(x, adj_in, instances_index_vm)

        y_mark = None
        y = self.de_embedding(y, y_mark)  # B T N F -> B T N embed_dim
        # if self.embed_dim != self.de_in_dim:
        y = self.input_de_linear(y)  # B T N embed_dim -> B T N de_in_dim(d_model)

        # B T N d_model, B T N de_in_dim(d_model) -> B T N d_model
        de_output, _ = self.decoder(en_output, y, en_vm_attention_outer_list, adj_out, instances_index_vm)

        forecast_output = self.forecast_layer(de_output)  # B T N d_model -> B T N num_features
        if self.co_train:
            de_output, _ = self.decoder(en_output, y_label, en_vm_attention_outer_list, adj_out, instances_index_vm)
            label_output = self.label_layer(de_output)
            label_output = torch.softmax(label_output, dim=-1)

        return forecast_output, label_output
