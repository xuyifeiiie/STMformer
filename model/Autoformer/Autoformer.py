import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.embed_3d import DataEmbedding, DataEmbedding_wo_pos
from layer.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from layer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs, device):
        super(Autoformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.history_steps
        self.label_len = configs.label_steps
        self.pred_len = configs.predict_steps
        self.num_features = configs.num_features
        self.output_attention = configs.if_output_attention
        self.moving_avg = 25
        self.factor = 3

        # Decomp
        kernel_size = self.moving_avg
        self.decomp = series_decomp(kernel_size)

        self.trend_projection = nn.Linear(configs.num_features * configs.num_instances, configs.d_model)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(c_in=configs.num_features * configs.num_instances,
                                                  d_model=configs.d_model,
                                                  embed_type=configs.embed,
                                                  freq=configs.freq,
                                                  dropout=configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        correlation=AutoCorrelation(False, self.factor, attention_dropout=configs.dropout,
                                                    output_attention=configs.if_output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.hidden_dim,
                    moving_avg=self.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.num_en_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(c_in=configs.num_features * configs.num_instances,
                                                      d_model=configs.d_model,
                                                      embed_type=configs.embed,
                                                      freq=configs.freq,
                                                      dropout=configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            correlation=AutoCorrelation(True, self.factor, attention_dropout=configs.dropout,
                                                        output_attention=False),
                            d_model=configs.d_model,
                            n_heads=configs.n_heads),
                        AutoCorrelationLayer(
                            correlation=AutoCorrelation(False, self.factor, attention_dropout=configs.dropout,
                                                        output_attention=False),
                            d_model=configs.d_model,
                            n_heads=configs.n_heads),
                        d_model=configs.d_model,
                        c_out=configs.d_model,
                        d_ff=configs.hidden_dim,
                        moving_avg=self.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.num_de_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.num_features * configs.num_instances, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        # B T NC -> B NC -> B 1 NC -> B T NC
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # B T NC
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, mean.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean[:, :self.label_len, :]], dim=1)
        trend_init = self.trend_projection(trend_init)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros[:, :self.label_len, :]], dim=1)
        # enc
        enc_in = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        # dec
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                adj_out, instances_index_vm, mask=None):
        x_enc = x_enc.reshape(x_enc.shape[0], x_enc.shape[1], -1)  # B T NC
        x_dec = x_dec.reshape(x_dec.shape[0], x_dec.shape[1], -1)  # B T NC
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            dec_out = dec_out[:, -self.pred_len:, :].reshape(x_dec.shape[0], x_dec.shape[1], -1, self.num_features)
            return dec_out  # [B, T, N, C]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
