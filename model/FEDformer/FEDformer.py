import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.embed_3d import DataEmbedding
from layer.autocorrelation import AutoCorrelationLayer
from layer.fouriercorrelation import FourierBlock, FourierCrossAttention
from layer.multiwaveletcorrelation import MultiWaveletCross, MultiWaveletTransform
from layer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(self, configs, device):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super(FEDformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.history_steps
        self.label_len = configs.label_steps
        self.pred_len = configs.predict_steps
        self.num_features = configs.num_features
        self.version = 'fourier'
        self.mode_select = 'random'
        self.modes = 32
        self.moving_avg = 25

        # Decomp
        self.decomp = series_decomp(self.moving_avg)

        self.trend_projection = nn.Linear(configs.num_features * configs.num_instances, configs.d_model)

        self.enc_embedding = DataEmbedding(configs.num_features * configs.num_instances,
                                           configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.num_features * configs.num_instances,
                                           configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=self.modes,
                                                  ich=configs.d_model,
                                                  base='legendre',
                                                  activation='tanh')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len // 2 + self.pred_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len // 2 + self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=self.modes,
                                                      mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model,
                        configs.n_heads),
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model,
                        configs.n_heads),
                    cross_attention=AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model,
                        configs.n_heads),
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
            self.projection = nn.Linear(configs.d_model, configs.num_features, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.num_features, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # decomp init
        # B T NC -> B 1 NC -> B T NC
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)  # x - moving_avg, moving_avg
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean[:, :self.label_len, :]], dim=1)  # B label_len+pred_len Cin
        trend_init = self.trend_projection(trend_init)
        # (0, 0, 0, self.pred_len)前两位表示左右填充数 后两位表示上下填充数
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.label_len))
        # enc
        enc_in = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        # dec
        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_in, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        # print("trend_part.shape: {}\nseasonal_part.shape: {}".format(trend_part.shape, seasonal_part.shape))
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
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
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
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
