import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler, autocast
from metrics import metrics
from model.FEDformer.FEDformer import FEDformer
from model.Informer.Informer import Informer
from model.PatchTST.PatchTST import PatchTST
from model.STMFormer.model import STMFormer
from model.Autoformer.Autoformer import Autoformer
from model.Dlinear.DLinear import DLinear
import utils.utils
import utils.tools
from model.STSGCN.STSGCN import STSGCN
from model.TimesNet.TimesNet import TimesNet
from model.STSGT.STSGT import STSGT


class trainer(nn.Module):
    def __init__(self, settings, log, device=torch.device('cuda:0')):
        """
        Trainer
        scaler: Converter
        log: log
        lrate: Initial learning rate
        w_decay: Weight decay rate
        l_decay_rate: lr decay rate after every epoch
        adj: Local Space-time matrix
        num_features: The number of features
        embed_dim: Embedding dimension
        en_in_dim: Encoder input dimension
        de_in_dim: Decoder input dimension
        out_dims: The List of middle layer output dimension
        hidden_dims: Lists, the convolution operation dimension of each STSGCL layer in the middle
        d_model: The dimension of Query, Key, Value for self-attention
        after_d_model: The dimension of the hidden layer of MLP after multi-head self attention
        n_heads: The number of heads for calculating multi-head self attention
        num_of_instances: The number of instances
        num_attention_layers: The number of attention layers
        num_en_layers: The number of encoder layers
        num_de_layers: The number of decoder layers
        factor: The amount of self-attentions (top queries) to be selected
        dropout: Dropout Ratio after multi-head self-attention
        attention: Which attention approach to use
        embed: Which embedding approach to use
        freq:
        activation: activation function {relu, GlU}
        use_mask: Whether to use the mask matrix to optimize adj
        if_output_attention: Whether to output self-attentions or not
        max_grad_norm: gradient threshold
        lr_decay: Whether to use the initial learning rate decay strategy
        co_train: Whether to co_train
        distil: Whether to scale the encoder layer
        mix: informer
        history_steps: input time steps
        predict_steps: output time steps
        strides: local spatio-temporal graph is constructed using these time steps, the default is 12
        device: computing device
        """

        super(trainer, self).__init__()

        self.model_dict = {
            'STMFormer': STMFormer,
            'TimesNet': TimesNet,
            'FEDformer': FEDformer,
            'PatchTST': PatchTST,
            'Informer': Informer,
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'STSGT': STSGT,
            'STSGCN': STSGCN,
        }
        self.model_name_str = settings.model
        self.model_name = self.model_dict[settings.model]
        self.model = self.model_name(settings, device).cuda()

        self.iters_to_accumulate = settings.iters_to_accumulate
        self.batch_size = settings.batch_size
        warmup_steps = int(10000 / (self.batch_size * self.iters_to_accumulate))

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # self.model.to(device)
        # print("Is Model in GPU?", next(self.model.parameters()).is_cuda)

        self.optimizer = optim.Adam(self.model.parameters(), lr=settings.learning_rate,
                                    weight_decay=settings.weight_decay)

        self.lr_decay = settings.lr_decay
        if self.lr_decay:
            utils.utils.log_string(log, 'Applying Lambda Learning rate decay.')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                                  lr_lambda=lambda epoch:
                                                                  settings.lr_decay_rate ** epoch)
        self.if_warmup = settings.if_warmup
        if self.if_warmup:
            self.warmup_scheduler = utils.tools.WarmupLR(self.optimizer, warmup_steps=warmup_steps,
                                                         iters_to_accumulate=self.iters_to_accumulate,
                                                         init_lr=settings.learning_rate)

        self.scaler = GradScaler()

        # Loss Function
        # self.loss = metrics.masked_mse
        self.loss = torch.nn.MSELoss()
        # self.loss = torch.nn.HuberLoss()
        self.loss.cuda()

        # self.x_scaler = x_scaler
        # self.y_scaler = y_scaler
        self.clip = settings.max_grad_norm

        self.if_amp = settings.if_amp

        for name, parameters in self.model.named_parameters():
            utils.utils.log_string(log, "{:<100} ： {:<15}".format(name, parameters.numel()))

        utils.utils.log_string(log, "Model trainable parameters: {:,}".format(utils.utils.count_parameters(self.model)))

        utils.utils.init_seed(seed=42)

    def train_model(self, x_enc, x_dec, x_mark_enc, x_mark_dec, label, adj_in, adj_out, instances_index_vm, loss_epoch,
                    ix, iters_to_accumulate, len_train, learning_rate, epoch):
        """
        input_data: B, T, N, F
        real_val: B, T, N, F
        """

        self.model.train()

        if self.if_amp:
            with autocast():
                if self.model_name_str == 'STMFormer':
                    forecast_predict, label_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                                                 adj_out, instances_index_vm, mask=None)

                elif self.model_name_str == 'MSDR':
                    forecast_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                                  adj_out, instances_index_vm, len_train, epoch, mask=None)
                else:
                    forecast_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                                  adj_out, instances_index_vm, mask=None)
                loss = self.loss(forecast_predict, x_dec)
                loss.requires_grad_(True)

                loss_epoch += loss.item() * len(x_enc)
                loss = loss / iters_to_accumulate

            # loss.backward()
            self.scaler.scale(loss).backward()

            # self.optimizer.step()
            if ((ix + 1) % iters_to_accumulate == 0) or ((ix + 1) == len_train):
                self.scaler.unscale_(self.optimizer)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.if_warmup and (self.warmup_scheduler.stop_flag == False):
                    self.warmup_scheduler.step()
                    # print(self.warmup_scheduler.get_last_lr())

        else:
            if self.model_name_str == 'STMFormer':
                forecast_predict, label_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                                             adj_out, instances_index_vm, mask=None)

            elif self.model_name_str == 'MSDR':
                forecast_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                              adj_out, instances_index_vm, len_train, epoch, mask=None)
            else:
                forecast_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in,
                                              adj_out, instances_index_vm, mask=None)
            loss = self.loss(forecast_predict, x_dec)
            loss.requires_grad_(True)

            loss_epoch += loss.item() * len(x_enc)
            loss = loss / iters_to_accumulate

            loss_start_time = time.time()
            loss.backward()
            loss_end_time = time.time()
            print("Loss backward time is {}s.".format(loss_end_time - loss_start_time))

            if ((ix + 1) % iters_to_accumulate == 0) or ((ix + 1) == len_train):
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.if_warmup and (self.warmup_scheduler.stop_flag == False):
                    self.warmup_scheduler.step()

        mae, mse, rmse, rmsle, mape, mspe = metrics.predict_metric(forecast_predict, x_dec)

        return loss.item(), mae, mse, rmse, rmsle, mape, mspe, loss_epoch

    def eval_model(self, x_enc, x_dec, x_mark_enc, x_mark_dec, label, adj_in, adj_out, instances_index_vm, mask=None):
        """
        input_data: B, T, N, F
        real_val: B, T, N , F
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_name_str == 'STMFormer':
                forecast_predict, _ = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in, adj_out,
                                                 instances_index_vm)
                loss = self.loss(forecast_predict, x_dec)
            else:
                forecast_predict = self.model(x_enc, x_dec, x_mark_enc, x_mark_dec, adj_in, adj_out,
                                              instances_index_vm)
                loss = self.loss(forecast_predict, x_dec)

        mae, mse, rmse, rmsle, mape, mspe = metrics.predict_metric(forecast_predict, x_dec)

        return loss.item(), mae, mse, rmse, rmsle, mape, mspe

    def test_model(self, x_enc, x_dec, y_gt, x_mark_enc, x_mark_dec, label_gt, adj_in, adj_out, instances_index_vm,
                   mask=None):
        self.model.eval()
        with torch.no_grad():
            if self.model_name_str == 'STMFormer':
                forecast_predict, y_label_pred = self.model(x_enc, y_gt, x_mark_enc, x_mark_dec, adj_in,
                                                            adj_out, instances_index_vm)
            else:
                forecast_predict = self.model(x_enc, y_gt, x_mark_enc, x_mark_dec, adj_in,
                                              adj_out, instances_index_vm)

            mae, mse, rmse, rmsle, mape, mspe = metrics.predict_metric(forecast_predict, y_gt)

        return mae, mse, rmse, rmsle, mape, mspe, forecast_predict
