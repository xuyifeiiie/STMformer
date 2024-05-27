import math
import sys
import os

import torch

sys.path.append(os.path.dirname(sys.path[0]))
import ast
import time
import wandb
import argparse
import configparser
import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from engines.forecasting_engine import trainer
from utils.tools import EarlyStopping, Settings
from utils.utils import *

TASK = 'short_term_forecast'
DATASET = 'TrainTicket_' + TASK

config_file = '../config/{}.conf'.format(DATASET)
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--no_cuda', action="store_true",
                    help="NO GPU")
parser.add_argument('--model', type=str, default=config['data']['model'],
                    help='The wanted model to run')
parser.add_argument('--task_name', type=str, default=config['data']['task_name'],
                    help='Task name')
parser.add_argument('--raw_datapath', type=str, default=config['data']['raw_datapath'],
                    help='raw data path')
parser.add_argument('--processed_datapath', type=str, default=config['data']['processed_datapath'],
                    help='processed data path')
parser.add_argument("--sample_rate", type=int, default=5,
                    help="the sample rate of collecting metrics", )
# parser.add_argument('--id_map_file', type=str, default=config['data']['id_map_file'],
#                     help='Id map index file')
# parser.add_argument('--nodes_connect_file', type=str, default=config['data']['nodes_connect_file'],
#                     help='Node Connect File')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'],
                    help="Training Batch Size")
parser.add_argument('--valid_batch_size', type=int, default=config['data']['valid_batch_size'],
                    help="Validation Batch Size")
parser.add_argument('--test_batch_size', type=int, default=config['test']['test_batch_size'],
                    help="Test Batch Size")
parser.add_argument('--if_shuffle', type=eval, default=config['data']['if_shuffle'],
                    help="Whether to shuffle data")
parser.add_argument('--if_scale', type=eval, default=config['data']['if_scale'],
                    help="Whether to scale the raw data")
parser.add_argument('--if_test', type=eval, default=config['data']['if_test'],
                    help="Loading small data when test the script")
parser.add_argument('--num_workers', type=int, default=config['data']['num_workers'],
                    help="num_workers of dataloader")
# parser.add_argument('--prefetch_factor', type=int, default=config['data']['prefetch_factor'],
#                     help="prefetch_factor of dataloader")
parser.add_argument('--num_class', type=int, default=config['data']['num_class'],
                    help="number of class for dataset")
# parser.add_argument('--fill_zeros', type=eval, default=config['data']['fill_zeros'],
#                     help="whether to fill zeros in data with average")

parser.add_argument('--num_features', type=int, default=config['model']['num_features'],
                    help='The number of features')
parser.add_argument('--embed_dim', type=int, default=config['model']['embed_dim'],
                    help='embed dimension')
parser.add_argument('--en_in_dim', type=int, default=config['model']['en_in_dim'],
                    help='Encoder input dimension')
parser.add_argument('--de_in_dim', type=int, default=config['model']['de_in_dim'],
                    help='Decoder input dimension')
parser.add_argument('--out_layer_dim', type=int, default=config['model']['out_layer_dim'],
                    help='Output module middle layer dimension')
parser.add_argument('--forward_expansion', type=int, default=config['model']['forward_expansion'],
                    help='Hidden Layer Dimension for the ST Synchronous Transformer')
parser.add_argument("--temporal_emb", type=eval, default=config['model']['temporal_emb'],
                    help="Whether to use temporal embedding vector")
parser.add_argument("--spatial_emb", type=eval, default=config['model']['spatial_emb'],
                    help="Whether to use spatial embedding vector")
parser.add_argument('--out_dims', type=list, default=ast.literal_eval(config['model']['out_dims']),
                    help='distil dimensions')
parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'],
                    help='The dimension of layer in the middle')
parser.add_argument('--hidden_dims', type=list, default=ast.literal_eval(config['model']['hidden_dims']),
                    help='Convolution operation dimension of each STSGCL layer in the middle')
parser.add_argument('--d_model', type=int, default=config['model']['d_model'],
                    help='Embedding dimension for the ST Synchronous Transformer')
parser.add_argument('--after_d_model', type=int, default=config['model']['after_d_model'],
                    help='multiheads output dimension')
parser.add_argument('--n_heads', type=int, default=config['model']['n_heads'],
                    help='Number of heads for the Multi-Head Attention')
parser.add_argument('--num_instances', type=int, default=config['model']['num_instances'],
                    help='The number of instances')
parser.add_argument('--num_attention_layers', type=int, default=config['model']['num_attention_layers'],
                    help='Number of attention layers')
parser.add_argument('--num_en_layers', type=int, default=config['model']['num_en_layers'],
                    help='Number of encoder layers')
parser.add_argument('--num_de_layers', type=int, default=config['model']['num_de_layers'],
                    help='Number of decoder layers')
parser.add_argument("--factor", type=int, default=config['model']['factor'],
                    help="The amount of self-attentions needed")
parser.add_argument("--kernels", type=int, default=config['model']['kernels'],
                    help="The number of kernels for TimesNet Block")
parser.add_argument("--nb_random_features", type=int, default=config['model']['nb_random_features'],
                    help="The number of random features for CrossRelation Block")
parser.add_argument("--nb_gumbel_sample", type=int, default=config['model']['nb_gumbel_sample'],
                    help="The number of gumbel samples for CrossRelation Block")
parser.add_argument('--tau', type=float, default=config['model']['tau'],
                    help='The temperature of gumbel softmax')
parser.add_argument("--rb_order", type=int, default=config['model']['rb_order'],
                    help="The layer number of relation bias")
parser.add_argument("--rb_trans", type=str, default=config['model']['rb_trans'],
                    help="The layer activation of relation bias")
parser.add_argument("--use_edge_loss", type=eval, default=config['model']['use_edge_loss'],
                    help="Whether to use edge loss to regularize the model")
parser.add_argument('--dropout', type=float, default=config['model']['dropout'],
                    help='dropout for the ST Synchronous Transformer')
parser.add_argument("--attention", type=str, default=config['model']['attention'],
                    help="attention type")
parser.add_argument("--embed", type=str, default=config['model']['embed'],
                    help="embedding type")
parser.add_argument("--freq", type=str, default=config['model']['freq'],
                    help="freq type")
parser.add_argument("--activation", type=str, default=config['model']['activation'],
                    help="Activation Function {ReLU, GLU}")
parser.add_argument("--use_mask", type=eval, default=config['model']['use_mask'],
                    help="Whether to use the mask matrix to optimize adj")
parser.add_argument("--if_output_attention", type=eval, default=config['model']['if_output_attention'],
                    help="Whether to output the self-attentions or not")
parser.add_argument("--decomp", type=eval, default=config['model']['decomp'],
                    help="Whether to decomp series")
parser.add_argument("--decomp_kernels", type=int, default=config['model']['decomp_kernels'],
                    help="The kernels for the single series decomp")
parser.add_argument("--multidecomp", type=eval, default=config['model']['multidecomp'],
                    help="Whether to multi decomp series")
parser.add_argument('--multidecomp_kernels', type=list,
                    default=ast.literal_eval(config['model']['multidecomp_kernels']),
                    help='The kernels list for the multi series decomp')
parser.add_argument("--decomp_stride", type=int, default=config['model']['decomp_stride'],
                    help="The stride for the docomp stride")
parser.add_argument("--if_patch", type=eval, default=config['model']['if_patch'],
                    help="Whether to use patch embedding in model")
parser.add_argument("--distil", type=eval, default=config['model']['distil'],
                    help="Whether to distil in encoder")
parser.add_argument("--mix", type=eval, default=config['model']['mix'],
                    help="Whether to mix")
parser.add_argument("--if_wavelet", type=eval, default=config['model']['if_wavelet'],
                    help="Whether to use wavelet")
parser.add_argument("--history_steps", type=int, default=config['model']['history_steps'],
                    help="The discrete time series of each sample input")
parser.add_argument("--predict_steps", type=int, default=config['model']['predict_steps'],
                    help="The discrete time series of each sample output (forecast)")
parser.add_argument("--label_steps", type=int, default=config['model']['label_steps'],
                    help="The discrete time series of each sample label input (forecast)")
parser.add_argument("--strides", type=int, default=config['model']['strides'],
                    help="The step size of the sliding window, "
                         "the local spatio-temporal graph is constructed using several time steps, "
                         "the default is 12")

parser.add_argument('--if_amp', type=eval, default=config['train']['if_amp'],
                    help="If use amp")
parser.add_argument('--seed', type=int, default=config['train']['seed'],
                    help='Seed Settings')
parser.add_argument('--iters_to_accumulate', type=int, default=config['train']['iters_to_accumulate'],
                    help='gradients accumulate iterations')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'],
                    help="Initial Learning Rate")
parser.add_argument("--weight_decay", type=float, default=config['train']['weight_decay'],
                    help="Weight Decay Rate")
parser.add_argument("--lr_decay", type=eval, default=config['train']['lr_decay'],
                    help="Whether to enable the initial learning rate decay strategy")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'],
                    help="Learning rate decay rate")
parser.add_argument('--if_warmup', type=eval, default=config['train']['if_warmup'],
                    help="If warmup")
parser.add_argument("--warmup_steps", type=int, default=config['train']['warmup_steps'],
                    help="Warmup steps")
parser.add_argument("--warmup_lr", type=float, default=config['train']['warmup_lr'],
                    help="Warmup learning rate")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'],
                    help="Number of training epochs")
parser.add_argument('--co_train', type=eval, default=config['train']['co_train'],
                    help="Whether to train the model in anomaly detection and forecasting")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'],
                    help='Print losses and metrics after print_every iterations')
parser.add_argument('--save', type=str, default=config['train']['save'],
                    help='Save Path')
parser.add_argument('--save_limit', type=int, default=config['train']['save_limit'],
                    help='The limit of the model to save')
parser.add_argument('--save_loss', type=str, default=config['train']['save_loss'],
                    help='Save Loss Path')
parser.add_argument('--expid', type=int, default=config['train']['expid'],
                    help='Experiment ID')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'],
                    help="Gradient Threshold")
parser.add_argument('--patience', type=int, default=config['train']['patience'],
                    help='Patience during training')
parser.add_argument('--log_file', default=config['train']['log_file'],
                    help='log file')
parser.add_argument('--if_wandb', type=eval, default=config['train']['if_wandb'],
                    help="If upload result to wandb")

args = parser.parse_args()
args_dict = vars(args)
settings = Settings(args_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_path = args.log_file + TASK + '/' + str(args.sample_rate)
if os.path.exists(log_path):
    pass
else:
    os.makedirs(log_path)
log_file_path = log_path + '/' + args.model + '_train.log'
log = open(log_file_path, 'w')
log_string(log, str(args))


def main():
    # task_dataset_path = args.processed_datapath + args.task_name
    task_dataset_path = args.raw_datapath
    print("Dataset loading from: {}".format(task_dataset_path))
    dataset = prepare_and_load_dataset(dataset_path=task_dataset_path,
                                       sample_rate=args.sample_rate,
                                       if_scale=args.if_scale,
                                       if_test=args.if_test)

    if args.if_scale:
        scaler = dataset['scaler']

    log_string(log, 'Experiment Date: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    log_string(log, 'Loading Data ...')

    log_string(log, f'x_train: {dataset["train"].data.x_train_shape}\t\t '
                    f'y_train: {dataset["train"].data.y_train_shape}')
    log_string(log, f'x_val:   {dataset["val"].data.x_val_shape}\t\t'
                    f'y_val:   {dataset["val"].data.y_val_shape}')
    log_string(log, f'x_test:   {dataset["test"].data.x_test_shape}\t\t'
                    f'y_test:   {dataset["test"].data.y_test_shape}')
    log_string(log, f'adj_train: {dataset["train"].data.adj_train_shape}\t\t')
    log_string(log, f'adj_val:   {dataset["val"].data.adj_val_shape}\t\t')
    log_string(log, f'adj_test:   {dataset["test"].data.adj_test_shape}\t\t')
    log_string(log, f'label_train: {dataset["train"].data.label_train_shape}\t\t')
    log_string(log, f'label_val:   {dataset["val"].data.label_val_shape}\t\t')
    log_string(log, f'label_test:   {dataset["test"].data.label_test_shape}\t\t')
    log_string(log, f'instances_count_train: {dataset["train"].data.instances_count_train_shape}\t\t')
    log_string(log, f'instances_count_val:   {dataset["val"].data.instances_count_val_shape}\t\t')
    log_string(log, f'instances_count_test:   {dataset["test"].data.instances_count_test_shape}\t\t')

    log_string(log, 'Data Loaded !!')

    engine = trainer(settings, log, device)
    # Start Training
    log_string(log, 'Training Model ...')
    his_loss = []
    val_time = []
    train_time = []

    wait = 0
    val_mae_min = float('inf')
    best_model_wts = None

    train_loss = []
    train_mae = []
    train_mse = []
    train_rmse = []
    train_rmsle = []
    train_mape = []
    train_mspe = []

    valid_loss = []
    valid_mae = []
    valid_mse = []
    valid_rmse = []
    valid_rmsle = []
    valid_mape = []
    valid_mspe = []

    test_loss = []
    test_mae = []
    test_mse = []
    test_rmse = []
    test_rmsle = []
    test_mape = []
    test_mspe = []

    dataloader_train = DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=args.if_shuffle,
                                  num_workers=args.num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset['val'], batch_size=args.valid_batch_size, shuffle=args.if_shuffle,
                                num_workers=args.num_workers, pin_memory=True)
    dataloader_test = DataLoader(dataset['test'], batch_size=args.test_batch_size, shuffle=args.if_shuffle,
                                 num_workers=args.num_workers, pin_memory=True)
    if args.if_wandb:
        wandb.init(
            config={
                "architecture": 'transformer',
                'epoch': args.epochs,
            },
            project="STMFormer",
            entity='xyf123',
            name='{}-{}'.format(args.model, args.expid),
            notes=args.task_name,
            job_type="training",
            reinit=True)

    early_stopping = EarlyStopping(expid=args.expid, patience=args.patience, verbose=True, save_limit=args.save_limit)
    loss_epoch_list = []
    nan_count = 0
    inf_count = 0
    for epoch in tqdm.tqdm(range(1, args.epochs + 1), desc="Epoch Bar", dynamic_ncols=True):

        train_loss.clear()
        train_mae.clear()
        train_mse.clear()
        train_rmse.clear()
        train_rmsle.clear()
        train_mape.clear()
        train_mspe.clear()

        t1 = time.time()

        iters_to_accumulate = args.iters_to_accumulate
        len_train = len(dataloader_train)  # num_samples/batch_size
        lr_decay = args.lr_decay

        for ix, data in tqdm.tqdm(enumerate(dataloader_train), desc="Iter Bar", dynamic_ncols=True):
            loss_epoch = 0
            x, y, adj_in, adj_out, l2, l3, instances_index_vm, l1 = data

            x_train = torch.Tensor(x).to(device)  # [B, T, N, C]
            y_train = torch.Tensor(y).to(device)  # [B, T, N, C]
            adj_in = torch.Tensor(adj_in).to(device)
            adj_out = torch.Tensor(adj_out).to(device)
            instances_index_vm = torch.Tensor(instances_index_vm).to(device)

            x_train = x_train.clone().detach()  # [B, T, N, C]
            y_train = y_train.clone().detach()  # [B, T, N, C]
            adj_in = adj_in.clone().detach()
            adj_out = adj_out.clone().detach()
            instances_index_vm = instances_index_vm.clone().detach()

            loss, tmae, tmse, trmse, trmsle, tmape, tmspe, loss_epoch = \
                engine.train_model(
                    x_enc=x_train, x_dec=y_train,
                    x_mark_enc=None, x_mark_dec=None,
                    label=l2,
                    adj_in=adj_in, adj_out=adj_out,
                    instances_index_vm=instances_index_vm, ix=ix,
                    iters_to_accumulate=iters_to_accumulate,
                    len_train=len_train, loss_epoch=loss_epoch,
                    learning_rate=args.learning_rate,
                    epoch=epoch
                )
            if math.isnan(loss) or math.isnan(tmse):
                nan_count += 1
                logs = 'nan occurs {:03d} times.'
                log_string(log, logs.format(nan_count))
            elif loss == float('inf') or tmae == float('inf') or tmse == float('inf'):
                inf_count += 1
                logs = 'inf occurs {:03d} times.'
                log_string(log, logs.format(inf_count))
            else:
                loss_epoch_list.append(loss_epoch)
                train_loss.append(loss)
                train_mae.append(tmae)
                train_mse.append(tmse)
                train_rmse.append(trmse)
                train_rmsle.append(trmsle)
                train_mape.append(tmape)
                train_mspe.append(tmspe)
                if ix % args.print_every == 0:
                    logs = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MSE: {:.4f},\n' \
                           'Train RMSE: {:.4f}, Train RMSLE: {:.4f}, Train MAPE: {:.4f}, Train MSPE: {:.4f}, lr: {}.'
                    log_string(log, logs.format(ix, train_loss[-1], train_mae[-1], train_mse[-1], train_rmse[-1],
                                                train_rmsle[-1], train_mape[-1], train_mspe[-1],
                                                engine.optimizer.param_groups[0]['lr']))
                    print('nan occurs {:03d} times.'.format(nan_count))
                    print('inf occurs {:03d} times.'.format(inf_count))
                    if args.if_wandb:
                        wandb.log({
                            'Iter': ix,
                            'Train Loss': train_loss[-1],
                            'Train MAE': train_mae[-1],
                            'Train MSE': train_mse[-1],
                            'Train RMSE': train_rmse[-1],
                            'Train RMSLE': train_rmsle[-1],
                            'Train MAPE': train_mape[-1],
                            'Train MSPE': train_mspe[-1],
                            'lr': engine.optimizer.param_groups[0]['lr']
                        })

        if lr_decay:
            engine.lr_scheduler.step()
            print(engine.lr_scheduler.get_last_lr())

        t2 = time.time()
        train_time.append(t2 - t1)
        logs = '\nEpoch: {:03d}, Epoch Train Loss: {:.4f}'
        print(logs.format(epoch, loss_epoch_list[epoch - 1] / len_train), flush=True)
        if args.if_wandb:
            wandb.log({
                'Epoch': epoch,
                'Epoch Train Loss': loss_epoch_list[epoch - 1] / len_train,
            })

        valid_loss.clear()
        valid_mae.clear()
        valid_mse.clear()
        valid_rmse.clear()
        valid_rmsle.clear()
        valid_mape.clear()
        valid_mspe.clear()

        s1 = time.time()

        print("Valid phrase of epoch {} begin...".format(epoch))
        for ix, data in tqdm.tqdm(enumerate(dataloader_val), desc="Iter Bar", dynamic_ncols=True):
            x, y, adj_in, adj_out, l2, l3, instances_index_vm, l1 = data

            x_val = torch.Tensor(x).to(device)  # [B, T, N, C]
            y_val = torch.Tensor(y).to(device)  # [B, T, N, C]
            adj_in = torch.Tensor(adj_in).to(device)
            adj_out = torch.Tensor(adj_out).to(device)
            instances_index_vm = torch.Tensor(instances_index_vm).to(device)

            x_val = x_val.clone().detach()  # [B, T, N, C]
            y_val = y_val.clone().detach()  # [B, T, N, C]
            adj_in = adj_in.clone().detach()
            adj_out = adj_out.clone().detach()
            instances_index_vm = instances_index_vm.clone().detach()

            vloss, vmae, vmse, vrmse, vrmsle, vmape, vmspe = engine.eval_model(x_enc=x_val, x_dec=y_val,
                                                                               x_mark_enc=None, x_mark_dec=None,
                                                                               label=l2,
                                                                               adj_in=adj_in, adj_out=adj_out,
                                                                               instances_index_vm=instances_index_vm)

            valid_loss.append(vloss)
            valid_mae.append(vmae)
            valid_mse.append(vmse)
            valid_rmse.append(vrmse)
            valid_rmsle.append(vrmsle)
            valid_mape.append(vmape)
            valid_mspe.append(vmspe)

            logs = 'Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MSE: {:.4f},\n' \
                   'Valid RMSE: {:.4f}, Valid RMSLE: {:.4f}, Valid MAPE: {:.4f}, Valid MSPE: {:.4f}'
            log_string(log,
                       logs.format(epoch, valid_loss[-1], valid_mae[-1], valid_mse[-1], valid_rmse[-1], valid_rmsle[-1],
                                   valid_mape[-1], valid_mspe[-1]))
            if args.if_wandb:
                wandb.log({
                    'Valid Epoch': epoch,
                    'Valid Loss': valid_loss[-1],
                    'Valid MAE': valid_mae[-1],
                    'Valid MSE': valid_mse[-1],
                    'Valid RMSE': valid_rmse[-1],
                    'Valid RMSLE': valid_rmsle[-1],
                    'Valid MAPE': valid_mape[-1],
                    'Valid MSPE': valid_mspe[-1],
                })

        s2 = time.time()
        logs = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        log_string(log, logs.format(epoch, (s2 - s1)))


        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_mse = np.mean(train_mse)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_rmsle = np.mean(train_rmsle)
        mtrain_mape = np.mean(train_mape)
        mtrain_mspe = np.mean(train_mspe)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mse = np.mean(valid_mse)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_rmsle = np.mean(valid_rmsle)
        mvalid_mape = np.mean(valid_mape)
        mvalid_mspe = np.mean(valid_mspe)
        his_loss.append(mvalid_loss)

        logs = '\nEpoch: {:03d}, Averge metrics:\n ' \
               'Train Loss: {:.4f}, Train MAE: {:.4f}, Train MSE: {:.4f}, ' \
               'Train RMSE: {:.4f}, Train RMSLE: {:.4f}, Train MAPE: {:.4f}, Train MSPE: {:.4f}\n' \
               'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MSE: {:.4f}, ' \
               'Valid RMSE: {:.4f}, Valid RMSLE: {:.4f}, Valid MAPE: {:.4f}, Valid MSPE: {:.4f}\n' \
               'Training Time: {:.4f}/epoch'
        log_string(log, logs.format(epoch,
                                    mtrain_loss, mtrain_mae, mtrain_mse,
                                    mtrain_rmse, mtrain_rmsle, mtrain_mape, mtrain_mspe,
                                    mvalid_loss, mvalid_mae, mvalid_mse,
                                    mvalid_rmse, mvalid_rmsle, mvalid_mape, mvalid_mspe,
                                    (t2 - t1)))

        save_path = args.save + TASK + '/' + args.model + '/' + str(args.sample_rate)
        save_loss_path = args.save_loss + TASK + '/' + str(args.sample_rate)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_loss_path):
            os.makedirs(save_loss_path)

        early_stopping(mvalid_mae, engine.model, save_path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        save_loss_path = args.save_loss + TASK + '/' + str(args.sample_rate)
        np.save(f'{save_loss_path}' + '/' + args.model + '_history_loss' + f'_{args.expid}', his_loss)

    log_string(log, 'Training Completed ...')
    log_string(log, "The Validation MAE of the best model is " + str(round(early_stopping.val_loss_min, 3)))

    log_string(log, "Average Epoch Training Loss: {:.4f}".format(np.mean(loss_epoch_list)))
    log_string(log, "Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string(log, "Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))

    # Test
    log_string(log, 'Testing Model ...')
    save_path = args.save + TASK + '/' + args.model + '/' + str(args.sample_rate)
    engine.model.load_state_dict(
        torch.load(save_path + "/exp_" + str(args.expid) + "_" +
                   str(round(early_stopping.val_loss_min, 3)) + "_best_model.pth"))

    s1 = time.time()
    for ix, data in enumerate(dataloader_test):
        x, y, adj_in, adj_out, l2, l3, instances_index_vm, l1 = data

        x_test = torch.Tensor(x).to(device)  # [B, T, N, C]
        y_test = torch.Tensor(y).to(device)
        adj_in = torch.Tensor(adj_in).to(device)
        instances_index_vm = torch.Tensor(instances_index_vm).to(device)

        B, T, N, C = x_test.shape
        y_zero = torch.zeros((B, T, N, C)).float().to(device)
        adj_out_zero = torch.zeros((B, T, N, N)).float().to(device)
        y_label_zero = torch.zeros((B, T, N, 1)).float().to(device)
        # [B, T, N, C]
        ttmae, ttmse, ttrmse, ttrmsle, ttmape, ttmspe, _ = \
            engine.test_model(x_enc=x_test, x_dec=y_zero, y_gt=y_test, x_mark_enc=None,
                              x_mark_dec=None, label_gt=l2, adj_in=adj_in, adj_out=adj_out_zero,
                              instances_index_vm=instances_index_vm, mask=None)

        test_mae.append(ttmae)
        test_mse.append(ttmse)
        test_rmse.append(ttrmse)
        test_rmsle.append(ttrmsle)
        test_mape.append(ttmape)
        test_mspe.append(ttmspe)

    s2 = time.time()
    logs = 'Test Inference Time: {:.4f} secs'
    log_string(log, logs.format((s2 - s1)))

    logs = 'On average over {} horizons, Test MAE: {:.5f}, Test MSE: {:.6f}, Test RMSE: {:.5f}, Test RMSLE: {:.5f}, ' \
           'Test MAPE: {:.4f}, Test MSPE: {:.4f},'
    log_string(log, logs.format(args.history_steps, np.mean(test_mae), np.mean(test_mse), np.mean(test_rmse),
                                np.mean(test_rmsle), np.mean(test_mape), np.mean(test_mspe)))
    if args.if_wandb:
        wandb.log({
            'Test MAE': np.mean(test_mae),
            'Test MSE': np.mean(test_mse),
            'Test RMSE': np.mean(test_rmse),
            'Test RMSLE': np.mean(test_rmsle),
            'Test MAPE': np.mean(test_mape),
            'Test MSPE': np.mean(test_mspe)
        })
        wandb.finish()
    log_string(log, 'Testing Completed ...')


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()

    log_string(log, 'total time: %.2fhours' % ((end - start) / 3600))
    log.close()
