import json
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from torch import optim
from torch.optim.lr_scheduler import LRScheduler

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class WarmupLR(LRScheduler):
    def __init__(self, optimizer, warmup_steps, iters_to_accumulate, init_lr=1e-4, last_epoch=-1):
        """
        optimizer: 优化器对象
        warmup_steps: 学习率线性增加的步数
        gamma: 学习率下降系数
        last_epoch: 当前训练轮数
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.iters_to_accumulate = iters_to_accumulate
        self.init_lr = init_lr
        self.real_iters = self.warmup_steps * self.iters_to_accumulate
        self.stop_flag = False
        # self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 学习率线性增加
            if self.last_epoch == 0:
                print("Warm up beginning...")
                return [base_lr * (self.last_epoch + 1) / self.real_iters for base_lr in self.base_lrs]
            else:
                return [base_lr * self.iters_to_accumulate * self.last_epoch / self.real_iters for base_lr in
                        self.base_lrs]
        else:
            # 学习率按指数衰减
            # return [base_lr * math.exp(-(self.last_epoch - self.warmup_steps + 1) * self.gamma) for base_lr in
            #         self.base_lrs]
            self.stop_flag = True
            print("Warm up finished!")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr
            return [self.init_lr]


class EarlyStopping:
    def __init__(self, expid, patience=7, verbose=False, delta=0, save_limit=2):
        self.expid = expid
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_limit = save_limit

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        files, subfolders = self.get_subfiles_subfolders(path)
        for file in files:
            expid_str = file.split('_')[-4]
            if expid_str != str(self.expid):
                files.remove(file)
        if len(files) >= self.save_limit:
            sort_files = sorted(files)
            print("The number of saved model has exceeded the limit.")
            for file in sort_files[(-self.save_limit + 1):]:
                print("Begin deleting")
                os.remove(file)
                print("{} has been deleted".format(file))
        else:
            pass
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        torch.save(model.state_dict(),
                   path + "/exp_" + str(self.expid) + "_" + str(round(self.val_loss_min, 3)) + "_best_model.pth")

    def get_subfiles_subfolders(self, folder_path):
        """
        Get subfile and subfolders of a directory
        :param folder_path: directory path
        :return: subfile and subfolders path list
        """
        files = []
        subfolders = []
        abs_path = os.path.abspath(folder_path)
        for file in os.listdir(abs_path):
            file_path = os.path.join(abs_path, file)
            if os.path.isdir(file_path):
                subfolders.append(file_path)
            else:
                files.append(file_path)
        return files, subfolders


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def visual(history, groundtruth, predicts=None, name='./plot/test.pdf', history_steps=64, node="", feature=""):
    """
    Results visualization
    """
    gt = torch.concat([history, groundtruth], dim=0).cpu()
    pred = torch.concat([history, predicts], dim=0).cpu()
    plt.figure()
    if predicts is not None:
        plt.plot(pred, label='Prediction', linewidth=2)
    plt.plot(gt, label='GroundTruth', linewidth=2)
    plt.axvline(x=history_steps, color='red', linestyle='--')
    plt.legend()
    plt.title("{} of {}".format(feature, node))
    plt.savefig(name, bbox_inches='tight')
    str = name.split('/')
    picname = str[-1]
    wandb.log({picname: plt})


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


class Settings:
    def __init__(self, args_dict):
        if isinstance(args_dict, dict):
            for k in args_dict.keys():
                setattr(self, k, args_dict[k])


def adj_to_patch_edge_index(adj, patch_len, num_patch, stride):
    """

    :param stride:
    :param num_patch:
    :param patch_len:
    :param adj: B T N N
    :return: B 2 E
    """
    B, T, N, N = adj.shape
    edge_b_list = []
    for b in range(B):
        edge_patchs_list = []
        unique_patch_edges_set = set()
        for t in range(num_patch):
            edge_one_patch_list = []
            unique_edges_set = set()
            for i in range(patch_len):
                adj_t = adj[b:b + 1, t:t + 1, :, :].squeeze(0).squeeze(0)  # N N
                edge_index_t = torch.nonzero(adj_t, as_tuple=False).t()  # 2 N
                edge_one_patch_list.append(edge_index_t)
            for eop in edge_one_patch_list:
                unique_edges_set |= set(map(tuple, eop.numpy()))
            edge_index_patch = torch.tensor(list(unique_edges_set)).t().contiguous()
            edge_patchs_list.append(edge_index_patch)

        for ep in edge_patchs_list:
            unique_patch_edges_set |= set(map(tuple, ep.numpy()))
        edge_index_patchs = torch.tensor(list(unique_patch_edges_set)).t().contiguous()
        edge_b_list.append(edge_index_patchs)

    patch_edge_index = torch.cat(edge_b_list, dim=0)
    return patch_edge_index


def adj_to_edge_index(adj):
    """

    :param adj: N N
    :return: edge_index 2 E
    """
    edge_index = torch.nonzero(adj, as_tuple=False).t()
    return edge_index
